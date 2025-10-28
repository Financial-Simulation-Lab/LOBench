"""
Created on : 2024-08-06
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/model/new_lob_ae.py
Description: NewLOB_AE Model
---
Updated    : 
---
Todo       : 
"""

import comet_ml
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data 
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.utilities.model_summary import ModelSummary as nofitModelSummary
# from metric import NewLoss, PriceLoss, VolumeLoss, WeightedPriceLoss, WeightedVolumeLoss, MSELoss
from metric import * 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import ExpConfigManager as ECM
from data.data_prepare import create_dataloaders_from_files
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=100,batch_size=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(batch_size,max_len,-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0),:]
        return x

  
class SingleLOB_AE(L.LightningModule):
    def __init__(self, 
                 batch_size=256,
                 d_model=256, 
                 max_len=100,
                 in_features=40,
                 lr=1e-4,
                 trans_embed_size=256,
                 multi_head_num=2,
                 trans_encoder_layers=6,
                 trans_decoder_layers=6,
                 feed_forward_dim=512,
                 optimizer_name='Adam',
                 alpha=0.5,
                 loss_func=F.mse_loss,
                 exp_metrics = ['mse_loss'],
                 log_freq= 50,
                 mask_k = 5,
                 **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.max_len = max_len
        self.in_features = in_features
        self.price_lr = lr
        self.volume_lr = lr 
        self.trans_embed_size = trans_embed_size
        self.multi_head_num = multi_head_num
        self.trans_encoder_layers = trans_encoder_layers
        self.trans_decoder_layers = trans_decoder_layers
        self.feed_forward_dim = feed_forward_dim
        self.optimizer_name = optimizer_name
        self.alpha = alpha
        
        
        self.mask = nn.Parameter(torch.ones(self.in_features, dtype=torch.bool), requires_grad=False)
        
        quarter = self.in_features // 4
        self.mask[quarter - mask_k : quarter + mask_k] = False 
        self.mask[3 * quarter - mask_k : 3 * quarter + mask_k] = False
        
        self.r_mask = nn.Parameter(torch.zeros(self.in_features, dtype=torch.bool), requires_grad=False)
        self.mask[quarter - mask_k : quarter + mask_k] = True 
        self.mask[3 * quarter - mask_k : 3 * quarter + mask_k] = True
        

        self.metrics = exp_metrics
        self.log_freq = log_freq
        
        self.kwargs = kwargs
        

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=multi_head_num, dim_feedforward=feed_forward_dim, batch_first=True)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=multi_head_num, dim_feedforward=feed_forward_dim, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_encoder_layers)

        self.price_decoder = nn.TransformerDecoder(decoder_layer, num_layers=trans_decoder_layers)
        
        self.volume_decoder = nn.TransformerDecoder(decoder_layer, num_layers=trans_decoder_layers)

        # Final linear layer to map the output to the desired shape
        self.linear_projection = nn.Linear(in_features, d_model)
        self.final_layer_price = nn.Linear(d_model, in_features//2)
        self.final_layer_volume = nn.Linear(d_model, in_features//2)
        
        self.positional_encoder = PositionalEncoding(d_model=d_model, max_len=max_len,batch_size=batch_size)
        self.automatic_optimization = False
        
        self.save_hyperparameters()


    def forward(self, src):
        masked_src= src * self.r_mask 
        project_x = self.linear_projection(masked_src)
        position_x = self.positional_encoder(project_x)
        memory = self.encoder(position_x)
        
        price_output = self.price_decoder(memory, memory)
        price_output = self.final_layer_price(price_output)
        
        detach_memory = memory.detach()
        volume_output = self.volume_decoder(detach_memory, detach_memory)
        volume_output = self.final_layer_volume(volume_output)
        
        return price_output, volume_output

    def training_step(self, batch, batch_idx):
        src = batch
        output = self.forward(src)
        price,volume = output 
        opt1, opt2 = self.optimizers()
        
        opt1_loss = metrics[self.metrics[0]](src, price, self.mask)
        opt1_loss.backward()
        opt1.step()
        opt1.zero_grad()
        
        opt2_loss = metrics[self.metrics[1]](src, volume, self.mask)
        opt2_loss.backward()
        opt2.step()
        opt2.zero_grad()
        
        if self.global_step % self.log_freq == 0:
            self.logger_metrics(src,output)
            
    def validation_step(self, batch, batch_idx):
        src = batch
        output = self.forward(src)
        return self.logger_metrics(src,output,"val_")

    def test_step(self, batch, batch_idx):
        src = batch
        output = self.forward(src)
        return self.logger_metrics(src,output,"test_")
    
    def configure_optimizers(self):
        optimizer_price = getattr(torch.optim, self.optimizer_name)(
            list(self.encoder.parameters())+
            list(self.price_decoder.parameters())+ 
            list(self.linear_projection.parameters())+
            list(self.final_layer_price.parameters()), 
            lr=self.price_lr)
        
        optimizer_volume = getattr(torch.optim, self.optimizer_name)(
            list(self.volume_decoder.parameters())+ 
            list(self.final_layer_volume.parameters()), 
            lr=self.price_lr)
        
        return [optimizer_price,optimizer_volume]

    def compute_loss(self, output, target):
        
        loss_func = metrics[self.metrics[0]]
        
        return loss_func(output, target)

    def logger_metrics(self,src,output,log_type="train_"):
        # logs = self.main_metrics(src,output,log_type)
        logs = {}
        
        price,volume = output

        with torch.no_grad():
            for mtc in self.metrics:
                if 'price' in mtc:
                    tgt = price 
                elif 'volume' in mtc:
                    tgt = volume 
                else:
                    tgt = torch.concat((price,volume), dim = 2)
                loss = metrics[mtc](src,tgt,self.mask)
                logs[log_type+mtc] = loss
        self.logger.log_metrics(logs,step = self.global_step)
        
#======================  end   =========================

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    exp_setting_folder = "./experiment/exp_settings/A6000"
    ecm = ECM(exp_setting_folder,'default.json')

    config_file = "mask_8.json"
    ecm.load_config(config_file)
    config_file_path = os.path.join(exp_setting_folder,config_file)

    model = NewLOB_AE_PV_Masked(batch_size=ecm.model.batch_size,
                    d_model=ecm.model.d_model,
                    max_len=ecm.model.max_len,
                    in_features=ecm.model.in_features,
                    lr=ecm.model.learning_rate,
                    trans_embed_size=ecm.model.feed_forward_dim,
                    multi_head_num=ecm.model.multi_head_num,
                    trans_encoder_layers=ecm.model.trans_encoder_layers,
                    trans_decoder_layers =ecm.model.trans_decoder_layers,
                    feed_forward_dim=ecm.model.feed_forward_dim,
                    optimizer_name= ecm.model.optimizer_name,
                    exp_metrics=ecm.model.metrics,
                    log_freq=ecm.model.log_freq,
                    mask_k=ecm.model.mask_k
                    )

    comet_logger = CometLogger(
                        api_key=ecm.comet.api_key,  # Optional
                        workspace=ecm.comet.workspace,  # Optional
                        save_dir=ecm.data.log_folder,  # Optional
                        project_name=ecm.comet.project_name,  # Optional
                        experiment_name=ecm.experiment.name+"_"+ecm.experiment.version,  # Optional
                        # experiment_key=ecm.comet.experiment_key # Optional
                    )
    
    
    
    trainer = L.Trainer(max_epochs=ecm.train.max_epochs,
                        profiler="advanced",
                        # callbacks=[EarlyStopping(monitor="train_volume_loss", mode="min",stopping_threshold=1e-5)],
                        # callbacks=[checkpoint_callback,],
                        logger=comet_logger,
                        num_sanity_val_steps=2,
                        devices=ecm.train.device)
    # comet_logger.experiment.log_parameters(ecm.model)
    # comet_logger.experiment.log_asset(config_file_path,config_file)
    trainer.logger.experiment.log_asset(config_file_path,config_file)
    
    train_loader, valid_loader, test_loader = create_dataloaders_from_files(ecm.data.data_file,batch_size=ecm.data.batch_size,workers=ecm.data.workers,split_seed=ecm.data.manual_seed)
    
    trainer.fit(model,train_loader,valid_loader)
    # trainer.test(model,dataloaders=test_loader)
