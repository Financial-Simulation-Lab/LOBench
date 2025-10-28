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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# from data.data_lobcast import prepare_data_fi
from metric import *
from data.data_processing import SimDataloader, SimDataset


import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary as nofitModelSummary
# from metric import NewLoss, PriceLoss, VolumeLoss, WeightedPriceLoss, WeightedVolumeLoss, MSELoss
from metric import * 

from config import ExpConfigManager as ECM

from lightning.pytorch.loggers import CometLogger, TensorBoardLogger
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
        x = x + self.pe[:x.size(0),:].clone()
        return x
    
# metrics = 


class LOB_AE_PRESENTATION(L.LightningModule):
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
                 **kwargs):
        
        super().__init__()
        self.lr = lr
        self.trans_embed_size = trans_embed_size
        self.multi_head_num = multi_head_num
        self.trans_encoder_layers = trans_encoder_layers
        self.trans_decoder_layers = trans_decoder_layers
        self.feed_forward_dim = feed_forward_dim
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.alpha = alpha
        self.kwargs = kwargs
        self.max_len = max_len
        self.metrics = exp_metrics 
        self.log_freq = log_freq

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=multi_head_num, dim_feedforward=feed_forward_dim, batch_first=True)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=multi_head_num, dim_feedforward=feed_forward_dim, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_encoder_layers)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=trans_decoder_layers)

        # Final linear layer to map the output to the desired shape
        self.linear_projection = nn.Linear(in_features, d_model)
        self.final_layer = nn.Linear(d_model, in_features)
        self.positional_encoder = PositionalEncoding(d_model=d_model, max_len=max_len,batch_size=batch_size)
        
        self.save_hyperparameters()
        
        # self.logger.experiment.log_text("Model Hyperparameters")

    def forward(self, src):
        # src= src[0]
        project_x = self.linear_projection(src)
        position_x = self.positional_encoder(project_x)
        memory = self.encoder(position_x)
        
        output = self.decoder(memory, memory)
        output = self.final_layer(output)
        
        return output

    def training_step(self, batch, batch_idx):
        src,_ = batch
        output = self.forward(src)
        return self.logger_metrics(output,src,"train_")  

    def validation_step(self, batch, batch_idx):
        src,_ = batch
        output = self.forward(src)
        return self.logger_metrics(output,src,"val_")

    def test_step(self, batch, batch_idx):
        src,_ = batch
        output = self.forward(src)
        return self.logger_metrics(output,src,"test_")
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), lr=self.lr)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
        return {"optimizer": optimizer}

    def compute_loss(self, output, target):
        
        loss_func = metrics[self.metrics[0]]
        
        return loss_func(output, target)
    
    def logger_metrics(self,src,tgt,log_type="train_"):
        if log_type == "train_":
            returned_loss = self.compute_loss(src,tgt)
        else:
            with torch.no_grad():
                returned_loss = self.compute_loss(src,tgt)

        logs = {
            log_type+self.metrics[0]: returned_loss
        }
        with torch.no_grad():
            for mtc in self.metrics[1:]:
                loss = metrics[mtc](src,tgt)
                logs[log_type+mtc] = loss
        if self.global_step % self.log_freq == 0:
            self.logger.log_metrics(logs,step = self.global_step)   
        
        return returned_loss 
  

class LOB_AE_PRESENTATION_PV(L.LightningModule):
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
                #  config_file_path = None,
                #  config_file_name = None,
                 **kwargs):
        
        super().__init__()
        self.price_lr = lr
        self.volume_lr = lr 
        self.trans_embed_size = trans_embed_size
        self.multi_head_num = multi_head_num
        self.trans_encoder_layers = trans_encoder_layers
        self.trans_decoder_layers = trans_decoder_layers
        self.feed_forward_dim = feed_forward_dim
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.alpha = alpha
        self.kwargs = kwargs
        self.max_len = max_len
        self.metrics = exp_metrics
        self.log_freq = log_freq

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
        project_x = self.linear_projection(src)
        position_x = self.positional_encoder(project_x)
        memory = self.encoder(position_x)
        
        price_output = self.price_decoder(memory, memory)
        price_output = self.final_layer_price(price_output)
        
        detach_memory = memory.detach()
        volume_output = self.volume_decoder(detach_memory, detach_memory)
        volume_output = self.final_layer_volume(volume_output)
        
        return price_output, volume_output

    def training_step(self, batch, batch_idx):
        src,_ = batch
        output = self.forward(src)
        # self.logger_metrics(src,output)
        price,volume = output 
        opt1, opt2 = self.optimizers()
        
        opt1_loss = metrics[self.metrics[0]](src, price)
        opt1_loss.backward()
        opt1.step()
        opt1.zero_grad()
        
        opt2_loss = metrics[self.metrics[1]](src, volume)
        opt2_loss.backward()
        opt2.step()
        opt2.zero_grad()
        
        if self.global_step % self.log_freq == 0:
            self.logger_metrics(src,output)
            
        
    def validation_step(self, batch, batch_idx):
        src,_ = batch
        output = self.forward(src)
        self.logger_metrics(src,output,"val_")

    def test_step(self, batch, batch_idx):
        src,_ = batch
        output = self.forward(src)
        self.logger_metrics(src,output,"test_")
    
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

    def logger_metrics(self,src,tgt,log_type="train_"):
        logs = {}
        price,volume = tgt 

        with torch.no_grad():
            for mtc in self.metrics:
                if 'price' in mtc:
                    tgt = price 
                elif 'volume' in mtc:
                    tgt = volume 
                else:
                    tgt = torch.concat((price,volume), dim = 2)
                loss = metrics[mtc](src,tgt)
                logs[log_type+mtc] = loss
                
        self.logger.log_metrics(logs,step = self.global_step)
        
        # if log_type == "train_":
        #     returned_loss = self.compute_loss(src,tgt)
        # else:
        #     with torch.no_grad():
        #         returned_loss = self.compute_loss(src,tgt)

        # logs = {
        #     log_type+self.metrics[0]: returned_loss
        # }
        # with torch.no_grad():
        #     for mtc in self.metrics[1:]:
        #         loss = metrics[mtc](src,tgt)
        #         logs[log_type+mtc] = loss
        # if self.global_step % self.log_freq == 0:
        #     self.logger.log_metrics(logs,step = self.global_step)   
        
        # return returned_loss 
#======================  end   =========================

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    exp_setting_folder = "experiment/exp_settings/A6000"
    ecm = ECM(exp_setting_folder,'default.json')

    config_file = "new_dataset.json"
    ecm.load_config(config_file)


    # model = LOB_AE_PRESENTATION(batch_size=ecm.model.batch_size,
    #                 d_model=ecm.model.d_model,
    #                 max_len=ecm.model.max_len,
    #                 in_features=ecm.model.in_features,
    #                 lr=ecm.model.learning_rate,
    #                 trans_embed_size=ecm.model.feed_forward_dim,
    #                 multi_head_num=ecm.model.multi_head_num,
    #                 trans_encoder_layers=ecm.model.trans_encoder_layers,
    #                 trans_decoder_layers =ecm.model.trans_decoder_layers,
    #                 feed_forward_dim=ecm.model.feed_forward_dim,
    #                 optimizer_name= ecm.model.optimizer_name,
    #                 exp_metrics=ecm.train.metrics,
    #                 log_freq=ecm.model.log_freq # Optional
    #                 )
    
    model = LOB_AE_PRESENTATION_PV(
                    batch_size=ecm.data.batch_size,
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
                    exp_metrics=ecm.train.metrics,
                    log_freq=ecm.model.log_freq # Optional
                    )

    # comet_logger = CometLogger(
    #                     api_key=ecm.comet.api_key,  # Optional
    #                     workspace=ecm.comet.workspace,  # Optional
    #                     save_dir=ecm.data.log_folder,  # Optional
    #                     project_name=ecm.comet.project_name,  # Optional
    #                     experiment_name=ecm.experiment.name+"_"+ecm.experiment.version,  # Optional
    #                     # experiment_key=ecm.comet.experiment_key # Optional
    #                 )
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_mse_loss',  # 监控的指标，通常是验证损失或验证准确度
    #     dirpath=f'dataset/exp_data/prediction/checkpoints/{ecm.experiment.name}_{ecm.model.model_name}',  # 保存检查点的目录
    #     filename='{epoch:02d}-{val_loss:.2f}',  # 检查点文件名格式
    #     save_top_k=3,  # 保存最好的三个检查点
    #     mode='min',  # 如果监控的是损失，使用 'min'，如果是准确度，使用 'max'
    #     save_weights_only=True  # 如果只保存模型权重而不是整个模型，设置为 True
    #     )
    
    tensorboard_logger = TensorBoardLogger(
                        save_dir="dataset/exp_data/prediction",
                        version=ecm.experiment.version,
                        name=ecm.experiment.name
                        )
    
    
    trainer = L.Trainer(max_epochs=ecm.train.max_epochs,
                        profiler="advanced",
                        # callbacks=[EarlyStopping(monitor="train_volume_loss", mode="min",stopping_threshold=1e-5)],
                        # callbacks=[checkpoint_callback,],
                        logger=tensorboard_logger,
                        num_sanity_val_steps=2,
                        devices=ecm.train.device)
    sd = SimDataset(dataset_path="./dataset/train_data/train_data_zscore_trend.pt")
    sdm = SimDataloader(sd, batch_size=ecm.data.batch_size)
    # fi_dm = prepare_data_fi()

    trainer.fit(model, sdm.train_dataloader(),sdm.val_dataloader())
    # trainer.test(model,dataloaders=test_loader)
