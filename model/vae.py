import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.base import LOBModel, LOBAutoEncoder 
from pathlib import Path

import lightning as L
from torch import nn
import torch
from torch import optim
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch.nn import functional as F
# from torch import tensor as Tensor
from metric import metrics
import torch.backends.cudnn as cudnn
torch.set_float32_matmul_precision('medium')
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.utilities.seed import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from data.data_ashare import AShareVaeData, AShareVaeDataModule
from model.Transformer import Trans_Decoder, Transformer_block

Tensor = TypeVar('torch.tensor')



class VallinaVAE(LOBAutoEncoder):
    def __init__(self,
                 in_channels: int = 2,
                 hidden_dims: List = None,
                 latent_dim: int = 64,
                 kld_weight: float = 0.00025,
                 weight_decay: float = 0.0000,
                 metrics: List[str] = ['mse_loss'],
                 log_freq:int = 50,
                 optimizer_name:str = "Adam",
                 scheduler_gamma:float = 0.95,
                 lr:float = 0.005,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args,**kwargs)

        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.metrics = metrics
        self.log_freq = log_freq
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay    
        self.scheduler_gamma = scheduler_gamma
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),   
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                    )        
                )
        # self.decoder = nn.Sequential(*modules)

        # # self.final_layer = nn.Sequential(
        # #                     nn.ConvTranspose2d(hidden_dims[-1],
        # #                                        hidden_dims[-1],
        # #                                        kernel_size=3,
        # #                                        stride=2,
        # #                                        padding=1,
        # #                                        output_padding=1),
        # #                     nn.BatchNorm2d(hidden_dims[-1]),
        # #                     nn.LeakyReLU(),
        # #                     nn.Conv2d(hidden_dims[-1], out_channels= 1,
        # #                               kernel_size= 3, padding= 1),
        # #                     nn.Tanh())
        # self.final_layer = nn.Sequential(
        #         nn.Upsample(size=(100, 20), mode='bilinear', align_corners=False),
        #         nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
        #         nn.Tanh()
        #     )
        
    # decoder Transformer
        # self.linear_min1=nn.Linear(trans_embed_size,512)
        # self.linear_min2=nn.Linear(512,1024)
        # self.linear_min3=nn.Linear(1024,100*40)
        self.project_layer = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 100*40)
        )
        
        self.decoder = Transformer_block(
            d_model=self.latent_dim,
            multi_head_num=4,
            num_layers=6,
            seq_len=100,
            d_input=40,
            d_output=40,
            feed_forward=True
        )

    def encode(self,input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
                
    def decode(self,z: Tensor) -> Tensor:
        # result = self.decoder_input(z)
        # result = result.view(-1, 512, 4, 1)
        # result = self.decoder(result)
        # result = self.final_layer(result)
        # return result
        result = self.project_layer(z)
        result = result.view(result.shape[0],100,40)
        result = self.decoder(result)
        # 转为2,100,20形状
        result = torch.stack([result[:,:,:20],result[:,:,20:]],dim=1)
        return result 
        
    
    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std 
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z)
        return [result, input, mu, log_var]
    
    def loss_function(self,
                      src,
                      **kwargs) -> dict:
        recons = src[0]
        input = src[1]
        mu = src[2]
        log_var = src[3]
        # stage = kwargs['stage']

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim = 1), dim = 0)
        
        # 价格错误惩罚, 对于每个[batch,2,100,20]数据的[batch,0,:,:]这20列，后面的列减前面的列的如果小于零就是价格档位发生了错误，需要惩罚
        prices = recons[:,0,:,:]    
        price_diff = prices[:,:,:-1] - prices[:,:,1:]
        price_loss = F.relu(price_diff).mean()

        loss = recons_loss + price_loss + kld_loss * self.kld_weight
        return {'loss': loss, 'rec_loss':recons_loss.detach(), 'kld':-kld_loss.detach(),'price_loss':price_loss.detach()}
    
    def sample(self,
                num_samples:int,
                current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
    
    def training_step(self, batch, batch_idx):
        src, y = batch 
        output = self.forward(src)
        return self.logger_metrics(output,log_type="train_")
    
    def validation_step(self, batch, batch_idx):
        src, y = batch 
        output = self.forward(src)
        self.logger_metrics(output,log_type="val_")
    
    def test_step(self, batch, batch_idx):
        src, y = batch 
        output = self.forward(src)
        self.logger_metrics(output,log_type="test_")

    def logger_metrics(self,src,log_type="train_"):
        loss = self.loss_function(src)
        logs = {
            log_type+'total_loss': loss['loss'],
            log_type+'rec_loss': loss['rec_loss'],
            log_type+'kld': loss['kld'],
            log_type+'price_loss': loss['price_loss']
        }

        if log_type == "val_":
            self.log("val_loss", loss['loss'], prog_bar=True, logger=True,sync_dist=True)
                
        if self.global_step % self.log_freq == 0:
            self.logger.log_metrics(logs,step = self.global_step)   
        
        return loss 
    
    # def configure_optimizers(self):
    #     optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), lr=self.lr)
    #     return {"optimizer": optimizer}
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.parameters(),lr=self.lr, weight_decay=self.weight_decay)
        optims.append(optimizer)


        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],gamma = self.scheduler_gamma)
        scheds.append(scheduler)
        return optims, scheds

    
if __name__ == "__main__":
    # For reproducibility
    # seed_everything(1234, True)

    ashare_vae_data = AShareVaeData(dataset_path="dataset/real_data/raw_csv/sz000001-level10.csv",balanced=False)
    vae_data_module = AShareVaeDataModule(ashare_vae_data, batch_size=64)

    tb_logger =  TensorBoardLogger(save_dir='./dataset/exp_data/vae_model',name='VallinaVAE')

    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=2, 
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "val_loss",
                                        save_last= True),
                    ],
                    strategy=DDPStrategy(find_unused_parameters=True),
                    devices = [0,1],
                    max_epochs = 200,
                    )
    vae = VallinaVAE()

    print(f"======= Training VallinaVAE =======")

    runner.fit(vae,datamodule=vae_data_module)
    