import lightning as L 
import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional,List,Tuple,Dict,Union,Protocol
   


class Encoder(nn.Module):
    def __init__(self, input_shape:Tuple, represent_size:int):
        super().__init__()
        self.represent_size = represent_size
        self.input_shape = input_shape
        self.encoder = self.create_encoder()
        
        
    def create_encoder(self):
        raise NotImplementedError("The encoder must be defined")
    
    def forward(self, x):
        return self.encoder(x)


    
class Decoder(nn.Module):
    def __init__(self,  represent_size:int,output_shape:Tuple):
        super().__init__()
        self.represent_size = represent_size
        self.output_shape = output_shape
        self.decoder = self.create_decoder()
        
    def create_decoder(self):
        raise NotImplementedError("The decoder must be defined")
    
    def forward(self, x):
        return self.decoder(x)
    

class AE(L.LightningModule):
    def __init__(self, encoder, decoder, optimizer_name:str, lr:float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_name = optimizer_name
        self.lr = lr
    
    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out
    
    def get_latent(self, x):
        return self.encoder(x)
    
    def loss_fun(self,pred, target):
        return F.mse_loss(pred, target)
        
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fun(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimizer_name)(self.parameters(), lr=self.lr)
    

class CNN_Encoder(Encoder):
    def __init__(self, input_shape:Tuple,represent_size:int):
        super().__init__(input_shape,represent_size)
        
    def create_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_shape,self.input_shape//2),
            nn.ReLU(),
            nn.Linear(self.input_shape//2, self.represent_size)
        )
        
class CNN_Decoder(Decoder):
    def __init__(self, represent_size:int,output_shape:Tuple):
        super().__init__(represent_size,output_shape)
        
    def create_decoder(self):
        return nn.Sequential(
            nn.Linear(self.represent_size,self.output_shape//2),
            nn.ReLU(),
            nn.Linear(self.output_shape//2, self.output_shape)
        )
        


    
    
    
