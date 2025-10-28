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
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import comet_ml 
import torch 
from torch import nn 
import lightning as L 
from model.prediction.base import PredictBase
from config import ExpConfigManager as ECM 
import math 

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


class Transformer(PredictBase):
    def __init__(self):
        
        super().__init__()
        ecm = ECM() 
        
        self.lr = ecm.train.learning_rate
        self.trans_embed_size = ecm.model.trans_embed_size
        self.multi_head_num = ecm.model.multi_head_num
        self.trans_encoder_layers = ecm.model.trans_encoder_layers
        self.trans_decoder_layers = ecm.model.trans_decoder_layers
        self.feed_forward_dim = ecm.model.feed_forward_dim
        self.batch_size = ecm.model.batch_size
        self.optimizer_name = ecm.train.optimizer_name

        self.max_len = ecm.model.max_len
        self.d_model = ecm.model.d_model

        self.in_features = ecm.model.in_features  
        self.out_features = ecm.model.out_features

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.multi_head_num, dim_feedforward=self.feed_forward_dim, batch_first=True)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.multi_head_num, dim_feedforward=self.feed_forward_dim, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.trans_encoder_layers)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.trans_decoder_layers)

        # Final linear layer to map the output to the desired shape
        self.linear_projection = nn.Linear(self.in_features, self.d_model)
        self.final_layer = nn.Sequential(
            nn.Linear(self.d_model*self.max_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model,self.d_model//8),
            nn.ReLU(),
            nn.Linear(self.d_model//8,self.out_features)
        )
        self.positional_encoder = PositionalEncoding(d_model=self.d_model, max_len=self.max_len,batch_size=self.batch_size)
        
        self.save_hyperparameters()
        

    def neural_architecture(self, src):
        project_x = self.linear_projection(src)
        position_x = self.positional_encoder(project_x)
        memory = self.encoder(position_x)
        
        output = self.decoder(memory, memory)
        flat_output = output.view(output.size(0),-1).float()
        output = self.final_layer(flat_output)
        
        return output
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), lr=self.lr)
        return optimizer