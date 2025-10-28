# Using Deep Learning for price prediction by exploiting stationary limit order book features
# Source: https://www.sciencedirect.com/science/article/pii/S1568494620303410

# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.base import LOBModel, LOBAutoEncoder

import torch
from torch import nn
import torch.nn.functional as F
     
class Decoder(LOBAutoEncoder):
    def __init__(self, unified_d,**kwargs):
        super().__init__(**kwargs)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(self.dropout)
            self.projection = nn.Linear(unified_d, self.num_class)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(unified_d, self.c_out, bias=True)

    def classification(self, x_enc):
        # Output
        output = self.act(x_enc) 
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    
    def imputation(self, x_enc):
        output = self.projection(x_enc)
        output=output.view(output.shape[0],self.seq_len, self.enc_in)
        return output   
        
    def forward(self,x_enc,mask=None):
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out
        return None

    
    
   