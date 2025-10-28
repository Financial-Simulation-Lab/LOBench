"""
Created on : 2024-08-08
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/model/prediction/mlp.py
Description: MLP Model for prediction
---
Updated    : 
---
Todo       : 
"""

# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import comet_ml 
import torch 
from torch import nn 
import lightning as L 
from model.prediction.base import PredictBase
from config import ExpConfigManager as ECM 



class MLP(PredictBase):
    def __init__(self):
        super().__init__()
        ecm = ECM()
        hidden_layer_dim = ecm.model.d_model
        p_dropout = ecm.model.dropout
        output_dim = ecm.data.num_class 
        # self.ecm = ecm 
        flat_dims = ecm.model.max_len * ecm.model.in_features
        self.lr = ecm.train.learning_rate
        self.optimizer_name =ecm.train.optimizer_name
        
        self.linear1 = nn.Linear(flat_dims, hidden_layer_dim)
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear2 = nn.Linear(hidden_layer_dim, output_dim)
        
        self.save_hyperparameters()
         

    def neural_architecture(self, batch):
        # [batch_size x 40 x observation_length]
        x = batch
        x = x.view(x.size(0), -1).float()
        out = self.linear1(x)
        out = self.leakyReLU(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), lr=self.lr)
        return optimizer
    



