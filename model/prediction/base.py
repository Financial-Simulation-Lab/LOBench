"""
Created on : 2024-07-04
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : model/base.py
Description: The base model of all AE-based neural network models.
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

import torch
import torch.nn as nn
import torch.nn.functional as F


import lightning as L

from typing import Optional,List,Tuple,Dict,Union,Protocol


########################################################
#######      Neural Network Base Model        ##########
########################################################
class PredictBase(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss()
        self.lr = 0.00001
    
    def neural_architecture(self, batch):
        raise NotImplementedError("The neural architecture must be defined")
        
    def forward(self, x):
        # time x features - 40 x 100 in general
        out = self.neural_architecture(x)
        logits = nn.Softmax(dim=1)(out)  # todo check if within model
        return out, logits

    def training_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        # if self.global_step % 100  == 0:
        #     self.logger.log_metrics({"train_CE_loss": loss_val}, step=self.global_step)
        self.logger.log_metrics({"train_CE_loss": loss_val}, step=self.global_step)
        return loss_val 

    def validation_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        # if batch_idx % 20 == 0:
        #     self.logger.log_metrics({"val_loss": loss_val}, step=self.global_step)
        self.logger.log_metrics({"val_CE_loss": loss_val}, step=self.global_step)
        return loss_val
    
    def test_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        # if batch_idx % 20 == 0: 
        #     self.logger.log_metrics({"test_loss": loss_val}, step=self.global_step)
        self.logger.log_metrics({"test_CE_loss": loss_val}, step=self.global_step)
        return loss_val

    def make_predictions(self, batch):
        x, y = batch
        out, logits = self.forward(x)
        loss_val = self.loss_func(out, y)

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(logits, dim=1)  # B
        return prediction_ind, y, loss_val, logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    

    
