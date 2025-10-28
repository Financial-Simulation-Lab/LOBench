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


class LSTM_encoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        ecm = ECM()
        self.n_features = ecm.model.in_features
        self.output_dim = ecm.data.num_class
        self.lr = ecm.train.learning_rate
        self.optimizer_name =ecm.train.optimizer_name
        self.embedding_size = ecm.model.embedding_dim
        self.hidden_size = ecm.model.feed_forward_dim

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True
        )
        
        self.proj=nn.Linear(self.n_features,self.embedding_size)
        self.leakyReLU = nn.LeakyReLU()
        self.output_proj=nn.Linear(self.embedding_size,self.hidden_size)

    def forward(self, x):
        x = x.float()

        x=self.proj(x)
        output, (hn, _) = self.lstm(x)          # lstm with input, hidden, and internal state (batch, time-step, features)
        
        # before hn.shape = [1, batch_size, features]
        hn=hn[-1]
        hn = hn.view(-1, 128)  # reshaping the data for Dense layer next
        # after hn.shape = [batch_size, features]
        
        out=self.output_proj(hn)
        return out

class LSTM_decoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        ecm = ECM()
        self.n_features = ecm.model.in_features
        self.output_dim = ecm.data.num_class
        self.lr = ecm.train.learning_rate
        self.optimizer_name =ecm.train.optimizer_name
        self.embedding_size = ecm.model.embedding_dim
        self.hidden_size = ecm.model.feed_forward_dim

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True
        )
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=3,
            batch_first=True
        )
        
        self.proj=nn.Linear(128,128)
        self.leakyReLU = nn.LeakyReLU()
        self.output_proj=nn.Linear(128,40)

    def forward(self, x):
        x = x.float()
        x=x.unsqueeze(1)
        x=x.repeat(1,100,1)
        x=self.proj(x)
        output, (hn, _) = self.lstm(x)          # lstm with input, hidden, and internal state (batch, time-step, features)
        
        out=self.output_proj(output)
        return out

class LSTM_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=LSTM_encoder()
        self.decoder=LSTM_decoder()
    
    def forward(self,x):
        encoded=self.encoder(x)
        output=self.decoder(encoded)
        return output

# matrix=torch.randn(5,100,40)
# autoencoder=LSTM_autoencoder()
# matrix_output=autoencoder(matrix)
# print(matrix_output.shape)
 