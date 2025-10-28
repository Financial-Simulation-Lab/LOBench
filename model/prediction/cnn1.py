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


class CNN1(PredictBase):

    def __init__(self):
        super().__init__()
        ecm = ECM()
        self.n_features = ecm.model.in_features
        self.output_dim = ecm.data.num_class
        self.lr = ecm.train.learning_rate
        self.optimizer_name =ecm.train.optimizer_name

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, self.n_features), padding=(3, 0), dilation=(2, 1))
        self.relu1 = nn.LeakyReLU()

        # Convolution 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(4,))
        self.relu2 = nn.LeakyReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        # Convolution 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(3,), padding=2)
        self.relu3 = nn.LeakyReLU()

        # Convolution 4
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(3,), padding=2)
        self.relu4 = nn.LeakyReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(26*32, 32)
        self.relu5 = nn.LeakyReLU()

        # Fully connected 2
        self.fc2 = nn.Linear(32, self.output_dim)

    def neural_architecture(self, x):
        # Adding the channel dimension
        x = x[:, None, :]  # x.shape = [batch_size, 1, 100, 40]

        # print('x.shape:', x.shape)

        # Convolution 1
        out = self.conv1(x)
        out = self.relu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        # print('After convolution1:', out.shape)

        # Convolution 2
        out = self.conv2(out)
        out = self.relu2(out)
        # print('After convolution2:', out.shape)

        # Max pool 1
        out = self.maxpool1(out)
        # print('After maxpool1:', out.shape)

        # Convolution 3
        out = self.conv3(out)
        out = self.relu3(out)
        # print('After convolution3:', out.shape)

        # Convolution 4
        out = self.conv4(out)
        out = self.relu4(out)
        # print('After convolution4:', out.shape)

        # Max pool 2
        out = self.maxpool2(out)
        # print('After maxcpool2:', out.shape)

        # flatten
        out = out.view(out.size(0), -1)
        # print('After flatten:', out.shape)

        # Linear function 1
        out = self.fc1(out)
        out = self.relu5(out)
        # print('After linear1:', out.shape)

        # Linear function (readout)
        out = self.fc2(out)
        # print('After linear2:', out.shape)

        return out
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), lr=self.lr)
        return optimizer
    
