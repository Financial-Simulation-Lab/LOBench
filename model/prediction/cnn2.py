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


class CNN2(PredictBase):
    def __init__(self):
        super().__init__()
        ecm = ECM()
        self.n_features = ecm.model.in_features
        self.output_dim = ecm.data.num_class
        self.lr = ecm.train.learning_rate
        self.optimizer_name =ecm.train.optimizer_name

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(10, 42), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        # Convolution 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(10,))  # 3
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # Convolution 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(8,))  # 1
        self.bn3 = nn.BatchNorm1d(32)
        self.prelu3 = nn.PReLU()

        # Convolution 4
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(6,))  # 1
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        # Convolution 5
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(4,))  # 1
        self.bn5 = nn.BatchNorm1d(32)
        self.prelu5 = nn.PReLU()

        # Fully connected 1
        self.fc1 = nn.Linear(249 * 32, 32)
        self.prelu6 = nn.PReLU()

        # Fully connected 2
        self.fc2 = nn.Linear(32, self.output_dim)

    def neural_architecture(self, x):
        # Adding the channel dimension
        x = x[:, None, :]  # x.shape = [batch_size, 1, 100, 40]

        # print('x.shape:', x.shape)

        # Convolution 1
        out = self.conv1(x)
        # print('After convolution1:', out.shape)

        out = self.bn1(out)
        # print('After bn1:', out.shape)

        out = self.prelu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        # print('After prelu1:', out.shape)

        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        # print('After convolution2, bn2, prelu2:', out.shape)

        # Convolution 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.prelu3(out)
        # print('After convolution3, bn3, prelu3:', out.shape)

        # Convolution 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.prelu4(out)
        # print('After convolution4, bn4, prelu4:', out.shape)

        # Convolution 5
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.prelu5(out)
        # print('After convolution5, bn5, prelu5:', out.shape)

        # flatten
        out = out.view(out.size(0), -1)
        # print('After flatten:', out.shape)

        # Linear function 1
        out = self.fc1(out)
        out = self.prelu6(out)
        # print('After fc1:', out.shape)

        # Linear function (readout)
        out = self.fc2(out)
        # print('After fc2:', out.shape)

        return out

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), lr=self.lr)
        return optimizer
