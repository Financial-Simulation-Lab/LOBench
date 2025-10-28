# Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8010701

import lightning as L 
from torch import nn
import torch
# from model_viz import Visualize


class CNN1_encoder(L.LightningModule):

    def __init__(self,temp=26):
        super().__init__()
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 40), padding=(3, 0), dilation=(2, 1))
        self.relu1 = nn.LeakyReLU()

        # Convolution 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(4,))
        self.relu2 = nn.LeakyReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, return_indices=True)

        # Convolution 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(3,), padding=2)
        self.relu3 = nn.LeakyReLU()

        # Convolution 4
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(3,), padding=2)
        self.relu4 = nn.LeakyReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, return_indices=True)

        # Fully connected 1
        self.fc1 = nn.Linear(temp*32, 128)
        self.relu5 = nn.LeakyReLU()

        # Fully connected 2
        # self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Adding the channel dimension
        x = x[:, None, :]  # x.shape = [batch_size, 1, 100, 40]

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
        out,indice1 = self.maxpool1(out)
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
        out,indice2 = self.maxpool2(out)
        # print('After maxcpool2:', out.shape)

        # flatten
        out = out.view(out.size(0), -1)
        # print('After flatten:', out.shape)

        # Linear function 1
        out = self.fc1(out)
        out = self.relu5(out)

        return out,indice1,indice2

class CNN1_decoder(L.LightningModule):
    def __init__(self):
        super().__init__()

        #transpose Convolution1
        self.conv1 = nn.ConvTranspose2d(16, 1, kernel_size=(4,40), padding=(3,0), dilation=(2,1))
        self.relu1 = nn.LeakyReLU()

        #transpose Convolution 2
        self.conv2 = nn.ConvTranspose1d(16, 16, kernel_size=(4,))
        self.relu2 = nn.LeakyReLU()

        # Max Unpool 1
        self.maxunpool1 = nn.MaxUnpool1d(kernel_size=3,stride=2)

        #transpose Convolution 3
        self.conv3 = nn.ConvTranspose1d(in_channels=32, out_channels=16,kernel_size=(3,),padding=2)
        self.relu3 = nn.LeakyReLU()

        #transpose Convolution 4
        self.conv4 = nn.ConvTranspose1d(in_channels=32, out_channels=32,kernel_size=(3,),padding=2)
        self.relu4 = nn.LeakyReLU()

        # Max unpool 2
        self.maxunpool2 = nn.MaxUnpool1d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(128, 26*32)
        self.relu5 = nn.LeakyReLU()

    def forward(self, x,indice1,indice2):

        # Adding the channel dimension
        x=self.fc1(x)
        x=self.relu5(x)

        x=x.view(x.shape[0],32,26)

        x=self.maxunpool2(x,indice2)
        #after maxunpool2 the shape is bsz,32,52

        x=self.conv4(x)
        x=self.relu4(x)

        x=self.conv3(x)
        x=self.relu3(x)

        x=self.maxunpool1(x,indice1)
        #shape is different from encoder

        # Convolution 2
        x = self.conv2(x)
        x = self.relu2(x)

        x=x.unsqueeze(3)
        # Convolution 1
        x = self.conv1(x)
        x = self.relu1(x)
        x=x.squeeze(1)

        return x
    
class CNN1_AE(L.LightningModule):
    def __init__(self, *args,**kargs):
        super().__init__()
        self.encoder=CNN1_encoder()
        self.decoder=CNN1_decoder()
    
    def forward(self,x):
        encoded,indice1,indice2=self.encoder(x)
        output=self.decoder(encoded,indice1,indice2)
        return output

    
   