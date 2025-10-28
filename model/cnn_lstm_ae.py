# Using Deep Learning for price prediction by exploiting stationary limit order book features
# Source: https://www.sciencedirect.com/science/article/pii/S1568494620303410

import lightning as L
import torch
from torch import nn
# from model_viz import Visualize
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.base import LOBModel, LOBAutoEncoder


class CNNLSTM_encoder(LOBModel):
    def __init__(self,batch_size):
        super().__init__()

        self.batch_size = batch_size

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 42), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        # Convolution 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(5,))
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # Convolution 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(5,))
        self.bn3 = nn.BatchNorm1d(32)
        self.prelu3 = nn.PReLU()

        # Convolution 4 
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(5,))
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        self.lstm_input = self.get_lstm_input_size()
        
        # lstm layers
        self.lstm = nn.LSTM(
            input_size=self.lstm_input,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        # fully connected
        self.fc1 = nn.Linear(128, 128)  # fully connected
        self.dropout = nn.Dropout(p=0.1)  # not specified
        self.prelu = nn.PReLU()

    def get_lstm_input_size(self):
        with torch.no_grad():
            sample_in = torch.ones(self.batch_size, 1, 100, 40) # batch_size, 1, seq_len, num_features
            sample_out = self.convolution_forward(sample_in)

        return sample_out.shape[-1]

    def forward(self, x):
        # Adding the channel dimension
        x = x[:, None, :]  # x.shape = [batch_size, 1, 100, 40]

        # print('x.shape:', x.shape)
        
        out = self.convolution_forward(x)
        # print('After convolution_forward:', out.shape)

        # lstm
        temp, (hn, _) = self.lstm(out)
        # print('After lstm:', hn.shape)

        # flatten
        hn=hn[-1]
        hn = hn.view(-1, 128)
        # print('After flatten:', hn.shape)

        out = self.fc1(hn)
        # print('After fc1:', out.shape)

        out = self.dropout(out)
        out = self.prelu(out)

        return out

    def convolution_forward(self, x):
        # print('Starting convolution_forward')

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

        # print('Ending convolution_forward')

        return out
    
class CNNLSTM_decoder(LOBModel):
    def __init__(self,batch_size):
        super().__init__()

        self.batch_size = batch_size

        # Convolution 1
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(5, 42), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(1)
        self.prelu1 = nn.PReLU()

        # Convolution 2
        self.conv2 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=(5,))
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # Convolution 3
        self.conv3 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=(5,))
        self.bn3 = nn.BatchNorm1d(16)
        self.prelu3 = nn.PReLU()

        # Convolution 4 
        self.conv4 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=(5,))
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        # self.lstm_input = self.get_lstm_input_size()
        
        # lstm layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=276,
            num_layers=2,
            batch_first=True
        )
        
        # fully connected
        self.fc1 = nn.Linear(128, 128)  # fully connected
        self.dropout = nn.Dropout(p=0.1)  # not specified
        self.prelu = nn.PReLU()

    def forward(self, x):

        out = self.fc1(x)

        out = self.dropout(out)
        out = self.prelu(out)

        out=out.unsqueeze(1)
        out=out.repeat(1,32,1)

        # lstm
        out, (hn, _) = self.lstm(out)

        out = self.convolution_forward(out)
        # print('After convolution_forward:', out.shape)

        out=out.squeeze(1)

        return out

    def convolution_forward(self, x):
        # print('Starting convolution_forward')

        # Convolution 1
        out = self.conv4(x)
        # print('After convolution1:', out.shape)

        out = self.bn4(out)
        # print('After bn1:', out.shape)

        out = self.prelu4(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        # print('After prelu1:', out.shape)

        # Convolution 2
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.prelu3(out)
        # print('After convolution2, bn2, prelu2:', out.shape)

        # Convolution 3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        # print('After convolution3, bn3, prelu2:', out.shape)

        out=out.reshape(out.shape[0],out.shape[1],96,3)

        # Convolution 4
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu1(out)
        # print('After convolution4, bn4, prelu4:', out.shape)

        # print('Ending convolution_forward')

        return out


class CNNLSTM_AE(LOBAutoEncoder):
    def __init__(self,batch_size, *args, **kwargs): 
        super().__init__( *args, **kwargs)
        self.encoder=CNNLSTM_encoder(batch_size)
        self.decoder=CNNLSTM_decoder(batch_size)
    
    def forward(self,x):
        encoded=self.encoder(x)
        output=self.decoder(encoded)
        return output
    
    
if __name__ ==  "__main__":
    x=torch.randn(12,100,40)
    
    