# Using Deep Learning for price prediction by exploiting stationary limit order book features
# Source: https://www.sciencedirect.com/science/article/pii/S1568494620303410

# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.base import LOBModel, LOBAutoEncoder 
from model.layers.Projection import encode_projection,decode_projection

import torch
from torch import nn
import torch.nn.functional as F

class CNN2_encoder(LOBModel):
    def __init__(self, input_shape=(128,100,40),output_shape =(128,128), temp=249, **kwargs):
        super().__init__(**kwargs)

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
        self.fc1 = nn.Linear(temp*32, 128)
        self.prelu6 = nn.PReLU()

    def forward(self, x):
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

        return out
    
class CNN2_decoder(LOBModel):
    def __init__(self,input_shape=(128,128), output_shape=(128,100,40), temp=249,**kwargs):
        super().__init__(**kwargs)

        # Convolution 1
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(10, 42), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(1)
        self.prelu1 = nn.PReLU()

        # Convolution 2
        self.conv2 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=(10,))  # 3
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # Convolution 3
        self.conv3 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=(8,))  # 1
        self.bn3 = nn.BatchNorm1d(16)
        self.prelu3 = nn.PReLU()

        # Convolution 4 
        self.conv4 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=(6,))  # 1
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        # Convolution 5
        self.conv5 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=(4,))  # 1
        self.bn5 = nn.BatchNorm1d(32)
        self.prelu5 = nn.PReLU()

        # Fully connected 1
        self.fc1 = nn.Linear(128, temp*32)
        self.prelu6 = nn.PReLU()

    def forward(self, out):

        # Linear function 1
        out = self.fc1(out)
        out = self.prelu6(out)
        # print('After fc1:', out.shape)

        out=out.reshape(out.shape[0],32,249)

        # Convolution 5
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.prelu5(out)
        # print('After convolution5, bn5, prelu5:', out.shape)

        # Convolution 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.prelu4(out)
        # print('After convolution4, bn4, prelu4:', out.shape)

         # Convolution 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.prelu3(out)
        # print('After convolution3, bn3, prelu3:', out.shape)

        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        # print('After convolution2, bn2, prelu2:', out.shape)

        out=out.reshape(out.shape[0],16,91,3)

        # Convolution 1
        out = self.conv1(out)
        # print('After convolution1:', out.shape)

        out = self.bn1(out)
        # print('After bn1:', out.shape)

        out = self.prelu1(out)
        # out = out.reshape(out.shape[0], out.shape[1], -1)
        # print('After prelu1:', out.shape)
        out=out.squeeze(1)

        return out
    
    
class CNN2(LOBAutoEncoder):
    def __init__(self, d_model, unified_d, ckpt_path=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder=CNN2_encoder()
        self.encode_proj = encode_projection(unified_d=unified_d,inc_d=d_model)
        
        if ckpt_path:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(ckpt_path, map_location=device)
            encoder_state_dict = {k.replace("encoder.", ""): v for k, v in ckpt['state_dict'].items() if "encoder" in k}
            self.encoder.load_state_dict(encoder_state_dict)
            encode_proj_state_dict = {k.replace("encode_proj.", ""): v for k, v in ckpt['state_dict'].items() if "encode_proj" in k}
            self.encode_proj.load_state_dict(encode_proj_state_dict)
            print("Load the parameters of LOB encoder part successfully.")
            
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encode_proj.parameters():
                param.requires_grad = False
            
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(self.dropout)
            self.projection = nn.Linear(unified_d, self.num_class)
        elif self.task_name == 'reconstruction':
            if self.decoder_name == 'transformer_decoder':
                decoder_layer = nn.TransformerDecoderLayer(unified_d, nhead=8, batch_first=True)
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
                self.decode_proj = decode_projection(unified_d=unified_d,seq_len=self.seq_len,enc_in=self.enc_in)
            elif self.decoder_name == 'cnn2_decoder':
                self.decoder = CNN2_decoder()
                self.decode_proj = decode_projection(unified_d=unified_d,seq_len=self.seq_len,enc_in=self.enc_in)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(unified_d, self.c_out, bias=True)
                
      
    def encode(self,x_enc):
        enc_out = self.encoder(x_enc)
        enc_out = self.encode_proj(enc_out)
        return enc_out
    
    def classification(self, x_enc):
        # Embedding
        enc_out = self.encode(x_enc)

        # Output
        output = self.act(enc_out) 
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    
    def imputation(self, x_enc):
        enc_out = self.encode(x_enc)
        output = self.projection(enc_out)
        output=output.view(output.shape[0],self.seq_len, self.enc_in)
        return output
        
    def reconstruction(self, x_enc):
        enc_out = self.encode(x_enc)
        enc_out = enc_out.unsqueeze(1)
        
        memory = torch.zeros(enc_out.shape[0], enc_out.shape[1],enc_out.shape[2], device=enc_out.device)
        output = self.decoder(enc_out,memory)
        output = self.decode_proj(output)
        return output
        
        
    def forward(self,x_enc,x_mark_enc, mask=None):
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        if self.task_name == 'reconstruction':
            dec_out = self.reconstruction(x_enc)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out
        return None

    
    
   