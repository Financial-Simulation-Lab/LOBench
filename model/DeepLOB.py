"""
author: Florian Krach

This code is based on the DeepLOB code provided in
https://github.com/FlorianKrach/PD-NJODE/blob/master/DeepLOB/
"""
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.base import LOBModel, LOBAutoEncoder
from model.layers.Projection import encode_projection,decode_projection

import torch
import torch.nn as nn
import torch.nn.functional as F

class deeplob_encoder(LOBModel):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            #             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, self.d_model)

    def forward(self, x):
        if len(x.shape)==3:
            x=x.unsqueeze(1)
        elif len(x.shape)==2:
            x=x.unsqueeze(0)
            x=x.unsqueeze(0)

        # h0: (number of hidden layers, batch size, hidden size)
        device = x.device 
        h0 = torch.zeros(1, x.size(0), 128, device=device)
        c0 = torch.zeros(1, x.size(0), 128, device=device)


        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        #         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        #forecast_y = torch.softmax(x, dim=1)

        return x
    
class deeplob_decoder(LOBModel):
    # decoder
    def __init__(self, d_model):
        super().__init__()

        self.fc2 = nn.Linear(d_model, 128)
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=192, 
                        num_layers=1, batch_first=True)
                        
        self.inp1_dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )
        
        self.inp2_dec = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,1), padding=(1,0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )
        
        self.inp3_dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=(5,1), padding=(2,0)),  
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )

        self.conv3_dec = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        self.conv2_dec = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        self.conv1_dec = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(1,2), stride=(1,2)), 
            nn.Tanh(),
            nn.BatchNorm2d(1),

            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(1),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(1),
        )
    
    def forward(self, x):
        device = x.device
        h0 = torch.zeros(1, x.size(0), 192, device=device)
        c0 = torch.zeros(1, x.size(0), 192, device=device)
        
        # 解码    
        x = self.fc2(x)
        x = x.unsqueeze(1)
        x = x.repeat(1,82,1)
        x, _ = self.lstm(x, (h0, c0)) 
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2],1))
        x=x.permute(0,2,1,3)
        x1=x[:,:64,:,:]
        x2=x[:,64:128,:,:]
        x3=x[:,128:192,:,:]
        x_inp1 = self.inp1_dec(x1)
        x_inp2 = self.inp2_dec(x2) 
        x_inp3 = self.inp3_dec(x3)
        x = x_inp1+x_inp2+x_inp3
        
        x = self.conv3_dec(x)
        x = self.conv2_dec(x)
        x = self.conv1_dec(x)
        
        return x.squeeze(1)

class DeepLOB(LOBAutoEncoder):
    def __init__(self,d_model,unified_d,ckpt_path=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = deeplob_encoder(d_model)  # Instantiate line_encoder
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
            elif self.decoder_name == 'deeplob_decoder':
                self.decoder = deeplob_decoder(d_model)  # Instantiate line_decoder
                self.decode_proj = decode_projection(unified_d=unified_d,seq_len=self.seq_len,enc_in=self.enc_in)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(unified_d, self.c_out, bias=True)
    
    def encode(self,x_enc):
        enc_out = self.encoder(x_enc)
        enc_out = self.encode_proj(enc_out)    
        return enc_out
          
    def classification(self, x_enc):
        enc_out = self.encode(x_enc)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1) 
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
        
    def forward(self, x_enc, x_mark_enc, mask=None):
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
    
