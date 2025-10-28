# Using Deep Learning to Detect Price Change Indications in Financial Markets
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8081663

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

class LSTM_encoder(LOBModel):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=3,
            batch_first=True
        )
        
        self.proj=nn.Linear(40,128)
        self.leakyReLU = nn.LeakyReLU()
        self.output_proj=nn.Linear(128,128)

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

class LSTM_decoder(LOBModel):
    def __init__(self):
        super().__init__()

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

class LSTM(LOBAutoEncoder):
    def __init__(self, d_model, unified_d, ckpt_path=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder=LSTM_encoder()
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
            elif self.decoder_name == 'lstm_decoder':
                self.decoder = LSTM_decoder()
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
    
    def forward(self,x_enc, x_mark_enc, mask =None):
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

    

 