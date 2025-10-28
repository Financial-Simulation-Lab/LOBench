import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.base import LOBAutoEncoder 
from model.layers.Projection import encode_projection,decode_projection

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

# class PositionalEncoding(nn.Module):
#     def __init__(self, model_dim, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, model_dim)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         # self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
    
class Transformer_Encoder(nn.Module):
    def __init__(self, 
                 enc_in, 
                 d_model, 
                 nhead, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dim_feedforward, 
                 dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(enc_in, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)

    def forward(self,x_enc):
        self.pe = self.pe.to(x_enc)
        enc_out = self.embedding(x_enc) * math.sqrt(self.d_model)
        enc_out = enc_out + self.pe[:enc_out.size(0), :]
        # enc_out = self.pos_encoder(enc_out)
        enc_out = self.transformer.encoder(enc_out)   
        return enc_out
        
class Transformer_AE(LOBAutoEncoder):
    def __init__(self, 
                 enc_in, 
                 d_model, 
                 nhead, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dim_feedforward, 
                 dropout,
                 seq_len,
                 unified_d,
                 ckpt_path=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Transformer_Encoder(enc_in, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.encode_proj = encode_projection(unified_d=unified_d,inc_d=d_model*seq_len)
        
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
            # self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(unified_d, self.num_class)
        elif self.task_name == 'reconstruction':
            if self.decoder_name == 'transformer_decoder':
                decoder_layer = nn.TransformerDecoderLayer(unified_d, nhead=8, batch_first=True)
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
                self.decode_proj = decode_projection(unified_d=unified_d,seq_len=self.seq_len,enc_in=self.enc_in)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(unified_d, self.c_out, bias=True)

    def encode(self, x_enc):
        enc_out = self.encoder(x_enc)
        enc_out = self.encode_proj(enc_out)
        return enc_out   
        
    def classification(self, x_enc):
        # Embedding
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
    
    def forward(self, x_enc, mask=None):
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
    
    
    # def forward(self, src, tgt):
        # src = self.embedding(src) * math.sqrt(self.model_dim)
        # tgt = self.embedding(tgt) * math.sqrt(self.model_dim)
        # src = self.pos_encoder(src)
        # tgt = self.pos_encoder(tgt)

        # output = self.transformer(src, tgt)
        # output = self.decoder(output)
        # return output

    # def get_encoder_hidden_states(self, src):
    #     src = self.embedding(src) * math.sqrt(self.model_dim)
    #     src = self.pos_encoder(src)

    #     memory = self.transformer.encoder(src)
    #     return memory

# import math

# # 超参数
# input_dim = 40
# model_dim = 512
# nhead = 8
# num_encoder_layers = 6
# num_decoder_layers = 6
# dim_feedforward = 2048
# dropout = 0.1
# d_model = 128
# num_class = 3

# # 创建模型
# model = Transformer("classification",input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,d_model,num_class)

# # 输入数据
# batch_size = 32
# src = torch.rand(batch_size, 100, input_dim)
# tgt = torch.rand(batch_size, 100, input_dim)

# # 前向传播
# # output = model(src, tgt)

# # 获取encoder的隐层输出
# encoder_hidden_states = model(src)
# print(encoder_hidden_states.shape)  # 输出隐层形状
