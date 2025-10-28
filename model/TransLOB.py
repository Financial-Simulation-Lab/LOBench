# Transformers for limit order books
# Source: https://arxiv.org/pdf/2003.00130.pdf

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
import math 

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         self.pe = self.pe.to(x)
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
    

class TransLob_encoder(LOBModel):
    def __init__(self, seq_len=100, 
                 enc_in=40, 
                 trans_embed_size=256, 
                 trans_encoder_layers=2, 
                 multi_head_num=4, 
                 d_model=256, 
                 feed_forward_dim=256,
                 **kwargs):
        super().__init__()

        '''
        Args:
          in_c: the number of input channels for the first Conv1d layer in the CNN
          out_c: the number of output channels for all Conv1d layers in the CNN
          seq_len: the sequence length of the input data
          n_attlayers: the number of attention layers in the transformer encoder
          n_heads: the number of attention heads in the transformer encoder
          dim_linear: the number of neurons in the first linear layer (fc1)
          dim_feedforward: the number of neurons in the feed-forward layer of the transformer encoder layer
          dropout: the dropout rate for the Dropout layer
        '''

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=enc_in, out_channels=trans_embed_size, kernel_size=2, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=trans_embed_size, out_channels=trans_embed_size, kernel_size=2, dilation=2, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=trans_embed_size, out_channels=trans_embed_size, kernel_size=2, dilation=4, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=trans_embed_size, out_channels=trans_embed_size, kernel_size=2, dilation=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=trans_embed_size, out_channels=trans_embed_size, kernel_size=2, dilation=16, padding="same"),
            nn.ReLU(),
        )
        
        self.dropout = nn.Dropout(p=0.1)
        self.pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        # self.positional_encoding = PositionalEncoding(d_model)

        self.activation = nn.ReLU()

        d_model = trans_embed_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=multi_head_num,
                                                        dim_feedforward=feed_forward_dim,
                                                        dropout=0.0, 
                                                        batch_first=True)

        self.layer_norm = nn.LayerNorm([seq_len, trans_embed_size])

        self.transformer = nn.TransformerEncoder(self.encoder_layer, trans_encoder_layers)

        self.fc1 = nn.Linear(seq_len * d_model, d_model)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        # Pass the input tensor through a series of convolutional layers
        x = self.conv(x)

        # Permute the dimensions of the output from the convolutional layers so that the second dimension becomes the first
        x = x.permute(0, 2, 1)

        # Normalize the output from the convolutional layers
        x = self.layer_norm(x)

        # Apply positional encoding to the output from the layer normalization
        self.pe = self.pe.to(x)
        x = x + self.pe[:x.size(0), :]

        # Pass the output from the previous steps through the transformer encoder
        x = self.transformer(x)

        # Reshape the output from the transformer encoder to have only two dimensions
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

        # # Apply dropout and activation function to the output from the previous step, then pass it through the first linear layer
        # x = self.dropout(self.activation(self.fc1(x)))

        return x

    # @staticmethod
    # def positional_encoding(x):
    #     device = x.device 
    #     n_levels = 100
    #     pos = torch.arange(0, n_levels, 1, dtype=torch.float32) / (n_levels )
    #     pos = (pos + pos)         # pos = np.reshape(pos, (pos.shape[0]))
    #     pos_final = torch.zeros((x.shape[0], n_levels, 1), dtype=torch.float32, device=device)
    #     for i in range(pos_final.shape[0]):
    #         for j in range(pos_final.shape[1]):
    #             pos_final[i, j, 0] = pos[j]

    #     # x = torch.cat((x, pos_final), 2)
    #     x+=pos_final
    #     return x
    
class TransLob_decoder(LOBModel):
    def __init__(self, max_len=100, in_features=40, trans_embed_size=256, trans_decoder_layers=2, multi_head_num=4, d_model=256, feed_forward_dim=256,**kwargs):
        super().__init__()

        '''
        Args:
          in_c: the number of input channels for the first Conv1d layer in the CNN
          out_c: the number of output channels for all Conv1d layers in the CNN
          seq_len: the sequence length of the input data
          n_attlayers: the number of attention layers in the transformer encoder
          n_heads: the number of attention heads in the transformer encoder
          dim_linear: the number of neurons in the first linear layer (fc1)
          dim_feedforward: the number of neurons in the feed-forward layer of the transformer encoder layer
          dropout: the dropout rate for the Dropout layer
        '''

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=trans_embed_size, out_channels=in_features, kernel_size=2, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=2, dilation=2, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=2, dilation=4, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=2, dilation=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=2, dilation=16, padding="same"),
            nn.ReLU(),
        )
        
        self.dropout = nn.Dropout(p=0.1)
        self.pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        
        # self.positional_encoding = PositionalEncoding(d_model)

        self.activation = nn.ReLU()

        d_model = trans_embed_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=multi_head_num,
                                                        dim_feedforward=feed_forward_dim,
                                                        dropout=0.0, batch_first=True)

        self.layer_norm = nn.LayerNorm([max_len, d_model])

        self.transformer = nn.TransformerEncoder(self.encoder_layer, trans_decoder_layers)

        self.fc1 = nn.Linear(d_model, max_len * (d_model))

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x))) #bsz 100*40

        # Reshape the output from the transformer encoder to have only two dimensions
        x = torch.reshape(x, (x.shape[0], 100, -1)) 

        # Apply positional encoding to the output from the layer normalization
        self.pe = self.pe.to(x)
        x = x + self.pe[:x.size(0), :]

        # Pass the output from the previous steps through the transformer encoder
        x = self.transformer(x) 

        # Normalize the output from the convolutional layers
        x = self.layer_norm(x)

        x = torch.permute(x, (0, 2, 1))

        # Pass the input tensor through a series of convolutional layers
        x = self.conv(x)

        x = torch.permute(x, (0, 2, 1))

        return x

    # @staticmethod
    # def positional_encoding(x):
    #     device = x.device 
    #     n_levels = 100
    #     pos = torch.arange(0, n_levels, 1, dtype=torch.float32) / (n_levels )
    #     pos = (pos + pos) 
    #     # pos = np.reshape(pos, (pos.shape[0]))
    #     pos_final = torch.zeros((x.shape[0], n_levels, 1), dtype=torch.float32, device=device)
    #     for i in range(pos_final.shape[0]):
    #         for j in range(pos_final.shape[1]):
    #             pos_final[i, j, 0] = pos[j]

    #     # x = torch.cat((x, pos_final), 2)
    #     x+=pos_final
    #     return x

class TransLOB(LOBAutoEncoder):
    def __init__(self,d_model,seq_len,unified_d,ckpt_path=None, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.encoder=TransLob_encoder(**kwargs)
        self.encode_proj = encode_projection(unified_d=unified_d,inc_d=d_model * seq_len)
        
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
                self.decode_proj = decode_projection(unified_d=unified_d,seq_len=seq_len,enc_in=self.enc_in)
            elif self.decoder_name == 'translob_decoder':
                self.decoder=TransLob_decoder(**kwargs)
                self.decode_proj = decode_projection(unified_d=unified_d,seq_len=seq_len,enc_in=self.enc_in)
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
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
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
    
    def forward(self,x_enc, x_mark_enc, mask=None):
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
