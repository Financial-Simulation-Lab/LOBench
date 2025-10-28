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
from model.layers.Transformer_EncDec import Encoder, EncoderLayer
from model.layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.layers.Embed import DataEmbedding_inverted

class iTransformer(LOBAutoEncoder):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self,
                 seq_len,
                 output_attention,
                 d_model,
                 e_layers,
                 embed,
                 freq,
                 dropout,
                 factor,
                 n_heads,
                 d_ff,
                 activation,
                 enc_in,
                 unified_d,
                 ckpt_path=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.output_attention = output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encode_proj = encode_projection(unified_d=unified_d,inc_d=d_model * enc_in)
        
        if ckpt_path:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(ckpt_path, map_location=device)
            embed_state_dict = {k.replace("enc_embedding.", ""): v for k, v in ckpt['state_dict'].items() if "enc_embedding" in k}
            self.enc_embedding.load_state_dict(embed_state_dict)
            encoder_state_dict = {k.replace("encoder.", ""): v for k, v in ckpt['state_dict'].items() if "encoder" in k}
            self.encoder.load_state_dict(encoder_state_dict)
            encode_proj_state_dict = {k.replace("encode_proj.", ""): v for k, v in ckpt['state_dict'].items() if "encode_proj" in k}
            self.encode_proj.load_state_dict(encode_proj_state_dict)
            print("Load the parameters of LOB encoder part successfully.")
            
            for param in self.enc_embedding.parameters():
                param.requires_grad = False          
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encode_proj.parameters():
                param.requires_grad = False
                
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(unified_d, self.num_class)
        elif self.task_name == 'reconstruction':
            if self.decoder_name == 'transformer_decoder':
                decoder_layer = nn.TransformerDecoderLayer(unified_d, nhead=8, batch_first=True)
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
                self.decode_proj = decode_projection(unified_d=unified_d,seq_len=seq_len,enc_in=enc_in)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(unified_d, self.c_out, bias=True)
                    
    def encode(self,x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
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
