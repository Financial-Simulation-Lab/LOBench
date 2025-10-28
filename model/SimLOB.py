# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from model.base import LOBAutoEncoder 
from model.layers.Projection import encode_projection,decode_projection

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from metric import metrics 

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiheadAttention, self).__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Linear layers for the query, key, and value projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Linear layer for the output of the attention heads
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional mask to mask out elements in the input sequence
                  (e.g., for padding or future elements in the decoder)
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Linearly project queries, keys, and values
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # Split the queries, keys, and values into multiple heads
        q = q.view(q.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Concatenate and linearly project the attention heads
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.d_model)
        output = self.out_linear(output)

        return output

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len):
#         super(PositionalEncoding, self).__init__()

#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (seq_len, batch_size, d_model)
#         Returns:
#             Output tensor with positional encodings added.
#         """
#         # print(self.pe[:x.size(0), :].shape)
#         batch_size, seq_len, d_model = x.size()
#         x = x + self.pe.unsqueeze(0).expand(batch_size, seq_len, -1)
#         return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()

        self.linear1 = nn.Linear(d_model, 4*d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4*d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor after applying position-wise feedforward network.
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
   
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Args:
            x: Input tensor
            sublayer: Sublayer module (e.g., self-attention, feedforward)
        Returns:
            Output tensor after applying layer normalization, dropout, and the sublayer.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to mask out elements in the input sequence
                  (e.g., for padding)
        Returns:
            Output tensor after applying self-attention and position-wise feedforward.
        """
        x = self.sublayer(x, lambda x: self.self_attn(x, x, x))
        if self.feed_forward is None:
            return x  
        else:
            x = self.sublayer(x, self.feed_forward)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1,feed_forward=PositionwiseFeedforward):
        super(TransformerEncoder, self).__init__()
        if feed_forward:
            self.layers = nn.ModuleList([
                EncoderLayer(d_model, MultiheadAttention(d_model, nhead), feed_forward(d_model, dropout), dropout)
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(d_model, MultiheadAttention(d_model, nhead), feed_forward, dropout)
                for _ in range(num_layers)
            ])

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to mask out elements in the input sequence
                  (e.g., for padding)
        Returns:
            Output tensor after applying the specified number of encoder layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

class Transformer_block(nn.Module):
    def __init__(self, d_model, multi_head_num, num_layers,seq_len,d_input,d_output,feed_forward=True):
        super(Transformer_block, self).__init__()
        
        self.linear_project=nn.Linear(d_input,d_model)
        
        # self.dropout = nn.Dropout(p=0.1)
        self.pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        # self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        # self.positional_encoding = PositionalEncoding(d_model,seq_len)

        # Transformer encoder and decoder layers (as defined in previous examples)
        if feed_forward:
            self.transformer_encoder = TransformerEncoder(num_layers,d_model,multi_head_num)
        else:
            self.transformer_encoder = TransformerEncoder(num_layers,d_model,multi_head_num,feed_forward=None)

        # Output layer
        self.fc = nn.Linear(d_model, d_output)

    def forward(self, src):
        # Embedding and positional encoding
        self.pe = self.pe.to(src)
        enc_out = self.linear_project(src)
        batch_size, seq_len, d_model = enc_out.size()
        enc_out = enc_out + self.pe.unsqueeze(0).expand(batch_size, seq_len, -1)
        # src = self.positional_encoding(self.linear_project(src))
        # Transformer encoder and decoder
        memory = self.transformer_encoder(enc_out)

        # Linear layer for the final output
        output = self.fc(memory)

        return output


class Trans_Encoder(nn.Module):
    def __init__(self,trans_embed_size=128,multi_head_num=8,trans_encoder_layers=6,feed_forward=True):
        super().__init__()

        self.transformer=Transformer_block(d_model=256,
                                           multi_head_num=multi_head_num,
                                           num_layers=trans_encoder_layers,
                                           seq_len=100,
                                           d_input=40,
                                           d_output=40,
                                           feed_forward=feed_forward)
        self.linear_min1=nn.Linear(100*40,1024)
        self.linear_min2=nn.Linear(1024,512)
        self.linear_min3=nn.Linear(512,trans_embed_size)

    def forward(self, x):
        x=self.transformer(x)
        x=x.view(x.shape[0],-1)
        x=self.linear_min1(x)
        x=F.relu(x)
        x=self.linear_min2(x)
        x=F.relu(x)
        x=self.linear_min3(x)
        return x



class Trans_Decoder(nn.Module):
    def __init__(self,trans_embed_size=256,multi_head_num=8,trans_encoder_layers=6,feed_forward=True):
        super().__init__()
        self.transformer=Transformer_block(256,
                                           multi_head_num,trans_encoder_layers,
                                           100,
                                           40,
                                           40,
                                           feed_forward=feed_forward)
        self.linear_min1=nn.Linear(trans_embed_size,512)
        self.linear_min2=nn.Linear(512,1024)
        self.linear_min3=nn.Linear(1024,100*40)

    def forward(self, x):

        x=self.linear_min1(x)
        x=F.relu(x)
        x=self.linear_min2(x)
        x=F.relu(x)
        x=self.linear_min3(x)

        x=x.view(x.shape[0],100,40)
        x=self.transformer(x)
        return x


class SimLOB(LOBAutoEncoder):
    def __init__(self,
                 trans_embed_size,
                 multi_head_num,
                 trans_encoder_layer,
                 feed_forward,
                 dropout,
                 unified_d,
                 ckpt_path=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.freeze = False
        self.encoder = Trans_Encoder(trans_embed_size=trans_embed_size,
                                     multi_head_num=multi_head_num,
                                     trans_encoder_layers=trans_encoder_layer,
                                     feed_forward=feed_forward)
        self.encode_proj = encode_projection(unified_d=unified_d,inc_d=trans_embed_size)
        
        if ckpt_path:
            self.freeze = True
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(ckpt_path, map_location=device)
            encoder_state_dict = {k.replace("encoder.", "", 1): v for k, v in ckpt['state_dict'].items() if "encoder" in k}
            self.encoder.load_state_dict(encoder_state_dict)
            encode_proj_state_dict = {k.replace("encode_proj.", "", 1): v for k, v in ckpt['state_dict'].items() if "encode_proj" in k}
            self.encode_proj.load_state_dict(encode_proj_state_dict)
            print("Load the parameters of LOB encoder part successfully.")
            
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encode_proj.parameters():
                param.requires_grad = False
                
            decoder_layer = nn.TransformerDecoderLayer(unified_d, nhead=8, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(unified_d, self.num_class)
        elif self.task_name == 'reconstruction':
            if self.decoder_name == 'transformer_decoder':
                decoder_layer = nn.TransformerDecoderLayer(unified_d, nhead=8, batch_first=True)
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
                self.decode_proj = decode_projection(unified_d=unified_d,seq_len=self.seq_len,enc_in=self.enc_in)
            elif self.decoder_name == 'simlob_decoder':
                self.decoder = Trans_Decoder(trans_embed_size=trans_embed_size,
                                     multi_head_num=multi_head_num,
                                     trans_encoder_layers=trans_encoder_layer,
                                     feed_forward=feed_forward)
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
        if self.freeze:
            enc_out = enc_out.unsqueeze(1)
            memory = torch.zeros(enc_out.shape[0], enc_out.shape[1],enc_out.shape[2], device=enc_out.device)
            enc_out = self.decoder(enc_out,memory)
            enc_out = enc_out.view(enc_out.shape[0],-1)
        
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
    
    def forward(self, x_enc,x_mark_enc,mask=None):
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
    

########################################################
#######           Lighting Module            ##########
########################################################
import lightning as L 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.utilities.model_summary import ModelSummary as nofitModelSummary
from metric import NewLoss, PriceLoss, VolumeLoss, WeightedPriceLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
class Pure_Trans_AE_Lightning(L.LightningModule):
    def __init__(self,lr=1e-4,
                 trans_embed_size=256,
                 multi_head_num=2,
                 trans_encoder_layers=6,
                 feed_forward=True,
                 batch_size=128,
                 optimizer_name='Adam',
                 metrics = ['mse_loss','price_loss','volume_loss','weighted_price_loss','weighted_volume_loss'],
                 **kwargs):
        super().__init__()
        self.lr = lr 
        self.trans_embed_size = trans_embed_size
        self.multi_head_num = multi_head_num
        self.trans_encoder_layer = trans_encoder_layers
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.metrics = metrics 

        self.encoder = Trans_Encoder(trans_embed_size=self.trans_embed_size,
                                     multi_head_num=self.multi_head_num,
                                     trans_encoder_layers=self.trans_encoder_layer,
                                     feed_forward=feed_forward)
        self.decoder = Trans_Decoder(trans_embed_size=self.trans_embed_size,
                                     multi_head_num=self.multi_head_num,
                                     trans_encoder_layers=self.trans_encoder_layer,
                                     feed_forward=feed_forward)
        # self.save_hyperparameters('lr','trans_embed_size','multi_head_num','trans_encoder_layers')
        self.save_hyperparameters()
        
    def training_step(self,batch,batch_idx):
        x = batch 
        h = self.encoder(x)
        y = self.decoder(h)
        # train_loss = nn.functional.mse_loss(x,y)
        # money_loss = MoneyLoss(x,y)
        mse_loss = metrics[self.metrics[0]](x,y)
        # train_loss = self.alpha * money_loss + (1-self.alpha) * price_loss
        # self.log("train_loss",train_loss,sync_dist=True,prog_bar=True)
        # self.log("mse_loss",mse_loss,sync_dist=True,prog_bar=True)
        # self.log("money_loss",money_loss,sync_dist=True,prog_bar=True)
        
        if batch_idx % 50 == 0:
            self.logger_metrics(x,y)
        return mse_loss 
    
    
    def validation_step(self,batch,batch_idx):
        
        x = batch 
        h = self.encoder(x)
        y = self.decoder(h)
        val_loss = metrics[self.metrics[0]](x,y)
        self.logger_metrics(x,y)
        return val_loss
        
        
    def test_step(self,batch,batch_idx):
        x = batch 
        h = self.encoder(x)
        y = self.decoder(h)
        test_loss = metrics[self.metrics[0]](x,y)
        self.logger_metrics(x,y)
        return test_loss
            
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return {
                "optimizer": optimizer
                }
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 
    
    def logger_metrics(self,src,output,log_type="train_"):
        # logs = self.main_metrics(src,output,log_type)
        logs = {}
        
        # price,volume = output

        with torch.no_grad():
            for mtc in self.metrics:
                if 'price' in mtc:
                    tgt = output[:,:,:20]
                elif 'volume' in mtc:
                    tgt = output[:,:,20:]
                else:
                    tgt = output 
                loss = metrics[mtc](src,tgt)
                logs[log_type+mtc] = loss
        self.logger.log_metrics(logs,step = self.global_step)
        
#======================  end   =========================


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    from config import ExpConfigManager as ECM 
    from data.data_prepare import create_dataloaders_from_files
    from lightning.pytorch.loggers import CometLogger
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    exp_setting_folder = "./experiment/exp_settings/AE_PV"
    ecm = ECM(exp_setting_folder,'compare_simlob.json')
    
    train_loader, valid_loader, test_loader = create_dataloaders_from_files(ecm.data.data_file,batch_size=ecm.data.batch_size,workers=ecm.data.workers,split_seed=ecm.data.manual_seed)
    m = Pure_Trans_AE_Lightning(batch_size=ecm.model.batch_size)
    
    comet_logger = CometLogger(
                            api_key=ecm.comet.api_key,  # Optional
                            workspace=ecm.comet.workspace,  # Optional
                            save_dir=ecm.data.log_folder,  # Optional
                            project_name=ecm.comet.project_name,  # Optional
                            experiment_name=ecm.experiment.name+"_"+ecm.experiment.version,  # Optional
                            # experiment_key=ecm.comet.experiment_key # Optional
                        )
    trainer = L.Trainer(max_epochs=ecm.train.max_epochs,
                    profiler="advanced",
                    logger=comet_logger,
                    num_sanity_val_steps=2,
                    devices=[0,1])
    
    trainer.fit(m,train_loader,valid_loader)
    # trainer.test(m,dataloaders=test_loader)