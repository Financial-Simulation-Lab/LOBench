from torch import nn
import torch.nn.functional as F

class decode_projection(nn.Module):
    def __init__(self, unified_d, seq_len, enc_in, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.linear_min1=nn.Linear(unified_d,512)
        self.linear_min2=nn.Linear(512,1024)
        self.linear_min3=nn.Linear(1024,self.seq_len * self.enc_in)
        
    def forward(self, x):
        x=x.view(x.shape[0],-1)
        x=self.linear_min1(x)
        x=F.relu(x)
        x=self.linear_min2(x)
        x=F.relu(x)
        x=self.linear_min3(x)

        x=x.view(x.shape[0],self.seq_len, self.enc_in)
        return x
    
class encode_projection(nn.Module):
    def __init__(self, unified_d, inc_d, **kwargs):
        super().__init__(**kwargs)
        self.linear_min1=nn.Linear(inc_d,1024)
        self.linear_min2=nn.Linear(1024,512)
        self.linear_min3=nn.Linear(512,unified_d)
        
    def forward(self, x):
        x=x.view(x.shape[0],-1)
        x=self.linear_min1(x)
        x=F.relu(x)
        x=self.linear_min2(x)
        x=F.relu(x)
        x=self.linear_min3(x)
        return x