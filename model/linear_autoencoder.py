import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.base import LOBModel, LOBAutoEncoder


from metric import metrics

class line_encoder(LOBModel):
    def __init__(self):
        super().__init__()

        self.linear_min1 = nn.Linear(100 * 40, 2048)
        self.linear_min2 = nn.Linear(2048, 1024)
        self.linear_min3 = nn.Linear(1024, 512)
        self.linear_min4 = nn.Linear(512, 256)
        self.linear_min5 = nn.Linear(256, 128)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.linear_min1(x)
        x = F.relu(x)
        x = self.linear_min2(x)
        x = F.relu(x)
        x = self.linear_min3(x)
        x = F.relu(x)
        x = self.linear_min4(x)
        x = F.relu(x)
        x = self.linear_min5(x)

        return x


class line_decoder(LOBModel):
    def __init__(self):
        super().__init__()
        self.linear_min1 = nn.Linear(128, 256)
        self.linear_min2 = nn.Linear(256, 512)
        self.linear_min3 = nn.Linear(512, 1024)
        self.linear_min4 = nn.Linear(1024, 2048)
        self.linear_min5 = nn.Linear(2048, 40 * 100)

    def forward(self, x):
        x = self.linear_min1(x)
        x = F.relu(x)
        x = self.linear_min2(x)
        x = F.relu(x)
        x = self.linear_min3(x)
        x = F.relu(x)
        x = self.linear_min4(x)
        x = F.relu(x)
        x = self.linear_min5(x)
        x = torch.reshape(x, (x.shape[0], 100, 40))

        return x

class Linear_AE(LOBAutoEncoder):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = line_encoder()  # Instantiate line_encoder
        self.decoder = line_decoder()  # Instantiate line_decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
