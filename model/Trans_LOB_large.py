# Transformers for limit order books
# Source: https://arxiv.org/pdf/2003.00130.pdf

import pytorch_lightning as pl
import torch
import torch.nn as nn


class TransLob_encoder(pl.LightningModule):
    def __init__(self, seq_len=100, in_c=40, out_c=64, n_attlayers=7, n_heads=8, dim_linear=128, dim_feedforward=1024):
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
            nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=2, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_c, out_channels=out_c, kernel_size=2, dilation=2, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_c, out_channels=out_c, kernel_size=2, dilation=4, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_c, out_channels=out_c, kernel_size=2, dilation=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_c, out_channels=out_c, kernel_size=2, dilation=16, padding="same"),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(0.1)

        self.activation = nn.ReLU()

        d_model = out_c+1
        dim_hidden=256
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_hidden, nhead=n_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=0.0, batch_first=True, device='cuda')

        self.layer_norm = nn.LayerNorm([seq_len, out_c])

        self.transformer = nn.TransformerEncoder(self.encoder_layer, n_attlayers)

        self.fc1 = nn.Linear(seq_len * d_model, dim_linear)

        self.fc_mida = nn.Linear(d_model, dim_hidden)

        self.fc_midb = nn.Linear(dim_hidden, d_model)


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        # Pass the input tensor through a series of convolutional layers
        x = self.conv(x)

        # Permute the dimensions of the output from the convolutional layers so that the second dimension becomes the first
        x = x.permute(0, 2, 1)

        # Normalize the output from the convolutional layers
        x = self.layer_norm(x)

        # Apply positional encoding to the output from the layer normalization
        x = self.positional_encoding(x)

        x=self.fc_mida(x)

        # Pass the output from the previous steps through the transformer encoder
        x = self.transformer(x)

        x=self.fc_midb(x)

        # Reshape the output from the transformer encoder to have only two dimensions
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

        # Apply dropout and activation function to the output from the previous step, then pass it through the first linear layer
        x = self.dropout(self.activation(self.fc1(x)))

        return x

    @staticmethod
    def positional_encoding(x):
        n_levels = 100
        pos = torch.arange(0, n_levels, 1, dtype=torch.float32) / (n_levels - 1)
        pos = (pos + pos) - 1
        # pos = np.reshape(pos, (pos.shape[0]))
        pos_final = torch.zeros((x.shape[0], n_levels, 1), dtype=torch.float32, device='cuda')
        for i in range(pos_final.shape[0]):
            for j in range(pos_final.shape[1]):
                pos_final[i, j, 0] = pos[j]

        x = torch.cat((x, pos_final), 2)
        return x
    
class TransLob_decoder(pl.LightningModule):
    def __init__(self, seq_len=100, in_c=40, out_c=64, n_attlayers=7, n_heads=8, dim_linear=128, dim_feedforward=1024):
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
            nn.Conv1d(in_channels=out_c+1, out_channels=in_c, kernel_size=2, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=2, dilation=2, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=2, dilation=4, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=2, dilation=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=2, dilation=16, padding="same"),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(0.1)

        self.activation = nn.ReLU()

        d_model = out_c+1
        dim_hidden=256
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_hidden, nhead=n_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=0.0, batch_first=True, device='cuda')

        self.layer_norm = nn.LayerNorm([seq_len, d_model])

        self.transformer = nn.TransformerEncoder(self.encoder_layer, n_attlayers)

        self.fc1 = nn.Linear(dim_linear, seq_len * (d_model-1))

        self.fc_mida = nn.Linear(64, dim_hidden-1)

        self.fc_midb = nn.Linear(dim_hidden, d_model)

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x))) #bsz 100*40

        # Reshape the output from the transformer encoder to have only two dimensions
        x = torch.reshape(x, (x.shape[0], 100,64)) 

        x=self.fc_mida(x)

        # Apply positional encoding to the output from the layer normalization
        x = self.positional_encoding(x) 

        # Pass the output from the previous steps through the transformer encoder
        x = self.transformer(x) 

        x=self.fc_midb(x)

        # Normalize the output from the convolutional layers
        x = self.layer_norm(x)

        x = torch.permute(x, (0, 2, 1))

        # Pass the input tensor through a series of convolutional layers
        x = self.conv(x)

        x = torch.permute(x, (0, 2, 1))

        return x

    @staticmethod
    def positional_encoding(x):
        n_levels = 100
        pos = torch.arange(0, n_levels, 1, dtype=torch.float32) / (n_levels - 1)
        pos = (pos + pos) - 1
        # pos = np.reshape(pos, (pos.shape[0]))
        pos_final = torch.zeros((x.shape[0], n_levels, 1), dtype=torch.float32, device='cuda')
        for i in range(pos_final.shape[0]):
            for j in range(pos_final.shape[1]):
                pos_final[i, j, 0] = pos[j]

        x = torch.cat((x, pos_final), 2)
        return x

class Trans_LOB_large(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=TransLob_encoder()
        self.decoder=TransLob_decoder()
    
    def forward(self,x):
        encoded=self.encoder(x)
        output=self.decoder(encoded)
        return output

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Trans_LOB_large().to(device)

    input = torch.randn(36,100, 40).to(device)
    with torch.no_grad():
        output = model.encoder(input)
    print(output.shape)
    print(output)