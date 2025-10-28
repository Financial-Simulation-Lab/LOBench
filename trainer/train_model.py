import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import torch.utils 
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader,random_split



import sys
import os
import argparse
# from Pure_Trans_AE_1286 import Pure_Trans_AE


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Append the parent directory to sys.path
sys.path.append(parent_dir)

#import your model
from model.linear_autoencoder import Linear_AE

# Argument parser
parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--data_path', type=str, required=True, help='The location of train data')
parser.add_argument('--split_label_path', type=str, required=True, help='The location of split label data')
parser.add_argument('--save_folder', type=str, required=True, help='The location to save the model parameters')
args = parser.parse_args()

data_path = args.data_path
split_label_path = args.split_label_path
save_folder = args.save_folder

#检查是否连接上gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_tensor = torch.load(data_path)

class MyDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, x):
        """Initialization""" 
        self.x=x

    def __len__(self):
        """Denotes the total number of samples"""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index]
    
with open(split_label_path, 'rb') as f:
  loaded_lists = pickle.load(f)

train_indices, valid_indices = loaded_lists  # Unpack the loaded lists

batch_size = 128

# train_dataset, test_dataset = random_split(data_tensor, [0.8, 0.2])

dataset_train = MyDataset(data_tensor[train_indices])
dataset_val = MyDataset(data_tensor[valid_indices])

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False,pin_memory=True)


# Variables to keep track of the best validation performance
num_epochs=20
# model=CNNLSTM_AE()
from model.linear_autoencoder import Linear_AE
from model.TransLOB import Trans_LOB
from model.Trans_LOB_large import Trans_LOB_large
from model.Pure_Trans_AE_1286 import Pure_Trans_AE
from model.LSTM import LSTM
from model.Pure_Trans_AE_128_2 import Pure_Trans_AE_2
from model.Pure_Trans_AE_128_4 import Pure_Trans_AE_4 

model=Pure_Trans_AE_4()
# model=Trans_LOB_large()
# model = linear_AE()

# model = LSTM_autoencoder()
# continue training
# checkpoint = torch.load('/root/workspace/model_parameter/pure_trans_1286_para/patch_trans_80_checkpoint_new.pth')
# model=linear_AE()
# model = Pure_Trans_AE()
model.to(device)
model = nn.DataParallel(model, device_ids=[0,1])
# model.load_state_dict(checkpoint['model_state_dict'])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# best_val_loss = checkpoint['best_val_loss']
best_val_loss=1e5
best_val_epoch=0
# best_val_epoch=checkpoint['epoch']
# initial_epoch=checkpoint['epoch']

new_learning_rate = 1e-4  # Replace this with your desired learning rate

# Update the learning rate in the optimizer
for param_group in optimizer.param_groups:
    param_group['lr'] = new_learning_rate

# Training loop
for it in tqdm(range(num_epochs)):
    # it=it+checkpoint['epoch']
    # Training
    # it+=(checkpoint['epoch']+1)
    model.train()
    t0 = datetime.now()
    train_loss = []
    for inputs in train_loader:
        # Forward pass, backward pass, and optimization
        inputs = inputs.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,inputs)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_loss = np.mean(train_loss)

    # Validation
    model.eval()
    val_loss = []
    with torch.no_grad():
        for inputs in val_loader:
            # Compute validation loss
            inputs = inputs.to(device, dtype=torch.float)   
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss.append(loss.item())
    val_loss = np.mean(val_loss)

    #Save the model every 20 epochs
    if (it + 1) % 5 == 0:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'epoch': it,
        }
        torch.save(checkpoint, save_folder+f'/patch_trans_{it+1}_checkpoint.pth')  # Use a dynamic filename
        print(f'model saved at epoch {it + 1}')

    # Check if the current model has the best validation performance
        
    if val_loss < best_val_loss:
        # Save the model and optimizer parameters
        best_val_loss = val_loss
        best_val_epoch=it+1
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'epoch': it,
        }
        torch.save(checkpoint, save_folder+f'/patch_trans_best_checkpoint_new.pth')
        print('model saved')
    
    dt = datetime.now() - t0
    print(f'Epoch {it+1}/{num_epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {val_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_val_epoch}')

    # Save train and validation loss to file
    with open(save_folder+f'/loss_history.txt', 'a') as f:
         f.write(f'Epoch {it+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n')


# Pure_trans_ae
## python model/train_model.py --data_path data/train_data/pv_100w.pt --split_label_path data/train_data/pv_100w_split.pkl --save_folder model/trained_models


# Linear_ae
## python model/train_model.py --data_path data/train_data/pv_100w.pt --split_label_path data/train_data/pv_100w_split.pkl --save_folder model/trained_models/linear_ae


# LSTM
## python model/train_model.py --data_path data/train_data/pv_100w.pt --split_label_path data/train_data/pv_100w_split.pkl --save_folder model/trained_models/lstm_ae


# LSTM
## python model/train_model.py --data_path data/train_data/pv_100w.pt --split_label_path data/train_data/pv_100w_split.pkl --save_folder model/trained_models/pure_trans_ae_2