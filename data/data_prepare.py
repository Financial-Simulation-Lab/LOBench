"""
Created on : 2024-08-05
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/data/data_prepare.py
Description: Prepare the data for the experiment
---
Updated    : 
---
Todo       : 
"""
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torch.utils.data import DataLoader
from torch.utils import data

import torch 
from typing import Optional,List,Tuple,Dict,Union,Protocol

def create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size, workers):
    train_loader = DataLoader(train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, num_workers=workers, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=workers, batch_size=batch_size, pin_memory=True)
    return train_loader, valid_loader, test_loader


def create_dataloaders_from_tensor(data_tensor, split_seed,batch_size, workers):
    split_seed=torch.Generator().manual_seed(split_seed)

    train_set_size = int(len(data_tensor) * 0.7)
    valid_set_size = int(len(data_tensor) * 0.2)
    test_set_size = len(data_tensor) - train_set_size - valid_set_size
    
    
    train_dataset, valid_dataset, test_dataset = data.random_split(data_tensor,[train_set_size,valid_set_size,test_set_size],generator=split_seed)
    
    return create_data_loaders(train_dataset, valid_dataset, test_dataset,batch_size=batch_size, workers=workers)

def create_dataloaders_from_files(data_file, split_seed,batch_size, workers):
    data_tensor = torch.load(data_file)
    
    return create_dataloaders_from_tensor(data_tensor, split_seed=split_seed,batch_size=batch_size, workers=workers)

