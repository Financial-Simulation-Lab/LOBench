"""
Created on : 2024-08-14
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/model/base.py
Description: Base Model of all the models
---
Updated    : 
---
Todo       : 
"""
# -----------------Project Modules -----------------
# Insert the path into sys.path for importing.
import comet_ml

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import Config 
import data 
from metric import metrics
from utils.file import merge_represented_files, load_represented_data
    
# -------------Python Standard Packages ------------
from typing import Optional,Dict,Any,Union,Protocol, Tuple, List
import importlib
import argparse
from tqdm import tqdm 
import random

# -------------3rd Party Packages ------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L 
from lightning.pytorch.loggers import CometLogger
import argparse
import pandas as pd 
import numpy as np



class LOBModel(nn.Module):
    log_freq:int = 50
    optimizer_name:str = "Adam"
    
    def __init__(self, *args,output_shape:Tuple=None, input_shape: Tuple=None, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if input_shape is not None and output_shape is not None:
            self._check_model_data(input_shape, output_shape)
        
    def forward(self, x):
        raise NotImplementedError("This method should be implemented in the child class")

    def _check_model_data(self, in_shape:Tuple, out_shape:Tuple):
        sample = torch.randn(in_shape)
        out = self.forward(sample)

        if out.shape != out_shape:
            raise ValueError(f"Output shape is not equal to the output shape: {out_shape}.")
        
        print("Model is working fine.")
        


class LOBAutoEncoder(L.LightningModule):
    def __init__(self,*args,**kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def forward(self, x):
        raise NotImplementedError("This method should be implemented in the child class")
    
    def training_step(self, batch, batch_idx):
        src,y,src_mark = batch
        if self.task_name == 'imputation':
            B, T, N = src.shape
            mask_len = int(self.mask_rate * T)
            mask = torch.ones((B, T),dtype=torch.bool).to(self.device)
            for i in range(B):
                start_idx = random.randint(0, T-mask_len)
                mask[i, start_idx:start_idx+mask_len] = 0
            mask = mask.unsqueeze(-1).expand(-1, -1, N)   
            inp = src.masked_fill(mask == 0, 0)
            input_x = inp
            target = src
        if self.task_name == 'classification':
            target = y
            input_x = src
        elif self.task_name == 'reconstruction':
            target = src
            input_x = src
        output = self.forward(input_x,src_mark)
        if self.task_name == 'imputation':
            return self.logger_metrics(output,target,mask,"train_") 
        return self.logger_metrics(output,target,None,"train_")  

    def validation_step(self, batch, batch_idx):
        src,y,src_mark = batch
        if self.task_name == 'imputation':
            B, T, N = src.shape
            mask_len = int(self.mask_rate * T)
            mask = torch.ones((B, T),dtype=torch.bool).to(self.device)
            for i in range(B):
                start_idx = random.randint(0, T-mask_len)
                mask[i, start_idx:start_idx+mask_len] = 0
            mask = mask.unsqueeze(-1).expand(-1, -1, N) 
            inp = src.masked_fill(mask == 0, 0)
            input_x = inp
            target = src
        if self.task_name == 'classification':
            target = y
            input_x = src
        elif self.task_name == 'reconstruction':
            target = src
            input_x = src
        output = self.forward(input_x,src_mark)
        if self.task_name == 'imputation':
            return self.logger_metrics(output,target,mask,"val_") 
        return self.logger_metrics(output,target,None,"val_")

    def test_step(self, batch, batch_idx):
        src,y,src_mark = batch
        if self.task_name == 'imputation':
            B, T, N = src.shape
            mask_len = int(self.mask_rate * T)
            mask = torch.ones((B, T),dtype=torch.bool).to(self.device)
            for i in range(B):
                start_idx = random.randint(0, T-mask_len)
                mask[i, start_idx:start_idx+mask_len] = 0
            mask = mask.unsqueeze(-1).expand(-1, -1, N)   
            inp = src.masked_fill(mask == 0, 0)
            input_x = inp
            target = src
        if self.task_name == 'classification':
            target = y
            input_x = src
        elif self.task_name == 'reconstruction':
            target = src
            input_x = src
        output = self.forward(input_x,src_mark)
        if self.task_name == 'imputation':
            return self.logger_metrics(output,target,mask,"test_") 
        return self.logger_metrics(output,target,None,"test_")
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), lr=self.lr)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
        return {"optimizer": optimizer}

    def compute_loss(self, output, target, log_type = "train_"):
        loss_func = metrics[self.metrics[0]]
        if log_type == "train_":
            loss = loss_func(output, target)
        else:
            with torch.no_grad():
                loss = loss_func(output, target)
        return loss
    
    def logger_metrics(self,src,tgt,mask=None,log_type="train_"):
        loss = self.compute_loss(src,tgt)
        logs = {
            log_type+self.metrics[0]: loss
        }
        with torch.no_grad():
            for mtc in self.metrics[1:]:
                if mask is None:
                    m_loss = metrics[mtc](src,tgt)
                else:
                    m_loss = metrics[mtc](src,tgt,mask)
                logs[log_type+mtc] = m_loss
                
        if log_type == "train_":
            if self.global_step % self.log_freq == 0:
                self.logger.log_metrics(logs,step = self.global_step)   
        else:
            self.logger.log_metrics(logs,step = self.global_step)   
        return loss 
        
class ExperimentFactory:
    def __init__(self, config_file_path, data_config='experiment/exp_settings/default.json', model_path=None):
        self.config = Config(data_config)
        self.config_file_path = config_file_path
        self.multi_task = not os.path.isfile(config_file_path)
        self.model_path = model_path
        self.config.update_config(self.config_file_path)
        self.create_dataloaders()

    def create_model(self):
        self.model_params = self.config.get_dict('model')
        self.model_name = self.model_params['name']

        model_module = importlib.import_module("model")
        self.model_class = getattr(model_module, self.model_params['name'])

        if self.model_path is None:
            model_instance = self.model_class(**self.model_params)
            self.model = model_instance
        else:
            self.model = self.model_class.load_from_checkpoint(self.model_path,**self.model_params)  
    
    def create_dataloaders(self):
        dataset_params = self.config.get_dict('data.dataset')
        
        dataloader_params = self.config.get_dict('data.dataloader')
        dataset_class = getattr(data, dataset_params['name'])
        dataloader_class = getattr(data, dataloader_params['name'])
        
        ds = dataset_class(**dataset_params)
        dl = dataloader_class(datasets=ds, **dataloader_params)
        self.data_module = dl 
        # dataset_module = importlib.import_module('data')
        self.train_dataloader = dl.train_dataloader()
        self.val_dataloader = dl.val_dataloader()
        self.test_dataloader = dl.test_dataloader()

        self.represented_path = dataset_params['represented_data_path']
        self.dataset_path = dataset_params['dataset_path']
        self.dataset_name = os.path.basename(self.dataset_path).split('.')[0]
        
    def create_trainer(self):
        logger_params = self.config.get_dict('comet')
        trainer_params = self.config.get_dict('trainer')
        self.logger = CometLogger(**logger_params)
        self.trainer = L.Trainer(
            strategy='ddp_find_unused_parameters_true',
            # strategy='ddp_notebook',
            logger=self.logger,
                                 **trainer_params)

    def save_hidden_variables(self):
        self.represented_file = os.path.join(self.represented_path, f'{self.dataset_name}-{self.model_name}.pt')
        self.model.eval()
        temp_dir = os.path.join(self.represented_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        count = 0
        batch_files = []
        hiddens = []
        y_labels = []
        for data in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            
            for batch in tqdm(data, desc="Processing batches", unit="batch"):
                X,y,_ = batch
                with torch.no_grad():
                    hidden = self.model.encode(X)
                    hiddens.append(hidden)
                    y_labels.append(y)
                # batch_file = os.path.join(temp_dir,f'batch_{count}.pt')
                # torch.save((hidden.cpu(),y.cpu()),batch_file)
                # batch_files.append(batch_file)

        # merge_represented_files(batch_files, self.represented_file)
        hidden_vas = torch.cat(hiddens, dim=0)
        labels = torch.cat(y_labels, dim=0)
        torch.save((hidden_vas, labels), self.represented_file)
        print(f"Hidden variables saved to {self.represented_file}")



    # Example of loading hidden variables
    def load_hidden_dataset(self, file_path, batch_size=128, shuffle=True):
        represented_dataset = load_represented_data(file_path)
        represented_dataset = TensorDataset(*represented_dataset)
        self.represented_dataloader = DataLoader(represented_dataset, batch_size=batch_size, shuffle=shuffle)
        return self.represented_dataloader
    
    def start_training(self,config_file):
        # print("Config file:", config_file)
        # self.config.update_config(config_file)

        self.create_model()
        self.create_trainer()
        # self.config.save()
        self.logger.experiment.log_asset("experiment/exp_settings/default.json")
        self.logger.experiment.log_asset(config_file)
        self.trainer.fit(self.model, 
                         datamodule=self.data_module)
                        #  datamodule=self.data_module)
        
        self.trainer.test(self.model, self.test_dataloader)

        # self.logger.finalize(status=None)
        # print(f"Begin saving hidden variables in {self.represented_path}")
        # self.save_hidden_variables()
        # print(f"End saving hidden variables in {self.represented_path}")
        self.trainer.test(self.model, self.test_dataloader)
        
        
    
    def run(self):
        torch.set_float32_matmul_precision('high')
        if not self.multi_task:
            self.start_training(self.config_file_path)
        else:
            # print(os.listdir(self.config_file_path))
            for file in os.listdir(self.config_file_path):
                config_file = os.path.join(self.config_file_path, file)
                if config_file.endswith('.json'):
                    self.start_training(config_file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='SimLOB Model')
    parser.add_argument('-c','--config', type=str, help='Path to the config file')
    args = parser.parse_args()
    
    mf = ExperimentFactory(args.config)
    mf.run()
    
    
    
    