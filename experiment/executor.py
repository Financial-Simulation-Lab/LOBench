"""
Created on : 2024-08-06
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/experiment/executor.py
Description: Execute the experiment from the configuration file.
---
Updated    : 
---
Todo       : 
"""

# Standard Library
import sys
import os
import argparse
from typing import Optional,List,Tuple,Dict,Union,Protocol

# Comet
import comet_ml

# Custom Modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from utils.logger import setup_logger
from utils.file import create_or_pass
from data.data_prepare import create_dataloaders_from_files
from metric.loss_func import NewLoss, PriceLoss, MoneyLoss, VolumeLoss, WeightedPriceLoss
from config import ExpConfigManager as ECM
from config import ConfigManager as CM 
from model import *

# Torch/Lightning modules 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as Data 
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint


# Evironment Settings
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
cm = CM()

exp_setting_folder = 'experiment/exp_settings/AE_PV'# os.path.join(cm.folders.exp_settings,'AE_PV')

ecm = ECM('experiment/exp_settings/AE_PV','default.json')


class Executor:
    ecm = ecm 
    def __init__(self, exp_setting_folder:str=exp_setting_folder):
        self.exp_setting_folder = exp_setting_folder
        
        self.exp_logger = setup_logger('experiment',os.path.dirname(exp_setting_folder))
        
        self.experiments = [config for config in os.listdir(exp_setting_folder) if config.endswith('.json')]
        
        self.current_data_file = ecm.data.data_file
        
        self.load_data(init=True)
        
    def load_data(self,init=False):
        # 如果数据文件没有改变，不需要重新加载
        if not init and ecm.data.data_file == self.current_data_file:
            return
        
        self.train_loader, self.valid_loader, self.test_loader = create_dataloaders_from_files(
            ecm.data.data_file,
            batch_size=ecm.data.batch_size,
            workers=ecm.data.workers,
            split_seed=ecm.data.manual_seed)
        
    def _execute_single_exp(self,config_file:str):
        
        ecm.load_config(config_file)
        model_name = ecm.model.model_name
        
        if model_name == 'NewLOB_AE':
            model = NewLOB_AE(batch_size=ecm.model.batch_size,
                        d_model=ecm.model.d_model,
                        max_len=ecm.model.max_len,
                        in_features=ecm.model.in_features,
                        lr=ecm.model.learning_rate,
                        trans_embed_size=ecm.model.feed_forward_dim,
                        multi_head_num=ecm.model.multi_head_num,
                        trans_encoder_layers=ecm.model.trans_encoder_layers,
                        trans_decoder_layers =ecm.model.trans_decoder_layers,
                        feed_forward_dim=ecm.model.feed_forward_dim,
                        optimizer_name= ecm.model.optimizer_name,
                        metrics=ecm.model.metrics,
                        )
            
        elif model_name == 'NewLOB_AE_PV':
            model = NewLOB_AE_PV(batch_size=ecm.model.batch_size,
                        d_model=ecm.model.d_model,
                        max_len=ecm.model.max_len,
                        in_features=ecm.model.in_features,
                        lr=ecm.model.learning_rate,
                        trans_embed_size=ecm.model.feed_forward_dim,
                        multi_head_num=ecm.model.multi_head_num,
                        trans_encoder_layers=ecm.model.trans_encoder_layers,
                        trans_decoder_layers =ecm.model.trans_decoder_layers,
                        feed_forward_dim=ecm.model.feed_forward_dim,
                        optimizer_name= ecm.model.optimizer_name,
                        metrics=ecm.model.metrics,
                        )
        
        else:
            try:
                #TODO:
                raise "Not supported Model name."
            except Exception as e:
                print(f"Model {model_name} is not supported yeat.\n Error: {e}")
                self.exp_logger.error(f"Model {model_name} is not supported yeat.")
                self.exp_logger.error(f"Error:{e}")
                return 

        comet_logger = CometLogger(
                        api_key=ecm.comet.api_key,  # Optional
                        workspace=ecm.comet.workspace,  # Optional
                        save_dir=ecm.data.log_folder,  # Optional
                        project_name=ecm.comet.project_name,  # Optional
                        experiment_name=ecm.experiment.name+ecm.experiment.version)
    
        trainer = L.Trainer(max_epochs=ecm.train.max_epochs,
                            profiler="advanced",
                            # callbacks=[EarlyStopping(monitor="train_volume_loss", mode="min",stopping_threshold=1e-5)],
                            # callbacks=[checkpoint_callback,],
                            logger=comet_logger,
                            num_sanity_val_steps=2,
                            devices=ecm.train.device)
    
        trainer.fit(model,self.train_loader,self.valid_loader)
        trainer.test(model,dataloaders=self.test_loader)
        
    def run_one(self,config_file:str):
        self._execute_single_exp(config_file)

    def run(self):
        for config_file in self.experiments:
            if config_file == 'default.json':
                continue
            try:
                self.exp_logger.info(f"Start Experiment by {config_file}.")
                self._execute_single_exp(config_file)
                
            except Exception as e:
                self.exp_logger.error(f"Error occurs when execute experiment of  {config_file}.")
                self.exp_logger.error(e)
                return 


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dir',type=str,default=exp_setting_folder)
    args = argparser.parse_args()
    
    executor = Executor()
    executor.run_one('pv_weighted_loss.json')
