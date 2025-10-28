# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import ExpConfigManager as ECM 

import comet_ml 
import lightning as L 

from lightning.pytorch.loggers import CometLogger

from data.data_lobcast import prepare_data_fi
from model.prediction.mlp import MLP

class PredictTrainer:
    def __init__(self,config_path="experiment/exp_settings/pred_FI_2010"):
        self.ecm = ECM(config_path)
        self.config_path = config_path

    def prepare(self,exp_file):
        self.ecm.load_config(exp_file)
  
        self.model = MLP()
        

        self.comet_logger = CometLogger(
                        api_key=self.ecm.comet.api_key,  # Optional
                        workspace=self.ecm.comet.workspace,  # Optional
                        save_dir=self.ecm.data.log_folder,  # Optional
                        project_name=self.ecm.comet.project_name,  # Optional
                        experiment_name=self.ecm.experiment.name+self.ecm.experiment.version)
    
        self.trainer = L.Trainer(max_epochs=self.ecm.train.max_epochs,
                            profiler="advanced",
                            logger=self.comet_logger,
                            num_sanity_val_steps=2,
                            devices=self.ecm.train.device)
        config_file_path = os.path.join(self.config_path,exp_file)
        self.trainer.logger.experiment.log_asset(config_file_path,exp_file)
        self.fi_dm = prepare_data_fi()
        
    def run(self):
        
        self.trainer.fit(self.model,self.fi_dm.train_dataloader(),self.fi_dm.val_dataloader())
        
        self.trainer.test(self.model,dataloaders=self.fi_dm.test_dataloader())