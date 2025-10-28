# Insert the path into sys.path for importing.
import sys
import os
import comet_ml 
import torch 
import lightning as L 
from lightning.pytorch.loggers import CometLogger, TensorBoardLogger

torch.set_float32_matmul_precision('high')

# Insert the path into sys.path for importing.
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trainer.predict_trainer import PredictTrainer
from config import ExpConfigManager as ECM 
from model.prediction.mlp import MLP 
from model.prediction.cnn1 import CNN1 
from model.prediction.cnn2 import CNN2 
from model.prediction.transformer import Transformer
from data.data_lobcast import prepare_data_fi


exp_folder = "experiment/exp_settings/pred_FI_2010"
exp_file = "mlp_1.json"
cnn_exp_file = "cnn2_1.json"
transformer_exp_file = "transformer_1.json"
ecm = ECM(exp_folder)
ecm.load_config(transformer_exp_file)

# model = MLP()
model_a100 = CNN1()
model_a100_cnn2 = CNN2()
model_a100_trans = Transformer()
# comet_logger = CometLogger(
#                 api_key=ecm.comet.api_key,  # Optional
#                 workspace=ecm.comet.workspace,  # Optional
#                 save_dir=ecm.data.log_folder,  # Optional
#                 project_name=ecm.comet.project_name,  # Optional
#                 experiment_name=ecm.experiment.name+ecm.experiment.version)

tensorboard_logger = TensorBoardLogger(
                        save_dir="dataset/exp_data/prediction",
                        version=ecm.experiment.version,
                        name=ecm.experiment.name
                        )
    
trainer = L.Trainer(max_epochs=ecm.train.max_epochs,
                    logger = tensorboard_logger,
                    profiler="advanced",
                    num_sanity_val_steps=2,
                    devices=ecm.train.device)

config_file_path = os.path.join(exp_folder,exp_file)
# trainer.logger.experiment.log_asset(config_file_path,exp_file)
fi_dm = prepare_data_fi()

trainer.fit(model_a100_trans, fi_dm.train_dataloader(),fi_dm.val_dataloader())
