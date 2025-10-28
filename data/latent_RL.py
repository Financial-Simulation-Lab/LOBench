import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from model import CNN2, LSTM, Transformer_AE, iTransformer, TimesNet, TimeMixer, DeepLOB, TransLOB, SimLOB
from data.data_ashare import AShareData,AShareDataModule
import numpy as np
from tqdm import tqdm 
import torch

def model_loader(model_name,task_name = "reconstruction",ckpt_path="",d_model=128,unified_d=256,dropout=0.1,seq_len=100,decoder_name="transformer_decoder"):
    if model_name == "cnn2":
        model_config= {
            "task_name": task_name,
            "d_model": d_model,
            "num_class": 3,
            "unified_d": unified_d,
            "dropout": dropout,
            "checkpoint_path": ckpt_path,
            "decoder_name":decoder_name,
            "seq_len": seq_len,
            "enc_in":40
        }
        model = CNN2.load_from_checkpoint(**model_config)


    elif model_name == "lstm":
        model_config = {
            "task_name": task_name,
            "d_model": d_model,
            "unified_d": unified_d,
            "dropout": dropout,
            "enc_in": 40,
            "seq_len": seq_len,
            "decoder_name":decoder_name,
            "checkpoint_path": ckpt_path
        }
        model = LSTM.load_from_checkpoint(**model_config)

    
    elif model_name == "transformer":
        model_config = {
            "task_name": task_name,
            "d_model": d_model,
            "unified_d": unified_d,
            "dropout": dropout,
            "enc_in": 40,
            "nhead": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 2048,
            "seq_len": seq_len,
            "decoder_name":decoder_name,
            "checkpoint_path": ckpt_path
            }
        model = Transformer_AE.load_from_checkpoint(**model_config)

    elif model_name == "itransformer":
        model_config = {
            "task_name": task_name,
            "d_model": d_model,
            "unified_d": unified_d,
            "dropout": dropout,
            "enc_in": 40,
            "seq_len": seq_len,
            "decoder_name":decoder_name,
            "checkpoint_path": ckpt_path,
            #####
            "e_layers": 3,
            "d_ff": 256,
            "freq": "s",
            "output_attention": False,
            "embed": "timeF",
            "factor": 1,
            "n_heads": 8,
            "activation": "gelu",
        }
        model = iTransformer.load_from_checkpoint(**model_config)
    elif model_name == "timesnet":
        model_config = {
            "task_name": task_name,
            "unified_d": unified_d,
            "dropout": dropout,
            "enc_in": 40,
            "seq_len": seq_len,
            "decoder_name":decoder_name,
            "checkpoint_path": ckpt_path,
            #####
            "pred_len": 0,
            "e_layers": 3,
            "d_model": 32,
            "d_ff": 64,
            "embed": "timeF",
            "freq": "s",
            "num_kernels": 6,
            "top_k": 3,
        }
        model = TimesNet.load_from_checkpoint(**model_config)
    elif model_name == "timemixer":
        model_config = {
            "task_name": task_name,
            "unified_d": unified_d,
            "dropout": dropout,
            "enc_in": 40,
            "seq_len": seq_len,
            "decoder_name":decoder_name,
            "checkpoint_path": ckpt_path,
            #########
            "pred_len": 0,
            "e_layers": 2,
            "d_model": 16,
            "d_ff": 32,
            "down_sampling_window": 2,
            "down_sampling_layers": 3,
            "down_sampling_method": "avg",
            "channel_independence": 0,
            "moving_avg": 25,
            "decomp_method": "moving_avg",
            "use_norm": 1,
            "c_out": 40,
            "embed": "timeF",
            "freq": "s",
            "num_kernels": 6,
            "top_k": 3
        }
        model = TimeMixer.load_from_checkpoint(**model_config)
    elif model_name == "deeplob":   
        model_config = {
            "task_name": task_name,
            "d_model": 256,
            "unified_d": unified_d,
            "enc_in": 40,
            "seq_len": seq_len,
            "decoder_name":decoder_name,
            "checkpoint_path": ckpt_path
        }
        model = DeepLOB.load_from_checkpoint(**model_config)
    elif model_name == "translob":
        model_config = {
            "task_name": task_name,
            "d_model": 256,
            "unified_d": unified_d,
            "enc_in": 40,
            "seq_len": seq_len,
            "decoder_name":decoder_name,
            "checkpoint_path": ckpt_path
        }
        model = TransLOB.load_from_checkpoint(**model_config)
    elif model_name == "simlob":
        model_config = {
                "task_name": task_name,
                "d_model": d_model,
                "unified_d": unified_d,
                "dropout": dropout,
                "trans_embed_size": 256,
                "multi_head_num": 2,
                "trans_encoder_layer": 6,
                "feed_forward": True,
                "checkpoint_path": ckpt_path,
                "decoder_name":decoder_name,
                "seq_len": seq_len,
                "enc_in":40
        }
        model = SimLOB.load_from_checkpoint(**model_config)
    else:
        print("Model not found")
        return None
        
    
    return model  

def load_data(dataset_path="dataset/real_data/raw_data/sz000001-level10.csv",dataset_name="balanced",index=0):
    ashare_data = AShareData(dataset_path=dataset_path, dataset_name=dataset_name)
    print(f"Dataset size: {len(ashare_data)}")
    ashare_data_module = AShareDataModule(datasets=ashare_data,batch_size=128)
    print(f"Dataset size: {len(ashare_data_module.train_dataloader().dataset)}")
    return ashare_data_module.train_dataloader() 



if __name__ == "__main__":
    d_dataset = load_data("dataset/real_data/sz000001-level10.csv")
    # models_name = [
    # "cnn2","lstm","transformer","timemixer","itransformer","timesnet","deeplob","translob","simlob"
    # ]
    models_name = ["cnn2"]
    reconstruction_paths_template = "report/results/experiments/models/[A5000]reconstruction_{}_all_ashare_sz000001:v1.0.0.ckpt"
    hiddens = []
    y_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in models_name:
        model_loaded = model_loader(name,ckpt_path=reconstruction_paths_template.format(name))
        model_loaded.eval()
        for batch in tqdm(d_dataset, desc="Processing batches", unit="batch"):
            X, y = batch
            X, y = X.to(device), y.to(device) 
            with torch.no_grad():
                hidden = model_loaded.encode(X)
                hiddens.append(hidden.cpu().numpy()) 
                y_labels.append((y-1).cpu().numpy())
               
        hiddens = np.concatenate(hiddens, axis=0)
        y_labels = np.concatenate(y_labels, axis=0)


        print(f"Dataset Shape: X={hiddens.shape}, Y={y_labels.shape}")
        
        np.savez(f"dataset/real_data/balanced/{name}_sz000001-level10_balanced.npz", X=hiddens, y=y_labels)

        print(f"{name} latent representation saved as dataset.npz")