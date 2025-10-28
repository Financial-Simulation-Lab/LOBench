import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from data import AShareData, AShareDataModule
from model import CNN2, LSTM, Transformer_AE, iTransformer, TimesNet, TimeMixer, DeepLOB, TransLOB, SimLOB
import torch
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

def load_test_data(dataset_path="dataset/real_data/raw_csv/sz000001-level10.csv",dataset_name="balanced"):
    ashare_data = AShareData(dataset_path=dataset_path, dataset_name=dataset_name)
    ashare_data_module = AShareDataModule(datasets=ashare_data,batch_size=256)
    return ashare_data_module.test_dataloader()

def model_loader(model_name,task_name = "classification",ckpt_path="",d_model=128,num_class=3,unified_d=256,dropout=0.1):
    if model_name == "cnn2":
        model_config= {
            "task_name": task_name,
            "d_model": d_model,
            "num_class": num_class,
            "unified_d": unified_d,
            "dropout": dropout,
            "checkpoint_path": ckpt_path,
        }
        model = CNN2.load_from_checkpoint(**model_config)
    elif model_name == "lstm":
        model_config = {
            "task_name": task_name,
            "d_model": d_model,
            "num_class": num_class,
            "unified_d": unified_d,
            "dropout": dropout,
            "checkpoint_path": ckpt_path
        }
        model = LSTM.load_from_checkpoint(**model_config)    
    elif model_name == "transformer":
        model_config = {
            "task_name": task_name,
            "d_model": d_model,
            "unified_d": unified_d,
            "dropout": dropout,
            "num_class": num_class,
            "enc_in": 40,
            "nhead": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 2048,
            "seq_len": 100,
            "checkpoint_path": ckpt_path
            }
        model = Transformer_AE.load_from_checkpoint(**model_config)
    elif model_name == "itransformer":
        model_config = {
            "task_name": task_name,
            "d_model": d_model,
            "e_layers": 3,
            "enc_in": 40,
            "d_ff": 256,
            "unified_d": unified_d,
            "dropout": dropout,
            "freq": "s",
            "output_attention": False,
            "embed": "timeF",
            "factor": 1,
            "n_heads": 8,
            "seq_len": 100,
            "activation": "gelu",
            "num_class": num_class,
            "checkpoint_path": ckpt_path,
            #####
        }
        model = iTransformer.load_from_checkpoint(**model_config)
    elif model_name == "timesnet":
        model_config = {
            "task_name": task_name,
            "unified_d": unified_d,
            "dropout": dropout,
            "num_class": num_class,
            "enc_in": 40,
            "seq_len": 100,
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
            "num_class": 3,
            "enc_in": 40,
            "seq_len": 100,
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
            "num_class": num_class,
            "dropout": dropout,
            "unified_d": unified_d,
            "checkpoint_path": ckpt_path
        }
        model = DeepLOB.load_from_checkpoint(**model_config)
    elif model_name == "translob":
        model_config = {
            "task_name": task_name,
            "d_model": 256,
            "unified_d": unified_d,
            "seq_len": 100,
            "num_class": num_class,
            "dropout": dropout,
            "checkpoint_path": ckpt_path
        }
        model = TransLOB.load_from_checkpoint(**model_config)
    elif model_name == "simlob":
        model_config = {
                "task_name": task_name,
                "unified_d": unified_d,
                "dropout": dropout,
                "trans_embed_size": 256,
                "multi_head_num": 2,
                "trans_encoder_layer": 6,
                "feed_forward": True,
                "num_class": num_class,
                "checkpoint_path": ckpt_path
        }
        model = SimLOB.load_from_checkpoint(**model_config)
    else:
        print("Model not found")
        return None
            
    return model 

if __name__ == "__main__":
    m_data = 'sz000001'
    dataset_template = "dataset/real_data/{}-level10.csv"
    d_dataset = load_test_data(dataset_path=dataset_template.format(m_data))
    print("Load test dataset successfully.")
    models_name = [
        "cnn2","lstm","transformer","timemixer","itransformer","timesnet","deeplob","translob","simlob"
    ]
    reconstruction_paths_template = "report/results/experiments/models/[A5000]prediction_{}_ashare_{}:v1.0.0.ckpt"

    for name in models_name:
        model_loaded = model_loader(name,ckpt_path=reconstruction_paths_template.format(name,m_data))
        model_loaded.eval()
        print("Load model successfully.")

        all_preds = []
        all_labels = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch in tqdm(d_dataset, desc="Processing batches", unit="batch"):
            X, y, _ = batch
            X, y = X.to(device), y.to(device) 
            with torch.no_grad():
                outputs = model_loaded(X, X)
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())  
            all_labels.extend(y.cpu().numpy().tolist())  

        # 计算 Precision 和 Recall
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        print(f"Model: {name}, Precision: {precision:.4f}, Recall: {recall:.4f}")
