"""
Created on : 2024-07-10
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/SimLOB/simlob-refined/dataset/simu_data/data_processing.py
Description: Data Preprocessing
---
Updated    : 
---
Todo       : 
"""


# Insert the path into sys.path for importing.
from functools import partial
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional,List,Tuple,Dict,Union,Protocol
from utils.logger import setup_logger
from data.data_database import RawDataManager as RDM 
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L


########################################################
#######        Convert RAW CSV to pkl         ##########
########################################################

#======================  end   =========================
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


price_columns = ['bestBidPrice10', 'bestBidPrice9', 
                 'bestBidPrice8', 'bestBidPrice7', 
                 'bestBidPrice6', 'bestBidPrice5', 
                 'bestBidPrice4', 'bestBidPrice3', 
                 'bestBidPrice2', 'bestBidPrice1', 
                 'bestAskPrice1', 'bestAskPrice2', 
                 'bestAskPrice3', 'bestAskPrice4', 
                 'bestAskPrice5', 'bestAskPrice6', 
                 'bestAskPrice7', 'bestAskPrice8', 
                 'bestAskPrice9', 'bestAskPrice10']

volume_columns = ['bestBidVolume10', 'bestBidVolume9',
                  'bestBidVolume8', 'bestBidVolume7',
                  'bestBidVolume6', 'bestBidVolume5', 
                  'bestBidVolume4', 'bestBidVolume3', 
                  'bestBidVolume2', 'bestBidVolume1', 
                  'bestAskVolume1', 'bestAskVolume2', 
                  'bestAskVolume3', 'bestAskVolume4', 
                  'bestAskVolume5', 'bestAskVolume6', 
                  'bestAskVolume7', 'bestAskVolume8', 
                  'bestAskVolume9', 'bestAskVolume10']

bid_volume_columns = volume_columns[:10]

def mid_price(data:DataFrame):
    return (data['bestBidPrice1'] + data['bestAskPrice1']) / 2

def spread(data:DataFrame):
    return data['bestAskPrice1'] - data['bestBidPrice1']

def log_return(data:DataFrame):
    return np.log(data['midPrice']).diff()

# def trend(data:DataFrame, window:int=5, threshold:float=0.0001):
#     return (data['midPrice'].rolling(window).mean().shift(-window)-data['midPrice'] > threshold).astype(int)

def trend(data: DataFrame, window: int = 5, threshold: float = 0.0001):
    trend_values = data['midPrice'].rolling(window).mean().shift(-window) - data['midPrice']
    return trend_values.apply(lambda x: 2 if x > threshold else (0 if x < -threshold else 1)).astype(int)

def get_labels(data:DataFrame):
    
    mp = mid_price(data)
    labels = pd.DataFrame({'midPrice':mp})
    
    sp = spread(data)
    labels['spread'] = sp
    
    # lr = log_return(labels)
    # labels['logReturn'] = lr

    tr1 = trend(labels, window=1)
    tr3 = trend(labels, window=3)
    tr5 = trend(labels, window=5)
    tr7 = trend(labels, window=7)
    tr10 = trend(labels, window=10)
    
    trends = pd.DataFrame({'trend1':tr1, 'trend3':tr3, 'trend5':tr5, 
                         'trend7':tr7, 'trend10':tr10})
    labels = labels.join(trends)
    final_labels = ['midPrice', 'spread','trend1', 'trend3', 'trend5', 'trend7', 'trend10']
    return labels[final_labels]
    
def normalize(df:DataFrame, method="zscore"):
    if method == "zscore":
        prices_mean = df[price_columns].values.mean()
        prices_std = df[price_columns].values.std()
        volumes_mean = df[volume_columns].values.mean()
        volumes_std = df[volume_columns].values.std()

        # Standardize the price and volume columns
        df[price_columns] = (df[price_columns] - prices_mean) / prices_std
        df[volume_columns] = (df[volume_columns] - volumes_mean) / volumes_std
        return df[price_columns+volume_columns]
            
    elif method == "minmax":
        prices_max = df[price_columns].values.max()
        prices_min = df[price_columns].values.min()
        volumes_max = df[volume_columns].values.max()
        volumes_min = df[volume_columns].values.min()

        df[price_columns]=(df[price_columns]-prices_min)/(prices_max-prices_min)
        df[volume_columns] = (df[volume_columns]-volumes_min)/(volumes_max-volumes_min)
        return df[price_columns+volume_columns]
    
    else:
        raise ValueError("Invalid normalization method. Choose 'zscore' or 'minmax'.")
    
def column_normalize(df:DataFrame, method="zscore"):
    if method == "zscore":
        prices_mean = df[price_columns].mean()
        price_std = df[price_columns].std()
        volumes_mean = df[volume_columns].mean()
        volumes_std = df[volume_columns].std()
        
        df[price_columns] = (df[price_columns] - prices_mean) / price_std
        df[volume_columns] = (df[volume_columns] - volumes_mean) / volumes_std
        
        return df[price_columns+volume_columns]
    
    elif method == "minmax":
        prices_max = df[price_columns].max()
        prices_min = df[price_columns].min()
        volumes_max = df[volume_columns].max()
        volumes_min = df[volume_columns].min()
        
        df[price_columns] = (df[price_columns] - prices_min) / (prices_max - prices_min)
        df[volume_columns] = (df[volume_columns] - volumes_min) / (volumes_max - volumes_min)
        
        return df[price_columns+volume_columns]
        
class SimDataset(Dataset):
    def __init__(self, 
                 dataset_path:str, 
                 start_time = 2000, 
                 observe_time = 100, 
                 max_samples:int=1800, 
                 normalize_method = "zscore",
                 column_wise=False, 
                 debug = False, 
                 **kargs ):
        self.dataset_path = dataset_path
        self.start_time = start_time
        self.observe_time = observe_time
        if column_wise:
            self.normalize_method = partial(column_normalize, method=normalize_method)
        else:
            self.normalize_method = partial(normalize, method=normalize_method)
        
        
        if os.path.exists(dataset_path):
            self.X, self.y = torch.load(dataset_path)
        else:
            dpath = os.path.dirname(dataset_path)
            os.makedirs(dpath, exist_ok=True)
            self.generate_dataset(max_samples)
            
    def __len__(self):
        return len(self.X)-self.start_time-self.observe_time-20
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
            
    def generate_dataset(self, max_samples:int=1800):
        self.rdm = RDM()
        self.X = []
        self.y = []
        
        train_records = self.rdm.get_train_data(max_samples, seed = 1234)
        
        with tqdm(total=len(train_records), desc="Proceeding (0/{})".format(len(train_records))) as pbar:
            for i,record in enumerate(train_records,1):
                pbar.set_description("Proceeding ({}/{})".format(i, len(train_records)))
                file_path = record["file_path"]
                df = pd.read_csv(file_path)
                
                df = self.normalize_method(df)
              
                labels = get_labels(df)
                
                X = torch.tensor(df.values, dtype=torch.float32)
                y = torch.tensor(labels.values, dtype=torch.float32)
                
                Xs, ys = self.prepare_data(X,y)
                
                self.X += Xs 
                self.y += ys 
                
                pbar.update(1)
                
        torch.save((self.X, self.y), self.dataset_path)
        
    def prepare_data(self,X,y):
        Xs = []
        ys = []
        
        slide_window = self.observe_time
        data_points = (len(X)-self.start_time-20) // slide_window
        
        for i in range(data_points):
            Xs.append(X[i*slide_window:i*slide_window+self.observe_time])
            ys.append(y[i*slide_window+self.observe_time-1])
            
        return Xs,ys
    
class SimDataloader(L.LightningDataModule):
    def __init__(self, datasets, batch_size,*args, is_shuffle_train=True, num_workers=20,**kargs):
        super().__init__()

        self.train_set, self.val_set, self.test_set = random_split(datasets, [int(0.8*len(datasets)), int(0.1*len(datasets)), int(0.1*len(datasets))])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_shuffle_train = is_shuffle_train

        # self.x_shape = self.test_set.x_shape
        # self.num_classes = 
        self.pin_memory = True

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.is_shuffle_train, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers)
          

def convert_raw_to_csv_np(input_folder, 
                      csv_output_folder=None,
                      np_output_folder=None, 
                      ts_output_folder=None,
                    #   log_filename=None,
                      minmax_scale=False,
                      save_csv=False,
                      save_np=False,
                      save_ts=False,
                      final_data_filename=None):
    
    # os.makedirs(input_folder, exist_ok=True)
    if not os.path.exists(input_folder):
        print("Input Folder is not exist!")
        return 
    
    
    if csv_output_folder:
        os.makedirs(csv_output_folder, exist_ok=True)
    if np_output_folder:
        os.makedirs(np_output_folder, exist_ok=True)
    if ts_output_folder:
        os.makedirs(ts_output_folder, exist_ok=True)
    
    
    # with open(log_filename,'w') as log:
    #     log.write("FileName,LineNumber\n")
        
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    final_result = []
    
    def process_file(file, normalize='zscore'):
        file_path = os.path.join(input_folder,file)
        
        # bid_volume_columns = volume_columns[:10]
        try:
            df = pd.read_csv(file_path)
            df = df.iloc[1000:,:]

            df[bid_volume_columns]=-df[bid_volume_columns]
            # Calculate the overall mean and standard deviation for prices and volumes
            if normalize == 'zscore':
                prices_mean = df[price_columns].values.mean()
                prices_std = df[price_columns].values.std()
                volumes_mean = df[volume_columns].values.mean()
                volumes_std = df[volume_columns].values.std()

                # Standardize the price and volume columns
                df[price_columns] = (df[price_columns] - prices_mean) / prices_std
                df[volume_columns] = (df[volume_columns] - volumes_mean) / volumes_std
            
            if normalize == 'minmax':
                prices_max = df[price_columns].values.max()
                prices_min = df[price_columns].values.min()
                volumes_max = df[volume_columns].values.max()
                volumes_min = df[volume_columns].values.min()

                df[price_columns]=(df[price_columns]-prices_min)/(prices_max-prices_min)
                df[volume_columns] = (df[volume_columns]-volumes_min)/(volumes_max-volumes_min)

            # Save the processed file as CSV
            output_path = os.path.join(csv_output_folder, file)
            df=df[price_columns + volume_columns]
            if save_csv:
                df.to_csv(output_path, index=False)

            # Save the processed data as NumPy array
            numpy_array = df.values
            if save_np:
                numpy_output_path = os.path.join(np_output_folder, file.replace('.csv', '.npy'))
                np.save(numpy_output_path, numpy_array)
                
            if save_ts:
                tensor_out_path = os.path.join(ts_output_folder,file.replace('.csv','.pt'))
                tensor_data = torch.tensor(numpy_array)
                torch.save(tensor_data,tensor_out_path)
            
            if final_data_filename:
                final_result.append(numpy_array)
                print("one added")
                
            if not final_data_filename:
                return numpy_array
                
        except Exception as e:
            print(f"Error in processing file: {file} \nError: {e}") 
                
                    
                    
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = []
        with tqdm(total=len(csv_files)) as pbar:
            futures = {executor.submit(process_file, file): file for file in csv_files}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
                
        # = list(executor.map(process_file,csv_files))
        
    # resluts = [res for res in resluts if res is not None ]
                
    if final_data_filename:
        final_result_tensor = torch.tensor(np.concatenate(final_result, axis=0))
        torch.save(final_result_tensor, final_data_filename)


def data_preprocess_from_csv(file_path, minmax_scale=False):
    # file_path = os.path.join(input_folder,file)
    
    # bid_volume_columns = volume_columns[:10]

    df = pd.read_csv(file_path )
    df = df.iloc[1000:,:]

    df[bid_volume_columns]=-df[bid_volume_columns]
    # Calculate the overall mean and standard deviation for prices and volumes
    prices_mean = df[price_columns].values.mean()
    prices_std = df[price_columns].values.std()
    volumes_mean = df[volume_columns].values.mean()
    volumes_std = df[volume_columns].values.std()

    # Standardize the price and volume columns
    df[price_columns] = (df[price_columns] - prices_mean) / prices_std
    df[volume_columns] = (df[volume_columns] - volumes_mean) / volumes_std
    
    if minmax_scale:
        prices_max = df[price_columns].values.max()
        prices_min = df[price_columns].values.min()
        volumes_max = df[volume_columns].values.max()
        volumes_min = df[volume_columns].values.min()

        # Normalize the standardized price and volume columns
        # scaler = MinMaxScaler()
        # df[price_columns] = scaler.fit_transform(df[price_columns])
        # df[volume_columns] = scaler.fit_transform(df[volume_columns])
        df[price_columns]=(df[price_columns]-prices_min)/(prices_max-prices_min)
        df[volume_columns] = (df[volume_columns]-volumes_min)/(volumes_max-volumes_min)

    numpy_array = df[price_columns+volume_columns].values
    
    return numpy_array
  
def data_preprocess(raw_data:DataFrame, minmax_scale=False ):
    # drop the first 1000 rows.
    raw_data = raw_data.iloc[1000:,:]
    raw_data[bid_volume_columns]=-raw_data[bid_volume_columns]
    # Calculate the overall mean and standard deviation for prices and volumes
    prices_mean = raw_data[price_columns].values.mean()
    prices_std = raw_data[price_columns].values.std()
    volumes_mean = raw_data[volume_columns].values.mean()
    volumes_std = raw_data[volume_columns].values.std()

    # Standardize the price and volume columns
    raw_data[price_columns] = (raw_data[price_columns] - prices_mean) / prices_std
    raw_data[volume_columns] = (raw_data[volume_columns] - volumes_mean) / volumes_std
    
    if minmax_scale:
        prices_max = raw_data[price_columns].values.max()
        prices_min = raw_data[price_columns].values.min()
        volumes_max = raw_data[volume_columns].values.max()
        volumes_min = raw_data[volume_columns].values.min()
        
        raw_data[price_columns]=(raw_data[price_columns]-prices_min)/(prices_max-prices_min)
        raw_data[volume_columns] = (raw_data[volume_columns]-volumes_min)/(volumes_max-volumes_min)
        
    return raw_data[price_columns+volume_columns]

if __name__ == "__main__":
    # convert_raw_to_csv_np(input_folder='./dataset/simu_data/log_generate_raw',
    #                   csv_output_folder='./dataset/simu_data/log_generate_csv',
    #                   np_output_folder='./dataset/simu_data/log_generate_np_no_scale',
    #                   log_filename='./dataset/simu_data/log_generate_log.txt',
    #                   minmax_scale=False,
    #                   save_csv=False,
    #                   save_np=True,
    #                   save_ts=False,
    #                   final_data_filename=None)
    
    column_zscore_data = "dataset/train_data/zscore_trend_column.pt"
    global_zscore_data = "dataset/train_data/zscore_trend_global.pt"
    column_minmax_data = "dataset/train_data/minmax_trend_column.pt"
    global_minmax_data = "dataset/train_data/minmax_trend_global.pt"
    
    # sd1 = SimDataset(dataset_path=column_minmax_data, normalize_method='minmax', column_wise=True)
    # sd2 = SimDataset(dataset_path=column_zscore_data, normalize_method='zscore', column_wise=True)
    
    # sdm = SimDataloader(sd, batch_size=256)
    
    # ts = sdm.train_dataloader()
    # for i in ts:
    #     print(i[0].shape)
    #     print(i[1].shape)
    #     break
    
    SimDataset(dataset_path=global_minmax_data, normalize_method='minmax', column_wise=False)
    
    SimDataset(dataset_path=global_zscore_data, normalize_method='zscore', column_wise=False)
    # print(len(sd))
    # print(sd[0])
    # print((sd[0][0][99][10]+sd[0][0][99][9])/2)
    # print(sd[0][0].shape)
    # print(sd[0][1][-1])
    # print(sd[0][1])