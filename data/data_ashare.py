# -*- encoding: utf-8 -*-
'''
@File    :   data_ashare.py
@Time    :   2024/11/06 12:17:24
@Author  :   Muyao ZHONG 
@Version :   1.0
@Contact :   zmy125616515@hotmail.com
@License :   (C)Copyright 2019-2020
@Title   :   This file is used to get the data of A-share stock market.
'''
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd    
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pprint import pprint

import torch
from torch.utils.data import DataLoader, random_split, Dataset
import lightning as L
import pickle

from utils.padding import collate_fn, padding_mask

# NewLOB modules
# from config import ExpConfigManager as ECM  
import collections
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading 

def get_ordered_columns(level=10, direct=("Bid","Ask"), pv=("Price", "Volume"),index=True,price=True,volume=True):
    """
    Get the ordered columns for the level of the order book.
    """

    columns = []

    if index:
        columns.append("index") 

    if price:
        for i in range(level,0,-1):
            columns.append(f"{direct[0]}{pv[0]}{i}") 
        for i in range(1,level+1):
            columns.append(f"{direct[1]}{pv[0]}{i}")

    if volume:
        for i in range(level,0,-1):
            columns.append(f"{direct[0]}{pv[1]}{i}")
        for i in range(1,level+1):
            columns.append(f"{direct[1]}{pv[1]}{i}")

    return columns

def get_all_rows(dates, frequecy=3):
    pass 

def get_data_filename(raw_path, name="unbalanced"):
    """
    Get the data file path.
    """
    resampled_path="dataset/real_data/resampled"
    normalized_path = "dataset/real_data/processed"
    balanced_path = "dataset/real_data/balanced"
    unblanced_path = "dataset/real_data/unbalanced"
    _ , basename = os.path.split(raw_path)
    resampled_file = basename.replace(".csv", "_resampled.csv")
    normalized_file = basename.replace(".csv", "_processed.csv")
    balanced_file = basename.replace(".csv", "_balanced.npz")
    unblanced_file = basename.replace(".csv", "_unbalanced.npz")
    if name == "resampled":
        return os.path.join(resampled_path, resampled_file)
    elif name == "normalized":
        return os.path.join(normalized_path, normalized_file)
    elif name == "balanced":
        return os.path.join(balanced_path, balanced_file)
    elif name == "unbalanced":
        return os.path.join(unblanced_path, unblanced_file)

class AShare(Dataset):
    def __init__(self,csv_path=None,
                 resampled_path="dataset/real_data/resampled",
                 normalized_path = "dataset/real_data/processed",
                 balanced_path = "dataset/real_data/balanced",
                 unblanced_path = "dataset/real_data/unbalanced",
                 max_len=100,
                 normalizing_method=['zscore'], 
                 balanced=True,
                 unblanced=True,
                 **kwargs):
        
        # self.balanced = balanced 
        self.csv_path = csv_path
        self.resampled_path = resampled_path  
        self.normalized_path = normalized_path
        self.balanced_path = balanced_path
        self.unbalanced_path = unblanced_path

        self.max_len = max_len
        self.normalizing_method = normalizing_method

        for key, value in kwargs.items():
            setattr(self, key, value)

        _,basename = os.path.split(csv_path)

        resampled_file = basename.replace(".csv", "_resampled.csv")
        normalized_file = basename.replace(".csv", "_processed.csv")
        balanced_file = basename.replace(".csv", "_balanced.npz")
        unblanced_file = basename.replace(".csv", "_unbalanced.npz")

        self.resampled_file = os.path.join(self.resampled_path,resampled_file)
        self.normalized_file = os.path.join(self.normalized_path,normalized_file)
        self.balanced_file = os.path.join(self.balanced_path,balanced_file)
        self.unbalanced_file = os.path.join(self.unbalanced_path,unblanced_file)

        if os.path.exists(self.resampled_path) == False:
            os.makedirs(self.resampled_path,exist_ok=True)

        if os.path.exists(self.normalized_path) == False:
            os.makedirs(self.normalized_path,exist_ok=True)

        if os.path.exists(self.balanced_path) == False:
            os.makedirs(self.balanced_path,exist_ok=True)

        if os.path.exists(self.unbalanced_path) == False:
            os.makedirs(self.unbalanced_path,exist_ok=True)

        self.raw_data = None 
        self.resampled_data = None
        self.normalized_data = None 
        self.balanced_data = None
        self.unbalanced_data = None 

        if os.path.exists(self.csv_path) == False:
            raise FileExistsError("file is not exist")
        else:
            self.raw_data = pd.read_csv(self.csv_path,index_col=0)

        if os.path.exists(self.resampled_file):
            print(f"Resampled data already exists at {self.resampled_file}")
            self.resampled_data = pd.read_csv(self.resampled_file)
            self.resampled_data.dropna(inplace=True)
        else:
            self.resample_data()

        if os.path.exists(self.normalized_file):
            print(f"Normalized data already exists at {self.normalized_file}")
            self.normalized_data = pd.read_csv(self.normalized_file)
        else:
            self.normalize_data()
        if unblanced:
            if os.path.exists(self.unbalanced_file):
                print(f"Unbalanced data already exists at {self.unbalanced_file}")
                self.unbalanced_data  = np.load(self.unbalanced_file)
                self.unbalanced_X = self.unbalanced_data['X']
                self.unbalanced_y = self.unbalanced_data['y']
            else:
                self.unbalance_data()

        if balanced:
            if os.path.exists(self.balanced_file):
                print(f"Balanced data already exists at {self.balanced_file}")
                self.balanced_data = np.load(self.balanced_file)
                self.balanced_X = self.balanced_data['X']
                self.balanced_y = self.balanced_data['y']
            else:
                self.balance_data()



    def resample_data(self):
        
        self.id = os.path.basename(self.csv_path).split("-")[0]

        self.raw_data.index.name = "index"

        # 如果需要将索引列保留为普通列，可以重置索引
        self.raw_data.reset_index(inplace=True)

        self.columns = get_ordered_columns()
        self.raw_data = self.raw_data[self.columns]
        self.raw_data['index'] = pd.to_datetime(self.raw_data['index'])
        #截取上午9:30到11:30以及下午1:00到3:00的数据
        
        self.raw_data = self.raw_data[(self.raw_data['index'].dt.time >= pd.to_datetime('09:30:00').time()) & (self.raw_data['index'].dt.time <= pd.to_datetime('11:30:00').time()) | (self.raw_data['index'].dt.time >= pd.to_datetime('13:00:00').time()) & (self.raw_data['index'].dt.time <= pd.to_datetime('15:00:00').time())]

        unique_dates = self.raw_data['index'].dt.date.unique()
        resampled_data = []

        for date in unique_dates:
            morning_start = pd.Timestamp(f"{date} 09:30:00")
            morning_end = pd.Timestamp(f"{date} 11:30:00")
            afternoon_start = pd.Timestamp(f"{date} 13:00:00")
            afternoon_end = pd.Timestamp(f"{date} 15:00:00")

            morning_index = pd.date_range(start=morning_start, end=morning_end, freq='3s')
            afternoon_index = pd.date_range(start=afternoon_start, end=afternoon_end, freq='3s')

            full_index = morning_index.append(afternoon_index)

            daily_data = self.raw_data[self.raw_data['index'].dt.date == date]
            daily_data = daily_data.set_index('index').reindex(full_index).ffill().reset_index()
            resampled_data.append(daily_data)

        self.resampled_data = pd.concat(resampled_data).reset_index(drop=True)
        # 删除所有包含nan的行

        self.resampled_data.dropna(inplace=True)
        self.resampled_data.to_csv(self.resampled_file, index=False)


    def get_data_by_start_end(self,start_date,end_date):
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        return self.raw_data[(self.raw_data['index'].dt.date >= start_date) & (self.raw_data['index'].dt.date <= end_date)]
    
    def normalize_data(self):

        price_columns = get_ordered_columns(index=False, price=True, volume=False)
        volume_columns = get_ordered_columns(index=False, price=False, volume=True)
        price_data = self.resampled_data[price_columns].values
        volume_data = self.resampled_data[volume_columns].values
        

        if "zscore" in self.normalizing_method:
            self.normalized_data = self.resampled_data.copy()
            self.save_path = self.normalized_file.replace(".csv","_zscore.csv")
            price_mean = np.nanmean(price_data)
            price_std = np.nanstd(price_data)
            volume_mean = np.nanmean(volume_data)
            volume_std = np.nanstd(volume_data)
            self.normalized_data[price_columns] = (self.resampled_data[price_columns] - price_mean) / price_std
            self.normalized_data[volume_columns] = (self.resampled_data[volume_columns] - volume_mean) / volume_std
            self.normalized_data.to_csv(self.save_path, index=False)

        if 'feature_zscore' in self.normalizing_method:
            self.normalized_data = self.resampled_data.copy()
            self.save_path = self.normalized_file.replace(".csv","_feature_zscore.csv")
            price_mean = np.nanmean(price_data,axis=0)
            price_std = np.nanstd(price_data,axis=0)
            volume_mean = np.nanmean(volume_data,axis=0)
            volume_std = np.nanstd(volume_data,axis=0)
            self.normalized_data[price_columns] = (self.resampled_data[price_columns] - price_mean) / price_std
            self.normalized_data[volume_columns] = (self.resampled_data[volume_columns] - volume_mean) / volume_std
            self.normalized_data.to_csv(self.save_path, index=False)


        if "minmax" in self.normalizing_method:
            # min-max normalization
            self.normalized_data = self.resampled_data.copy()
            self.save_path = self.normalized_file.replace(".csv","_minmax.csv")
            price_min = np.min(price_data)
            price_max = np.max(price_data)
            volume_min = np.min(volume_data)
            volume_max = np.max(volume_data)
            self.normalized_data[price_columns] = (self.resampled_data[price_columns] - price_min) / (price_max - price_min)
            self.normalized_data[volume_columns] = (self.resampled_data[volume_columns] - volume_min) / (volume_max - volume_min)
            self.normalized_data.to_csv(self.save_path, index=False)
        self.normalized_data.to_csv(self.normalized_file, index=False)
        


    

    def get_price(self):
        columns = get_ordered_columns(price=True, volume=False)
        return self.raw_data[columns]
    
    def get_volume(self):
        columns = get_ordered_columns(price=False, volume=True)
        return self.raw_data[columns]
    
    def midprice(self):
        midprice = (self.normalized_data['AskPrice1']+self.normalized_data['BidPrice1'])/2
        return midprice

    def get_trend(self,k=5,theta=0.002):
        midprice = self.midprice()
        trend_tmp = midprice.rolling(window=k,center=False).mean()
        #如果trend大于midprice*(1+theta)则为1，如果小于midprice*(1-theta)则为-1，否则为0
        trend = np.where(trend_tmp > midprice*(1+theta),1,np.where(trend_tmp < midprice*(1-theta),-1,0))
        #计算trend中0，1，-1的占比
        ups = np.sum(trend==1)/len(trend)
        downs = np.sum(trend==-1)/len(trend)
        steady = np.sum(trend==0)/len(trend)
        # print(f"ups:{ups},downs:{downs},steady:{steady}")
        return f"{downs:.04}|{steady:.04}|{ups:.04}"
    
    # def unbalance_data(self,k=5,theta=0.002):
    #     if self.unbalanced_data is not None:
    #         return
        
    #     print("Start Generate Imbalanced Data...")

    #     midprice = self.midprice()
    #     trend_tmp = midprice.rolling(window=k,center=False).mean()
    #     trend = np.where(trend_tmp > midprice*(1+theta),1,np.where(trend_tmp < midprice*(1-theta),-1,0))

    #     self.unbalanced_X = []
    #     self.unbalanced_y = []
    #     train_tmp = []

    #     for i in range(self.max_len, len(trend)-1, self.max_len//25):
    #         d = self.normalized_data.iloc[i-self.max_len:i,1:].values
    #         train_tmp.append([d,trend[i]+1])

    #     random.shuffle(train_tmp)
    #     self.unbalanced_X = np.array([x[0] for x in train_tmp],dtype=np.float16)
    #     self.unbalanced_y = np.array([y[1] for y in train_tmp],dtype=np.float16)

    #     print("Save Unbalanced Data...")
    #     np.savez(self.unbalanced_file,X=self.unbalanced_X,y=self.unbalanced_y)
    
    def unbalance_data(self,window=100,step=4,dtype=np.float16):
        """
        进行时序数据采样，支持指定窗口大小和步长。
        
        参数：
            data (ndarray): 归一化后的时序数据，形状为 (num_samples, num_features)
            window (int): 采样窗口大小
            step (int): 采样步长
            dtype (numpy dtype): 结果数据类型
        
        返回：
            unbalanced_X (ndarray): 形状为 (num_windows, window, num_features) 的采样数据
        """
        
        
        if self.unbalanced_data is not None:
            return
        
        print("Start Generate Unbalanced Data...")

        # midprice = self.midprice()
        # trend_tmp = midprice.rolling(window=k,center=False).mean()
        # trend = np.where(trend_tmp > midprice*(1+theta),1,np.where(trend_tmp < midprice*(1-theta),-1,0))

        print("Start step2.")
        data = self.normalized_data.iloc[:,1:].values
        self.unbalanced_X = sliding_window_view(data, (window,data.shape[1]))[::step,0].astype(dtype)
        
        self.unbalanced_y = np.array([0 for y in range(self.unbalanced_X.shape[0])],dtype=np.float16)

        print("Save Unbalanced Data...")
        np.savez(self.unbalanced_file,X=self.unbalanced_X,y=self.unbalanced_y)

    def balance_data(self, k=5, theta=0.002, window=100, step=4, dtype=np.float16):
        """
        计算 LOB 订单簿的中间价变化趋势，进行类别平衡采样，并存储为 npz 文件。
        
        参数：
            k (int): 计算移动平均的窗口大小
            theta (float): 趋势判定的阈值
            window (int): 采样窗口大小
            step (int): 记录趋势的步长
            dtype (numpy dtype): 结果数据类型

        返回：
            无返回值，数据保存在 self.balanced_X, self.balanced_y，并存入文件
        """

        if self.balanced_data is not None:
            return

        print("Start Balancing Data...")

        # 计算 midprice 并提取趋势信息
        midprice = self.midprice()
        trend_tmp = midprice.rolling(window=k, center=False).mean()
        trend = np.where(trend_tmp > midprice * (1 + theta), 1, 
                        np.where(trend_tmp < midprice * (1 - theta), -1, 0))

        # 记录趋势及对应索引
        trend_records = [(i, trend[i]) for i in range(self.max_len, len(trend) - 1, step)]

        # 分类存储索引
        categorized_indices = {1: [], -1: [], 0: []}
        for index, trend_value in trend_records:
            categorized_indices[trend_value].append(index)

        # 找到最少类别的样本数
        min_samples = min(map(len, categorized_indices.values()))

        print(f"Balancing Data: min_samples={min_samples}")

        # 进行下采样（最少的类别全选，其他类别随机采样相同数量）
        sampled_indices = []
        for key in categorized_indices:
            random.shuffle(categorized_indices[key])
            sampled_indices.extend(categorized_indices[key][:min_samples])

        # 乱序索引
        random.shuffle(sampled_indices)

        print("Extracting and preparing dataset...")

        # 提取数据并构造 LOB 数据集
        dataset_X = np.array([
            self.normalized_data.iloc[i - window:i, 1:].values for i in sampled_indices
        ], dtype=dtype)

        dataset_y = np.array([
            trend[i] for i in sampled_indices
        ], dtype=dtype)

        print(f"Dataset Shape: X={dataset_X.shape}, Y={dataset_y.shape}")

        # 存储数据
        print("Saving balanced dataset...")
        np.savez(self.balanced_file, X=dataset_X, y=dataset_y)
        
    

def get_all_info_single(folder,info_csv):
    df = pd.DataFrame(columns=['id','midprice-max|min|mean|std',
                               'k=1|theta=0.01','k=3|theta=0.01','k=5|theta=0.01','k=10|theta=0.01','k=1|theta=0.001','k=3|theta=0.001','k=5|theta=0.001','k=10|theta=0.001','k=1|theta=0.002','k=3|theta=0.002','k=5|theta=0.002','k=10|theta=0.002','k=1|theta=0.005','k=3|theta=0.005','k=5|theta=0.005','k=10|theta=0.005'])
    a = AShare()

    for folder,subfolders,files in os.walk(folder):
        for file in files:
            if 'sz' not in file:
                continue
            if 'resampled' in file:
                continue

            if file.endswith(".csv") and '-' in file:
                print("Start to process:",file)
                path = os.path.join(folder,file)
                a.data_preprocess(path)
                k1t001 = a.trend(k=1,theta=0.01)
                k3t001 = a.trend(k=3,theta=0.01)
                k5t001 = a.trend(k=5,theta=0.01)
                k10t001 = a.trend(k=10,theta=0.01)
                k1t0001 = a.trend(k=1,theta=0.001)
                k3t0001 = a.trend(k=3,theta=0.001)
                k5t0001 = a.trend(k=5,theta=0.001)
                k10t0001 = a.trend(k=10,theta=0.001)
                k1t0002 = a.trend(k=1,theta=0.002)
                k3t0002 = a.trend(k=3,theta=0.002)
                k5t0002 = a.trend(k=5,theta=0.002)
                k10t0002 = a.trend(k=10,theta=0.002)
                k1t0005 = a.trend(k=1,theta=0.005)
                k3t0005 = a.trend(k=3,theta=0.005)
                k5t0005 = a.trend(k=5,theta=0.005)
                k10t0005 = a.trend(k=10,theta=0.005)
                midprice = a.midprice()
                midprice_max = midprice.max()
                midprice_min = midprice.min()
                midprice_mean = midprice.mean()
                midprice_std = midprice.std()


                df.loc[len(df)] = [a.id,
                                   {midprice_max:.04}|{midprice_min:.04}|{midprice_mean:.04}|{midprice_std:.04},
                                   k1t001,k3t001,k5t001,k10t001,k1t0001,k3t0001,k5t0001,k10t0001,k1t0002,k3t0002,k5t0002,k10t0002,k1t0005,k3t0005,k5t0005,k10t0005]

    df.to_csv(info_csv,index=False)

                

class AShareData(Dataset):
    def __init__(
            self,
            dataset_path,
            dataset_name = "balanced", # balanced data path
            # balanced = True,
            **kwargs
    ):
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        # print(kwargs)
        self.dataset_path = dataset_path
        # print(self.dataset_path)
        self.balanced_file = get_data_filename(self.dataset_path, name=dataset_name)
        # self.a = AShare(csv_path=dataset_path)
        # self.balanced_file = dataset_path.replace(".csv","_balanced.npz")
        with np.load(self.balanced_file) as data:
            self.X = torch.from_numpy(data['X']).type(torch.FloatTensor)
            self.y = torch.from_numpy(data['y']).type(torch.FloatTensor) + 1

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        sample = self.X[index],self.y[index]
        return sample 
    
class AShareVaeData(Dataset):
    def __init__(
            self,
            dataset_path, # balanced data path
            balanced = True,
            **kwargs
    ):
        self.dataset_path = dataset_path
        self.a = AShare(csv_path=dataset_path)

        if balanced:
            # with np.load(self.a.balanced_file) as data:
            self.X = torch.from_numpy(self.a.balanced_X).type(torch.FloatTensor)
            #现在x的维度是[100,40]，把列[,:20],[,20:]拆到两个channel中成为[2,100,20]的形状
            self.X = torch.stack([self.X[:,:,:20],self.X[:,:,20:]],dim=1)
            self.y = torch.from_numpy(self.a.balanced_y).type(torch.FloatTensor) + 1
        else:
            # self.a.train_data()
            self.X = torch.from_numpy(self.a.unbalanced_X).type(torch.FloatTensor)
            self.X = torch.stack([self.X[:,:,:20],self.X[:,:,20:]],dim=1)
            self.y = torch.from_numpy(self.a.unbalanced_y).type(torch.FloatTensor) + 1


    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        sample = self.X[index],self.y[index]
        return sample 
    
class AShareVaeDataModule(L.LightningDataModule):

    def __init__(self, datasets, batch_size, is_shuffle_train=True, num_workers=4,seq_len=100,**kwargs):
        super().__init__()
        train_size = int(0.8*len(datasets))
        var_size = int(0.1*len(datasets))
        test_size = len(datasets)-train_size-var_size
        self.train_set, self.val_set, self.test_set = random_split(datasets, [train_size,var_size,test_size])

        self.batch_size = batch_size
        self.is_shuffle_train = is_shuffle_train
        self.num_workers = num_workers
        self.num_classes = 3
        self.pin_memory = True
        self.seq_len = seq_len

    def setup(self,stage=None):
        pass 

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.is_shuffle_train, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers)
    

class AShareDataModule(L.LightningDataModule):

    def __init__(self, datasets, batch_size, is_shuffle_train=True, num_workers=4,seq_len=100,**kwargs):
        super().__init__()
        train_size = int(0.8*len(datasets))
        var_size = int(0.1*len(datasets))
        test_size = len(datasets)-train_size-var_size
        self.train_set, self.val_set, self.test_set = random_split(datasets, [train_size,var_size,test_size])

        self.batch_size = batch_size
        self.is_shuffle_train = is_shuffle_train
        self.num_workers = num_workers
        self.num_classes = 3
        self.pin_memory = True
        self.seq_len = seq_len

    def setup(self,stage=None):
        pass 

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.is_shuffle_train, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers,collate_fn=lambda x: collate_fn(x, max_len=self.seq_len))

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers,collate_fn=lambda x: collate_fn(x, max_len=self.seq_len))

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers,collate_fn=lambda x: collate_fn(x, max_len=self.seq_len))




if __name__ == '__main__':
    # a = AShare("sz000999", 'dataset/real_data/sz000999-level10.csv')
    # # prices = a.get_price()
    # data = a.get_data_by_start_end("2019-10-8","2019-10-20")
    # print(data.head())
    # print(len(data))
    # print(prices.head())
    # volumes = a.get_volume()
    # print(volumes.head(1000))
    # print(len(a))
    # print(a.get_unique_dates())
    # get_all_info('dataset/real_data/ashare/000-001','dataset/real_data/info.csv')
    # infor = process_file('sz000065-level10.csv','dataset/real_data/ashare/000-001')
    # get_all_info('dataset/real_data/ashare','dataset/real_data/info.csv',worker_num=60)

    # a = AShare(csv_path='dataset/real_data/resampled/sz000001-level10.csv',
    #        resampled_path='dataset/real_data/resampled',
    #        processed_path='dataset/real_data/processed',
    #        balanced_path='dataset/real_data/balanced',)
    # sz001 = AShare(csv_path='dataset/real_data/raw_csv/sz000001-level10.csv',
    #                resampled_path='dataset/real_data/resampled',
    #                normalized_path='dataset/real_data/processed',
    #                balanced_path='dataset/real_data/balanced',
    #                unblanced_path='dataset/real_data/unbalanced',
    #                max_len=100,
    #                normalizing_method='zscore',)
    ashare_data = AShare(csv_path="dataset/real_data/sz300147-level10.csv",normalizing_method=['feature_zscore','zscore'],balanced=False,unblanced=True)
    # data_module = AShareDataModule(ashare_data.unbalance_data,batch_size=128)

    