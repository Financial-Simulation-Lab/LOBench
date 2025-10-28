"""
Created on : 2024-08-03
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/metric/loss_func.py
Description: Loss Functions for LOBs
---
Updated    : 
---
Todo       : 
"""

# Insert the path into sys.path for importing.
from math import exp, sqrt
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional,List,Tuple,Dict,Union,Protocol
import torch.nn.functional as F 
import numpy as np 
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def NewLoss(x1,x2, alpha=0.2,fi2010=False):
    """计算预处理后的两个连续LOB的损失函数.

    Args:
        x1 (_type_): _description_
        x2 (_type_): _description_
    """

    result = alpha * PriceLoss(x1,x2) + (1-alpha) * MoneyLoss(x1,x2)
    return result

def PriceLoss(src, tgt):
    """src和tgt的价格损失函数.

    Args:
        src (torch.tensor): 输入得LOBs
        tgt (torch.tensor): 输出得LOBs

    Returns:
        _type_: _description_
    """
    h=src.shape[2]//2
    price1 = src[:, :, :h]
    if tgt.shape[2] == h:
        price2 = tgt
    else:
        price2 = tgt[:, :, :h]
    
    # 计算每个样本的均方误差
    mse_loss = F.mse_loss(price1, price2)
    # mse_loss = mse_loss.mean(dim=(1, 2))  # 对每个样本求均值
    
    return mse_loss

def VolumeLoss(src,tgt):
    """src和tgt的成交量损失函数.

    Args:
        src (torch.tensor): 输入得LOBs
        tgt (torch.tensor): 输出得LOBs

    Returns:
        _type_: _description_
    """
    h=src.shape[2]//2
    volume1 = src[:, :, h:]
    if tgt.shape[2] == h:
        volume2 = tgt
    else:
        volume2 = tgt[:, :, h:]
    
    # 计算每个样本的均方误差
    mse_loss = F.mse_loss(volume1, volume2)
    
    return mse_loss

def Weights(r,h):
    n = h//2
    weights1 = torch.tensor([r**i for i in range(n)], dtype=torch.float32,device=device)
    weights1 = weights1 / weights1.sum() * n / sqrt(2) # 归一化，使权重和为N
    weights2 = torch.tensor([r**i for i in range(n-1,-1,-1)], dtype=torch.float32,device=device)
    
    weights2 = weights2 / weights2.sum() *  n / sqrt(2)# 归一化，使权重和为N
    
    weights = torch.cat([weights1, weights2])
    
    return weights


def WeightedPriceLoss(src, tgt,factor=1.5):
    h=src.shape[2]//2
    
    price1 = src[:, :, :h]
    if tgt.shape[2]==h:
        price2 = tgt 
    else:
        price2 = tgt[:, :, :h]
    
    
    # 计算每个样本的均方误差
    
    weights = Weights(factor, h).view(1,1,-1).expand_as(price1)
    
    mse_loss = F.mse_loss(price1 * weights, price2 * weights)
    # mse_loss = mse_loss.mean(dim=(1, 2))  # 对每个样本求均值
    
    return mse_loss

def WeightedVolumeLoss(src,tgt,factor=1.5):
    h=src.shape[2]//2
    
    volume1 = src[:, :, h:]
    if tgt.shape[2]==h:
        volume2 = tgt 
    else:
        volume2 = tgt[:, :, h:]
    
    # 计算每个样本的均方误差
    
    weights = Weights(factor, h).view(1,1,-1).expand_as(volume1)
    
    mse_loss = F.mse_loss(volume1 * weights, volume2 * weights)
    # mse_loss = mse_loss.mean(dim=(1, 2))  # 对每个样本求均值
    
    return mse_loss

def MoneyLoss(L1,L2,acc=1e-5):
    """计算两个LOB的损失函数.

    Args:
        L1 (torch.Tensor): 第一个LOB
        L2 (torch.Tensor): 第二个LOB
    """
    h = L1.shape[2]//2
    price1 = L1[:,:, :h]
    price2 = L2[:,:, :h]
    
    volume1 = L1[:,:, h:]
    volume2 = L2[:,:, h:]
    
    money1 = price1 * volume1
    money2 = price2 * volume2
    
    money_matrix1 = money1.unsqueeze(2).expand(-1,-1,h,-1)
    money_matrix2 = money2.unsqueeze(3).expand(-1,-1,-1,h)
    
    money_matrix = money_matrix1 - money_matrix2
    zero_money = abs(money_matrix) < acc
    # tmp = money_matrix > 1e-5
    # print(tmp.sum())

    matrix1 = price1.unsqueeze(2).expand(-1,-1,h,-1)
    # matrix1 = matrix1.repeat(1,1,1,20)
    
    
    matrix2 = price2.unsqueeze(3).expand(-1, -1, -1, h)

    bool_matrix = abs(matrix1 - matrix2) < acc
    
    
    # print(bool_matrix)
    # print(bool_matrix.shape)
    
    row_bool = bool_matrix.any(dim=3)
    col_bool = bool_matrix.any(dim=2)

    same_money = money_matrix[torch.logical_and(bool_matrix,zero_money)] 

    
    l1 = squre_sum(money1[~col_bool])
    l2 = squre_sum(money2[~row_bool])
    l3 = squre_sum(same_money)

    return torch.sqrt((l1+l2+l3)/(money1[~col_bool].numel()+money2[~row_bool].numel()+same_money.numel()))

def MAELoss(src,tgt):
    return F.l1_loss(src,tgt)

def MSELoss(src,tgt):
    return F.mse_loss(src,tgt)

def regulerization(price):
    price_diff = price[:,:,:-1] - price[:,:,1:] 
    price_disorder_loss = F.relu(price_diff).mean()
    return price_disorder_loss

def RegLoss(src,tgt,splited=False):
    h = src.shape[2]//2
    if  splited:
        price,volume = tgt 
    else:
        price = tgt[:,:,:h]
        volume = tgt[:,:,h:]
    
    loss = regulerization(price)
    return loss
    

def WeightedMSELoss(src,tgt,factor=1.5,splited=False,alpha = 0.8):
    h = src.shape[2]//2
    if  splited:
        price,volume = tgt 
    else:
        price = tgt[:,:,:h]
        volume = tgt[:,:,h:]
        
    volume_loss = WeightedVolumeLoss(src,volume,factor=factor)
    price_loss = WeightedPriceLoss(src,price,factor=factor)
    
    return alpha * price_loss + (1-alpha) * volume_loss

def WeightedMSELoss_with_reg(src,tgt,factor=1.5,splited=False,alpha = 0.8,reg_factor=100):
    h = src.shape[2]//2
    if  splited:
        price,volume = tgt 
    else:
        price = tgt[:,:,:h]
        volume = tgt[:,:,h:]
        
    volume_loss = WeightedVolumeLoss(src,volume,factor=factor)
    price_loss = WeightedPriceLoss(src,price,factor=factor)
    reg_loss = regulerization(price)
    
    return alpha * price_loss + (1-alpha) * volume_loss + reg_loss*reg_factor

def AllLoss(src,tgt,factor=1.5,splited=False,alpha = 0.8,reg_factor=10):
    h = src.shape[2]//2
    if  splited:
        price,volume = tgt 
    else:
        price = tgt[:,:,:h]
        volume = tgt[:,:,h:]
        
    volume_loss = WeightedVolumeLoss(src,volume,factor=factor)
    price_loss = WeightedPriceLoss(src,price,factor=factor)
    reg_loss = regulerization(price)
    mse_loss = F.mse_loss(src,tgt)
    
    return mse_loss + alpha * price_loss + (1-alpha) * volume_loss + reg_loss*reg_factor
    
def TotalMoneyLoss(src,tgt):
    
    h = src.shape[2]//2
    price1 = src[:,:, :h]
    volume1 = src[:,:, h:]
    
    price2 = tgt[:,:,:h]
    volume2 = tgt[:,:, h:]
    
    
    money1 = price1 * volume1
    money2 = price2 * volume2
    
    return abs(torch.sum(money1)-torch.sum(money2))
    
def squre_sum(x):
    return torch.sum(abs(x))

def CELoss(src, tgt):
    loss = nn.CrossEntropyLoss()
    return loss(src, tgt.long().squeeze(-1))

def MAELoss(src, tgt):
    return F.l1_loss(src, tgt)
    
if __name__ == "__main__":
    # x1 = torch.tensor([[[1,2,1,1],[1,2,2,2],[1,2,3,3]],
    #                    ])
    # x2 = torch.tensor([[[2,3,1,1],[1,3,2,2],[4,5,3,3]],
    #                    ])
    x1=torch.randn(512,100,40).to(device)
    x2=torch.randn(512,100,40).to(device)
    print(PriceLoss(x1, x2))
    print(WeightedPriceLoss(x1,x2))
    # print(NewLoss(x1,x1))
    # print(weights_1_5)
