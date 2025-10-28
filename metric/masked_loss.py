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
    
from .loss_func import Weights

from typing import Optional,List,Tuple,Dict,Union,Protocol
import torch.nn.functional as F 
import numpy as np 
import torch 



def MaskedPriceLoss(src, tgt, mask):
    """src和tgt的价格损失函数.

    Args:
        src (torch.tensor): 输入得LOBs
        tgt (torch.tensor): 输出得LOBs

    Returns:
        _type_: _description_
    """
    
    h=src.shape[2]//2
    mask_half = mask[:h]
    price1 = src[:, :, :h]
    
    if tgt.shape[2] == h:
        price2 = tgt
    else:
        price2 = tgt[:, :, :h]
    
    # 计算每个样本的均方误差
    mse_loss = F.mse_loss(price1*mask_half, price2*mask_half)
    # mse_loss = mse_loss.mean(dim=(1, 2))  # 对每个样本求均值
    
    return mse_loss

def MaskedVolumeLoss(src, tgt, mask):
    """src和tgt的价格损失函数.

    Args:
        src (torch.tensor): 输入得LOBs
        tgt (torch.tensor): 输出得LOBs

    Returns:
        _type_: _description_
    """
    
    h=src.shape[2]//2
    mask_half = mask[:h]
    volume1 = src[:, :, h:]
    
    if tgt.shape[2] == h:
        volume2 = tgt
    else:
        volume2 = tgt[:, :, h:]
    
    # 计算每个样本的均方误差
    mse_loss = F.mse_loss(volume1*mask_half, volume2*mask_half)
    # mse_loss = mse_loss.mean(dim=(1, 2))  # 对每个样本求均值
    
    return mse_loss

def WeightedMaskedPriceLoss(src,tgt,mask,factor=1.5):
    
    
    h = src.shape[2]//2
    mask_half = mask[:h]
    masked_weights = Weights(r=factor,h=h)*mask_half
    
    price1 = src[:, :, :h]
    
    if tgt.shape[2] == h:
        price2 = tgt
    else:
        price2 = tgt[:, :, :h]
    
    # 计算每个样本的均方误差
    mse_loss = F.mse_loss(price1*masked_weights, price2*masked_weights)
    # mse_loss = mse_loss.mean(dim=(1, 2))  # 对每个样本求均值
    
    return mse_loss

def WeightedMaskedVolumeLoss(src,tgt,mask,factor=1.5):
    h = src.shape[2]//2
    mask_half = mask[:h]
    masked_weights = Weights(r=factor,h=h)*mask_half
    
    price1 = src[:, :, h:]
    
    if tgt.shape[2] == h:
        price2 = tgt
    else:
        price2 = tgt[:, :, h:]
    
    # 计算每个样本的均方误差
    mse_loss = F.mse_loss(price1*masked_weights, price2*masked_weights)
    # mse_loss = mse_loss.mean(dim=(1, 2))  # 对每个样本求均值
    
    return mse_loss

def MaskedMSELoss(src, tgt, mask):
    mse_loss = F.mse_loss(src[mask == 0], tgt[mask == 0])
    return mse_loss