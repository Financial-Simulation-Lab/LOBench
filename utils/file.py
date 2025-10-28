"""
Created on : 2024-07-02
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/SimLOB/simlob-refined/utils/file.py
Description: Generic Utility Functions
---
Updated    : 
---
Todo       : 
"""

# Insert the path into sys.path for importing.
import sys
import os
from typing import AnyStr
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional,List,Tuple,Dict,Union,Protocol
from config.config import ConfigManager as CM 

from tqdm import tqdm 

import torch 

cm = CM()



def create_or_pass(file_path:str,debug:bool = False) -> bool:
    
    """
    * Check if the file path exists, or create all the folders needed.
    
    ! The file_path should contain the file, like "path/to/file.txt". 
    * `create_or_pass` will help you create all the folder on the path recursively.

    Args:
        file_path (str): The file path.Required.
        
    Returns:
        bool: if the path exists or created return True, else return False.
    """
    
    if os.path.exists(file_path):
        if debug:
            print(f"路径 '{file_path}' 已经存在。")
        return True
    else:
        try:
            # 获取路径的目录部分
            directory = os.path.dirname(file_path)
            # 如果目录部分为空，说明file_path本身是一个目录
            if not directory:
                directory = file_path

            # 递归创建目录
            os.makedirs(directory, exist_ok=True)
            if debug:
                print(f"路径 '{file_path}' 已成功创建。")
            return True
        except Exception as e:
            if debug:
                print(f"创建路径 '{file_path}' 时出错: {e}")
            return False
        

########################################################
#######          Move File to Hash Folder     ##########
########################################################
import numpy as np
import hashlib
import shutil
import pandas as pd 

# accuracy = cm.accuracy

def compute_array_hash(array: List) -> str:
    """
    Compute the hash value of a NumPy array.
    """
    float_list_str = ','.join(f'{num}' for num in array)
    # print(float_list_str)
    array_bytes = bytes(float_list_str.encode())
    hash_value = hashlib.md5(array_bytes).hexdigest()
    return hash_value


def save_pd_to_hashed_folder(array: List, df: pd.DataFrame, base_folder:str=None, debug:bool=False  )->str:
    """
    Compute the hash value of the array, create directories based on the hash value,
    and move the file to the newly created directories with the hash value as its name.
    """
    # Compute the hash value of the array
    if not base_folder:
        base_dir = cm.folders.raw
    else:
        base_dir = base_folder
        
    hash_value = compute_array_hash(array)
    
    # Generate the two-layer folder structure from the hash value
    folder1 = hash_value[:2]
    folder2 = hash_value[2:4]
    target_dir = os.path.join(base_dir, folder1, folder2)
    
    # Recursively create directories if they do not exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate the target file path
    target_file_path = os.path.join(target_dir, f"{hash_value}.csv")
    
    try:
        df.to_csv(target_file_path)
        if debug:
            print(f"Save raw {hash_value}.csv to: {target_file_path}")
            
    except Exception as e:
        if debug:
            print(f"Error:{e}. When try to save {target_file_path} to Disk.")
        return None 
    return target_file_path
#======================  end   =========================

def merge_represented_files(batch_files,output_file):
    hidden_vas_list = []
    labels_list = []
    # padding_masks_list = []
    
    # 使用 tqdm 显示加载进度
    for batch_file in tqdm(batch_files, desc="Merging batches", unit="batch"):
        hidden_vas, labels = torch.load(batch_file)
        hidden_vas_list.append(hidden_vas)
        labels_list.append(labels)
        # padding_masks_list.append(padding_masks)
    
    # 合并所有数据
    hidden_vas = torch.cat(hidden_vas_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    # padding_masks = torch.cat(padding_masks_list, dim=0)
    
    # 保存最终合并的数据
    torch.save((hidden_vas, labels), output_file)
    print(f"All batches merged and saved to {output_file}")

def load_represented_data(file_path):
    """
    读取一个 `.pt` 文件，并返回存储在其中的 hidden_vas, labels 和 padding_masks。
    
    Args:
        file_path (str): 数据文件的路径
    
    Returns:
        tuple: 包含 (hidden_vas, labels, padding_masks) 的元组
    """
    # 使用 torch.load 加载数据
    data = torch.load(file_path)
    
    # 返回隐藏状态、标签和填充掩码
    hidden_vas, labels = data
    
    return hidden_vas, labels


if __name__ == "__main__":
    # Test the create or pass function.
    # file_path = input("请输入要检查的文件路径: ").strip()
    # create_or_pass(file_path)
    
    df = pd.DataFrame([1,2,3,4])
    # save_pd_to_hashed_folder(np.array([1,2,3,4,5,6]),df)
    ddd = compute_array_hash([0.12,0.1230,0.1234,0.12345])
    print(ddd)    