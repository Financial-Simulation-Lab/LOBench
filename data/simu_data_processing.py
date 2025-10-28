"""
Created on : 2024-08-02
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/data/simu_data_processing.py
Description: Processing the Simulated Raw Data
---
Updated    : 
---
Todo       : 
"""


# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from typing import Optional,List,Tuple,Dict,Union,Protocol
from config import ConfigManager as CM 
from utils.logger import setup_logger

cm = CM()



########################################################
#######     Convert Raw CSV to Tensorboard    ##########
########################################################

import pandas as pd
from tensorboardX import SummaryWriter
import os
from data import RawDataManager as RDM 

rdm = RDM()

logger = setup_logger(log_type=0, logger_name="Convert Raw Data")

def csvs_to_tensorboard(csv_files, log_dir=None):
    """
    将指定的 CSV 文件中的时序数据转换为 TensorBoardX 可以可视化的记录。

    :param csv_files: List of CSV file paths.
    :param log_dir: Directory where TensorBoard logs will be saved.
    """
    
    
    
    if log_dir is None:
        log_dir = cm.folders.experiment
    
    for csv_file in csv_files:
        # 获取实验名称（文件名）
        
        experiment_name = os.path.splitext(os.path.basename(csv_file))[0]
        logger.info(f"Convert {csv_file} to  {os.path.join(log_dir, experiment_name)}")
        # 创建 TensorBoardX 记录器
        writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
        
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)
        
        for index, row in df.iterrows():
            # 将整行数据写入 TensorBoard
            for col in df.columns:
                writer.add_scalar(col, row[col], index)
            # writer.add_scalar(experiment_name, row.to_dict(), index)
        
        # 关闭记录器
        writer.close()
        
        logger.info(f"{os.path.join(log_dir, experiment_name)} converted.")

def convert_csv(csv_file):
    writer = SummaryWriter('dataset/exp_data/converted/')
        
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    for index, row in df.iterrows():
        # 将整行数据写入 TensorBoard
        for col in df.columns:
            writer.add_scalar(col, row[col], index)
    writer.close()
    
    logger.info(f"{csv_file[:10]} converted.")

def csv_to_tensorboard_by_param(param):
    result = rdm.select(param)
    if result is None:
        logger.error(f"Record not found for params: {param}")
        return
    
    data_file_path = result['file_path']
    convert_csv(data_file_path)
    
def csv_to_tensorboard_by_coefs(coef_csv_file):
    df = pd.read_csv(coef_csv_file)
    for _, row in df.iterrows():
        csv_to_tensorboard_by_param(row.to_list())
        break 
    
    

#======================  end   =========================


########################################################
#######           Main Test            ##########
########################################################
if __name__ == "__main__":
    csv_files = ['./data/test_data/9.csv','./data/test_data/10.csv','./data/test_data/11.csv']
    # csvs_to_tensorboard(csv_files)
    coef_file_path = "./dataset/coef_data/clambda_sensitivity.csv"
    csv_to_tensorboard_by_coefs(coef_file_path)
    
    
#======================  end   =========================