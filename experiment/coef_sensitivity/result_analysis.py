"""
Created on : 2024-08-02
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/experiment/coef_sensitivity/result_analysis.py
Description: Analyze the results of the Coefficient Sensitivity Experiment
---
Updated    : 
---
Todo       : 
"""

# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional,List,Tuple,Dict,Union,Protocol

from data import RawDataManager as RDM 
import pandas as pd
from pprint import pprint as print 
from tensorboardX import SummaryWriter
from data.data_compare import MSE_range

rdm = RDM()

def analyze_results_to_tensorboard(csv_file):
    """ 分析结果，读取CSV文件中的coef，使用rdm的select方法查询数据文件的路径， 并将该数据读入DataFrame中并返回。
    Args:
        csv_file (str): CSV文件路径。

    Returns:
        pd.DataFrame: 包含数据文件内容的DataFrame。
    """
    # 读取 CSV 文件
    
    
    df = pd.read_csv(csv_file)

    # 假设 CSV 文件中有一列名为 'coef'
    for index, coef in df.iterrows():
        result=rdm.select(coef)
        # print(result)
        data_file_path = result['file_path']
        
        data_df = pd.read_csv(data_file_path)
        # print(data_df.head(5))
        
        data_df = data_df.iloc[1000:5000,1:]
        
        analysis = {
            'id': result['_id'],
            'mean': data_df.mean(),
            'std':data_df.std(),
            'max': data_df.max(),
            'min': data_df.min(),
        }
        print(analysis['max']-analysis['min'])
        break 
        # writer = SummaryWriter(f"./dataset/exp_data/{analysis['id']}")
        for stat_name, stat_values in analysis.items():
            for column_name, value in stat_values.items():
                # writer.add_scalar(f'{coef[:5]}/{column_name}/{stat_name}', value,index)
                print(column_name,stat_name,value)
        # 关闭SummaryWriter
        # writer.close()
    return
from data.data_processing import data_preprocess_from_csv
def compare_results(coef_csv, minmax_scale=False):
    coefs = pd.read_csv(coef_csv)
    result = []
    to_compare_coef = coefs.iloc[0]
    writer = SummaryWriter(f"./dataset/exp_data/{coef_csv}")
    data_csv_1 = rdm.select(to_compare_coef)['file_path']
    nd1 = data_preprocess_from_csv(data_csv_1,minmax_scale=minmax_scale)
    
    for index , coef in coefs.iterrows():
        data_csv_2 = rdm.select(coef)['file_path']
        nd2 = data_preprocess_from_csv(data_csv_2,minmax_scale=minmax_scale)
        diff = MSE_range(nd1,nd2)
        print(f"{index-1}:{index} diff is ----> {diff}")
        # break 
        writer.add_scalar(f'MSE', diff, index)
        nd2 = nd1 
        
def compare_all_csv_in_folder(folder:str):
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            compare_results(os.path.join(folder,file))
                 
                
if __name__ == "__main__":
    
    # compare_results("./dataset/coef_data/clambda_sensitivity.csv")
    compare_all_csv_in_folder("./dataset/coef_data")
