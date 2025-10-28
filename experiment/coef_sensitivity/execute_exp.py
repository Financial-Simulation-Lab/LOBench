"""
Created on : 2024-08-02
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/experiment/params_study.ipynb
Description: 参数敏感性实验
---
Updated    : 
---
Todo       : 
"""

# Insert the path into sys.path for importing.
# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional,List,Tuple,Dict,Union,Protocol
from utils.file import create_or_pass
from simulation.generator import generate_coef
from config import ConfigManager as CM
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import setup_logger
from simulation.generator import generate_data_from_param_to_pd, compute_array_hash
from tqdm import tqdm 

import random
import pandas as pd

cm = CM()

def read_and_simulate_all_csvs(input_folder:str = None):
    """
    读取指定文件夹下的所有CSV文件，并对每个文件中的每个参数执行一次模拟。
    
    Args:
        input_folder (str): CSV文件所在的文件夹路径。
    """
    logger = setup_logger(log_type=0, logger_name='Sinsitive Exp')
    
    if input_folder is None:
        input_folder = cm.folders.coef 
        
    # 获取文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_file =  os.path.join(input_folder, csv_file)
        logger.info(f"Start processing: {csv_file}")
        simulation_via_csv(csv_file)
    
            
            
def simulation_via_csv(csv_file):
    logger = setup_logger(4,logger_name="Sensitive Exp")
    logger2 = setup_logger(0, logger_name="Coef Sensitive")
    
    df = pd.read_csv(csv_file)
    coefs = []
    for _, row in df.iterrows():
        params = [row['lambda_init'], row['clambda'], row['frequency1'], row['frequency2'], row['increment'], row['delta']]
        coefs.append(params)
    # coefs = generate_coef_n(num)
    # results = []
    
    def single_task(coef):
        logger2.info(f"Star processing: {coef}")
        generate_data_from_param_to_pd(coef,template='train')
        hash_key = compute_array_hash(coef)
        logger.info(f"{coef} -> {hash_key}")
        
    with ThreadPoolExecutor(max_workers=cm.max_cpu_workers) as executor:
        with tqdm(total=len(coefs)) as qbar:
            
            futures = {executor.submit(single_task,coef):coef for coef in coefs}
            
            for future in as_completed(futures):
                # result = future.result()
                qbar.update(1)


if __name__ == "__main__":
    read_and_simulate_all_csvs()

        