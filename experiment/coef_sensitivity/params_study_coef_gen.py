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

import os
import random
import pandas as pd

cm = CM()


# Define the parameter ranges
# Define the parameter ranges and their precisions
param_ranges = {
    0: (70, 130, 2),    # lambda_init
    1: (5, 15, 2),      # clambda
    2: (0.1, 0.2, 4),   # freq1
    3: (0.01, 0.1, 5),  # freq2
    4: (0.001, 0.003, 5), # incre
    5: (0.02, 0.05, 5)  # delta
}


def generate_sensitivity_params(origin_params: list, param_index: int, num: int = 100, step: float = None, out_file: str = 'sensitivity_params.csv') -> str:
    """
    Generate a series of new parameter sets by modifying a specified parameter while keeping other parameters unchanged.
    
    Args:
        origin_params (list): The original parameter set.
        param_index (int): The index of the parameter to be modified.
        num (int, optional): The number of new parameter sets to generate. Defaults to 100.
        step (float, optional): The step size for modifying the specified parameter. Defaults to 1.0.
        out_file (str, optional): The file name of the generated file (relative path to the default folder of "data/meta_data/"). Defaults to 'sensitivity_params.csv'.
        
    Returns:
        str: The path of the outfile of .csv.
    """
    # Create the path to save the generated file.
    base_coef_path = cm.folders.coef
    
    out_file = os.path.join(base_coef_path, out_file)
    create_or_pass(out_file)
    
    # Get the parameter range
    param_min, param_max, precision = param_ranges[param_index]
    
    # Determine the step size if not specified
    if step is None:
        step = 10 ** -precision
    
    # Determine the direction of change
    if origin_params[param_index] + (num - 1) * step > param_max:
        step = -step
    
    # Generate new parameter sets
    param_values = []
    for i in range(num):
        new_value = origin_params[param_index] + i * step
        if new_value < param_min:
            break
        param_values.append(round(new_value, precision))
    
    new_params_list = [origin_params.copy(),]
    
    for value in param_values:
        new_params = origin_params.copy()
        new_params[param_index] = value
        new_params_list.append(new_params)
    
    # Save the new parameter sets to a CSV file
    df = pd.DataFrame(new_params_list, columns=['lambda_init', 'clambda', 'frequency1', 'frequency2', 'increment', 'delta'])
    df.to_csv(out_file, index=False)
    
    return out_file

if __name__ == "__main__":
    param_names = ['lambda_init', 'clambda', 'frequency1', 'frequency2', 'increment', 'delta']
    coef = generate_coef()
    for i in range(6):
        generate_sensitivity_params(coef, i, num=100, out_file=f"{param_names[i]}_sensitivity.csv")
        