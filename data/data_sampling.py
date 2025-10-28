"""
Created on : 2024-07-10
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/SimLOB/simlob-refined/data/data_sampling.py
Description: Data Sampling Class 
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


import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from tqdm import tqdm

class TimeSeriesSampler:
    def __init__(self, input_folder: str, sampling_method: Callable, num_samples: int, output_file: str, float64=False):
        self.input_folder = input_folder
        self.sampling_method = sampling_method
        self.num_samples = num_samples
        self.output_file = output_file
        self.npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
        self.float64=float64

    def sampling(self, sample_length: int):
        num_files = len(self.npy_files)
        samples_per_file = self.num_samples // num_files
        all_samples = []

        def process_file(file: str):
            file_path = os.path.join(self.input_folder, file)
            data = np.load(file_path)
            tensor_data = torch.tensor(data)
            samples = self.sampling_method(tensor_data, sample_length, samples_per_file)
            return samples.clone().detach()

        with ThreadPoolExecutor(max_workers=32) as executor:
            results=[]
            with tqdm(total=len(self.npy_files)) as pbar:
                futures = {executor.submit(process_file,file): file for file  in self.npy_files}
                # for result in results:
                #     all_samples.append(result)
                
                for future in as_completed(futures):
                    result = future.result()
                    all_samples.append(result)
                    pbar.update(1)
                    
        all_samples_tensor = torch.cat(all_samples,dim=0)
        if not self.float64:
            all_samples_tensor=all_samples_tensor.to(torch.float32)
        torch.save(all_samples_tensor, self.output_file)

# Example sampling method: Sliding window sampling
def sliding_window_sampling(tensor_data:torch.Tensor, window_size:int, num_samples:int):
    samples = []
    step_size = max(1, (tensor_data.shape[0] - window_size) // num_samples)
    for start in range(0, tensor_data.shape[0] - window_size, step_size):
        end = start + window_size
        samples.append(tensor_data[start:end])
    return torch.stack(samples)

if __name__ == "__main__":
    # Usage
    input_folder = './dataset/simu_data/log_generate_np_no_scale'
    output_file = './dataset/train_data/no_neg_start_1000_slide_window_no_scale_float32.pt'
    num_samples = 1000000
    sample_length = 100

    sampler = TimeSeriesSampler(input_folder, sliding_window_sampling, num_samples, output_file, float64=False)
    sampler.sampling(sample_length)
