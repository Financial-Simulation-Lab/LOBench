"""
Created on : 2024-07-12
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/SimLOB/simlob-refined/data/data_compare.py
Description: Compare RAW CSV files 
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
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# 指定文件夹路径


# 获取文件夹下所有的CSV文件

def compare_folder(folder_path, alpha=0.5, x_step=10):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 读取所有CSV文件到一个字典中
    data = {}
    for file in csv_files:
        data[file] = pd.read_csv(file)

    # 获取所有的列名（除去 'time' 列）
    columns = data[csv_files[0]].columns[2:]

    # 创建一个画布和40个子图
    fig, axs = plt.subplots(10, 4, figsize=(20, 20), constrained_layout=True)

    # 遍历所有列并绘制子图
    for idx, col in enumerate(columns):
        ax = axs[idx // 4, idx % 4]
        for file in csv_files:
            df = data[file]
            ax.plot(df['time'][::x_step], df[col][::x_step], label=os.path.basename(file), alpha=alpha)
        
        ax.set_title(col)
        ax.legend()

    # 保存图形到文件
    plt.suptitle('Comparison of CSV Files')
    plt.savefig(os.path.join(folder_path,'comparison_plot.png'))
    plt.show()
    
def paint_csv(data,x_step=10,alpha=0.5, fig_label="Calib"):
    columns = data.columns[2:]
    if len(columns)!=40:
        print(len(columns), "\n",columns)
    
    fig, axs = plt.subplots(10, 4, figsize=(20,20),constrained_layout=True)
    
    for idx, col in enumerate(columns):
        ax = axs[idx//4, idx % 4]
        ax.plot(data['time'][::x_step],data[col][::x_step], label=fig_label, alpha=alpha)
        
        ax.set_title(col)
        ax.legend()
        
    plt.suptitle("LOB Viz")
    plt.savefig(os.path.join("/data1/zmy/project/SimLOB/simlob-refined/dataset/analy_data",fig_label+'.png'))
    plt.show()
    
def paint_csvs(datas,x_step=10,alpha=0.5, fig_label=[]):
    columns = datas[0].columns[2:]
    
    if len(fig_label)==0:
        fig_label=[f"{i}-th data" for i in range(len(datas))]
        print(fig_label)
    elif len(fig_label)!=len(datas):
        print(len(fig_label),len(datas))
        return 
    
    if len(columns)!=40:
        print(len(columns), "\n",columns)
    
    fig, axs = plt.subplots(10, 4, figsize=(20,20),constrained_layout=True)
    for idx, col in enumerate(columns):
        ax = axs[idx//4, idx % 4]
        for ii,data in enumerate(datas):
            ax.plot(data['time'][::x_step],data[col][::x_step], label=fig_label[ii], alpha=alpha)
        
        ax.set_title(col)
        ax.legend()
        
    plt.suptitle("LOB Viz")
    plt.savefig(os.path.join("/data1/zmy/project/SimLOB/simlob-refined/dataset/analy_data",'output.png'))
    plt.show()

from data.data_processing import data_preprocess_from_csv

def MSE_range(nd1,nd2,start=1000,end=5000):
    # df1 = pd.read_csv(csv_1)
    # df2 = pd.read_csv(csv_2)
    if type(nd1)==str:
        nd1 = data_preprocess_from_csv(nd1)
        nd1 = nd1[start:end,1:]
    if type(nd2)==str:
        nd2 = data_preprocess_from_csv(nd2)
        nd2 = nd2[start:end,1:]
    
    # 计算均方误差
    mse = np.mean((nd1-nd2) ** 2)
    
    return mse 
    
    


if __name__ == "__main__":
    # fp = 'dataset/analy_data/to_be_compared'
    # compare_folder(fp)
    csv_path = "./dataset/analy_data/to_be_compared/raw_7.csv"
    csv_path2 = "./dataset/analy_data/to_be_compared/raw_8.csv"
    # df1 = pd.read_csv(csv_path)
    # df2 = pd.read_csv(csv_path2)
    # datas=[df1,df2]
    # paint_csvs(datas)
    res = MSE_range(csv_path,csv_path2)
    print(res, type(res))
    
    
