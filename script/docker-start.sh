#!/bin/bash

# 克隆GitLab仓库到/app/simlob-refined目录
# git clone https://gitlab.muiao.com/github/simlob-refined.git /app/simlob-refined


# 创建软链接，将/dataset/real_data链接到/app/simlob-refined/dataset/real_data
# ln -s /dataset /app/simlob-refined/dataset

# 切换工作目录到/app/simlob-refined
cd /app/simlob-refined

# 运行Python脚本
python model/base.py --config experiment/exp_settings/recon_A_share/test_cnn2_sz000001.json

# 保持容器运行
tail -f /dev/null