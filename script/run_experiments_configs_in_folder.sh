#!/bin/bash

# 指定conda环境
CONDA_ENV="simlob_final"

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 设置工作目录为当前目录
WORK_DIR=$(pwd)

# 解析--dir参数
TARGET_DIR=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir) TARGET_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# 检查是否提供了目标文件夹
if [[ -z "$TARGET_DIR" ]]; then
    echo "Error: --dir parameter is required."
    exit 1
fi

# 确保目标文件夹存在
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# 遍历目标文件夹中的每个 .json 文件并执行命令
for JSON_FILE in "$TARGET_DIR"/*.json; do
    if [[ -f "$JSON_FILE" ]]; then
        echo "Processing file: $JSON_FILE"
        python "$WORK_DIR/model/base.py" --config "$JSON_FILE"
    else
        echo "No JSON files found in $TARGET_DIR"
    fi
done

echo "All tasks completed."