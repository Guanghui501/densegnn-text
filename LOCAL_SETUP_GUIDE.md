# 本地环境设置指南

本指南帮助你在本地环境配置 DenseGNN 和下载 JARVIS 数据集。

## 1. 环境准备

### 1.1 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n densegnn python=3.9
conda activate densegnn

# 或使用 venv
python -m venv densegnn_env
source densegnn_env/bin/activate  # Linux/Mac
# 或
densegnn_env\Scripts\activate  # Windows
```

### 1.2 安装依赖包

```bash
# 核心依赖
pip install numpy==1.26.4 tensorflow==2.15.0 scikit-learn pandas scipy matplotlib
pip install tensorflow-addons networkx sympy pyyaml ase h5py

# PyMatGen 及其依赖
pip install pymatgen monty tabulate tqdm uncertainties spglib plotly palettable

# JARVIS 工具
pip install jarvis-tools

# 其他工具
pip install brotli click
```

### 1.3 克隆项目

```bash
git clone https://github.com/Guanghui501/densegnn-text.git
cd densegnn-text

# 切换到训练分支
git checkout claude/train-densegnn-jarvis-zOOFP
```

### 1.4 设置 PYTHONPATH

```bash
# Linux/Mac
export PYTHONPATH=/path/to/densegnn-text:$PYTHONPATH

# Windows (PowerShell)
$env:PYTHONPATH="/path/to/densegnn-text;$env:PYTHONPATH"

# 或者添加到 ~/.bashrc 或 ~/.zshrc (Linux/Mac)
echo 'export PYTHONPATH=/path/to/densegnn-text:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## 2. 下载 JARVIS 数据集

### 2.1 使用提供的下载脚本

```bash
python download_jarvis_bulk_modulus.py
```

### 2.2 手动下载（如果脚本失败）

```python
# save as download_manual.py
from jarvis.db.figshare import data as jdata
import pandas as pd
import os

# 下载 JARVIS-DFT 数据
print("正在下载 JARVIS-DFT 3D 数据集...")
dft_3d = jdata(dataset='dft_3d')
print(f"下载完成，共 {len(dft_3d)} 条记录")

# 提取 bulk_modulus_kv 数据
dataset_name = 'bulk_modulus_kv'
output_dir = '/home/datasets/jarvis_dft_3d_bulk_modulus_kv'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)

data_list = []
for i, entry in enumerate(dft_3d):
    if dataset_name in entry and entry[dataset_name] is not None:
        jid = entry.get('jid', f'jid_{i}')
        value = entry[dataset_name]
        data_list.append({'index': jid, dataset_name: value})

        # 保存 CIF 文件
        if 'atoms' in entry:
            from jarvis.core.atoms import Atoms
            atoms = Atoms.from_dict(entry['atoms'])
            cif_file = os.path.join(output_dir, dataset_name, f'{jid}.cif')
            atoms.write_cif(cif_file)

        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(dft_3d)} 条记录")

# 保存 CSV
df = pd.DataFrame(data_list)
csv_file = os.path.join(output_dir, f'{dataset_name}.csv')
df.to_csv(csv_file, index=False)

print(f"\n完成！")
print(f"共保存 {len(df)} 条有效数据")
print(f"CSV 文件: {csv_file}")
print(f"CIF 文件: {os.path.join(output_dir, dataset_name)}/")
```

运行：
```bash
python download_manual.py
```

### 2.3 验证数据下载

```bash
# 检查数据目录
ls -la /home/datasets/jarvis_dft_3d_bulk_modulus_kv/

# 应该看到:
# - bulk_modulus_kv.csv
# - bulk_modulus_kv/ (包含 .cif 文件)
```

## 3. 运行训练

### 3.1 测试数据集加载

```python
# test_dataset.py
from kgcnn.data.datasets.JarvisBulkModulusKvDataset import JarvisBulkModulusKvDataset

print("加载 JarvisBulkModulusKvDataset...")
dataset = JarvisBulkModulusKvDataset(reload=False, verbose=10)

print(f"\n数据集信息:")
print(f"- 样本数量: {len(dataset)}")
print(f"- 标签名称: {dataset.label_names}")
print(f"- 标签单位: {dataset.label_units}")

# 查看第一个样本
sample = dataset[0]
print(f"\n第一个样本的属性:")
for key in sample.keys():
    print(f"  - {key}: {type(sample[key])}")
```

运行：
```bash
python test_dataset.py
```

### 3.2 开始训练

使用 DenseGNN 模型：

```bash
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category DenseGNN \
  --model DenseGNN \
  --make make_model \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42
```

使用其他模型（Megnet, Schnet, CGCNN, coGN）：

```bash
# Megnet
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category Megnet.make_crystal_model \
  --model Megnet \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42

# Schnet
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category Schnet.make_crystal_model \
  --model Schnet \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42
```

### 3.3 在 GPU 上训练

```bash
# 使用第一个 GPU
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category DenseGNN \
  --model DenseGNN \
  --make make_model \
  --dataset JarvisBulkModulusKvDataset \
  --gpu 0 \
  --seed 42

# 使用多个 GPU
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category DenseGNN \
  --model DenseGNN \
  --make make_model \
  --dataset JarvisBulkModulusKvDataset \
  --gpu 0 1 \
  --seed 42
```

### 3.4 训练特定折（fold）

```bash
# 只训练第 0 折
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category DenseGNN \
  --model DenseGNN \
  --make make_model \
  --dataset JarvisBulkModulusKvDataset \
  --fold 0 \
  --seed 42

# 训练第 0, 1, 2 折
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category DenseGNN \
  --model DenseGNN \
  --make make_model \
  --dataset JarvisBulkModulusKvDataset \
  --fold 0 1 2 \
  --seed 42
```

## 4. 查看训练结果

训练结果保存在 `results/` 目录下：

```bash
# 查看结果目录
ls -la results/JarvisBulkModulusKvDataset/

# 结果文件包括:
# - history_fold_0.pickle: 训练历史
# - weights_fold_0.h5: 模型权重
# - model_fold_0.keras: 完整模型
# - predict_fold_0.png: 预测 vs 真实值图表
# - scaler_fold_0/: 标准化器
# - score.yaml: 性能评估指标
# - DenseGNN_hyper.json: 超参数配置
```

查看性能指标：

```bash
cat results/JarvisBulkModulusKvDataset/DenseGNN/score.yaml
```

## 5. 常见问题排查

### 问题 1: 数据目录不存在

```bash
# 错误: FileNotFoundError: /home/datasets/jarvis_dft_3d_bulk_modulus_kv/bulk_modulus_kv.csv

# 解决方案: 创建目录并下载数据
mkdir -p /home/datasets/jarvis_dft_3d_bulk_modulus_kv
python download_jarvis_bulk_modulus.py
```

### 问题 2: ModuleNotFoundError: No module named 'kgcnn'

```bash
# 解决方案: 设置 PYTHONPATH
export PYTHONPATH=/path/to/densegnn-text:$PYTHONPATH
```

### 问题 3: 网络下载失败

```bash
# 如果在代理后面，设置代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 或者禁用代理
unset HTTP_PROXY
unset HTTPS_PROXY
```

### 问题 4: 内存不足

编辑配置文件 `training/hyper/hyper_jarvis_bulk_modulus_kv.py`，减小 batch_size：

```python
"fit": {
    "batch_size": 64,  # 从 128 改为 64 或更小
    "epochs": 300,
    ...
}
```

### 问题 5: CUDA/GPU 问题

```bash
# 检查 GPU 是否可用
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 如果没有 GPU，使用 CPU
python training/train_crystal.py ... --gpu None
```

## 6. 使用 Jupyter Notebook

```python
# notebook_example.ipynb
import sys
sys.path.insert(0, '/path/to/densegnn-text')

from kgcnn.data.datasets.JarvisBulkModulusKvDataset import JarvisBulkModulusKvDataset
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
dataset = JarvisBulkModulusKvDataset(reload=False, verbose=10)

# 获取标签
labels = np.array(dataset.obtain_property("graph_labels"))

# 可视化标签分布
plt.figure(figsize=(10, 6))
plt.hist(labels, bins=50, edgecolor='black')
plt.xlabel('Bulk Modulus (GPa)')
plt.ylabel('Frequency')
plt.title('JarvisBulkModulusKv Dataset - Label Distribution')
plt.grid(True, alpha=0.3)
plt.show()

print(f"统计信息:")
print(f"- 最小值: {labels.min():.2f} GPa")
print(f"- 最大值: {labels.max():.2f} GPa")
print(f"- 平均值: {labels.mean():.2f} GPa")
print(f"- 标准差: {labels.std():.2f} GPa")
```

## 7. 自定义训练

如果需要自定义训练流程，可以参考 `test.py` 文件：

```bash
# 复制并修改
cp test.py my_custom_training.py

# 编辑 my_custom_training.py
# 修改超参数、模型配置等

# 运行自定义训练
python my_custom_training.py
```

## 8. 完整示例脚本

保存为 `quick_start.sh`：

```bash
#!/bin/bash

# 1. 激活环境
conda activate densegnn  # 或 source densegnn_env/bin/activate

# 2. 设置路径
export PYTHONPATH=/path/to/densegnn-text:$PYTHONPATH

# 3. 下载数据（首次运行）
# python download_jarvis_bulk_modulus.py

# 4. 运行训练
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category DenseGNN \
  --model DenseGNN \
  --make make_model \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42

echo "训练完成！查看结果："
ls -la results/JarvisBulkModulusKvDataset/DenseGNN/
```

运行：
```bash
chmod +x quick_start.sh
./quick_start.sh
```

## 9. 参考资源

- **项目文档**: `README.md`, `JARVIS_TRAINING_README.md`
- **JARVIS 官网**: https://jarvis.nist.gov/
- **JARVIS 文档**: https://jarvis-tools.readthedocs.io/
- **DenseGNN 论文**: 查看 `README.md`
- **问题反馈**: GitHub Issues

祝训练顺利！🚀
