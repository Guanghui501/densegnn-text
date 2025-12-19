# JarvisBulkModulusKvDataset 训练指南

本指南介绍如何使用 DenseGNN 在 JarvisBulkModulusKvDataset 上进行训练。

## 1. 环境准备

### 1.1 安装依赖

```bash
# 安装必要的 Python 包
pip install numpy tensorflow==2.15.0 scikit-learn pandas scipy matplotlib
pip install tensorflow-addons pymatgen jarvis-tools networkx sympy pyyaml ase h5py
pip install monty tabulate tqdm uncertainties spglib plotly palettable brotli
```

### 1.2 设置 PYTHONPATH

```bash
export PYTHONPATH=/home/user/densegnn-text:$PYTHONPATH
```

## 2. 下载数据集

### 方法 1: 使用提供的下载脚本

```bash
python download_jarvis_bulk_modulus.py
```

该脚本会：
- 从 JARVIS-DFT 数据库下载 bulk_modulus_kv 数据
- 将数据保存到 `/home/datasets/jarvis_dft_3d_bulk_modulus_kv/`
- 保存 CSV 文件和结构文件 (CIF 格式)

### 方法 2: 手动下载

如果自动下载失败（例如网络限制），你可以：
1. 访问 https://jarvis.nist.gov/
2. 下载 JARVIS-DFT 3D 数据集
3. 提取 bulk_modulus_kv 数据
4. 将文件放置在正确的目录

## 3. 运行训练

### 3.1 使用 DenseGNN 模型（推荐）

```bash
PYTHONPATH=/home/user/densegnn-text:$PYTHONPATH python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category DenseGNN \
  --model DenseGNN \
  --make make_model \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42
```

### 3.2 使用其他模型

配置文件还支持以下模型：

#### Megnet
```bash
PYTHONPATH=/home/user/densegnn-text:$PYTHONPATH python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category Megnet.make_crystal_model \
  --model Megnet \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42
```

#### Schnet
```bash
PYTHONPATH=/home/user/densegnn-text:$PYTHONPATH python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category Schnet.make_crystal_model \
  --model Schnet \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42
```

#### CGCNN
```bash
PYTHONPATH=/home/user/densegnn-text:$PYTHONPATH python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category CGCNN.make_crystal_model \
  --model CGCNN \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42
```

#### coGN
```bash
PYTHONPATH=/home/user/densegnn-text:$PYTHONPATH python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category coGN \
  --model coGN \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42
```

## 4. 超参数配置

### 4.1 数据集信息
- **数据集名称**: JarvisBulkModulusKvDataset
- **数据来源**: JARVIS-DFT 3D 数据库
- **样本数量**: ~640 个晶体结构
- **任务类型**: 回归
- **目标变量**: bulk_modulus_kv (体模量)
- **单位**: GPa

### 4.2 模型配置特点

#### DenseGNN
- **深度**: 5 层
- **节点特征**: 128 维
- **边特征**: Voronoi 区域面积
- **图表示**: Voronoi 单元格
- **批量大小**: 128
- **训练轮数**: 300
- **优化器**: Adam with Exponential Decay

#### 其他模型
所有模型配置都包含：
- 标准化标签缩放 (StandardScaler)
- 5 折交叉验证
- 学习率调度
- 早停和模型检查点

## 5. 训练输出

训练完成后，结果将保存在 `results/` 目录下，包括：
- 模型权重文件 (.h5, .keras)
- 训练历史 (pickle 文件)
- 预测结果图表
- 性能评估指标 (YAML 文件)

## 6. 常见问题

### Q: 如果遇到 "ModuleNotFoundError: No module named 'kgcnn'"
A: 确保设置了 PYTHONPATH 环境变量

### Q: 如果数据下载失败
A:
1. 检查网络连接
2. 如果在代理后，设置 HTTP_PROXY 和 HTTPS_PROXY 环境变量
3. 尝试手动从 JARVIS 网站下载数据

### Q: 训练过程中内存不足
A: 减小配置文件中的 batch_size 参数

### Q: 如何在 GPU 上训练
A: 使用 `--gpu` 参数指定 GPU ID，例如：
```bash
python training/train_crystal.py ... --gpu 0
```

## 7. 性能基准

根据 JARVIS 论文，bulk_modulus_kv 数据集的典型性能指标：
- **MAE**: ~15-20 GPa
- **RMSE**: ~25-30 GPa

## 8. 参考文献

- JARVIS-DFT: https://www.nature.com/articles/s41524-020-00440-1
- DenseGNN: [相关论文]
- 数据集详情: https://jarvis.nist.gov/

## 9. 联系方式

如有问题，请提交 Issue 或联系维护者。
