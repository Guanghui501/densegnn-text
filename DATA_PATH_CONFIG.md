# JARVIS 数据集路径配置说明

## 📍 路径配置位置

数据集路径配置在以下文件中：

### 1. 主配置文件（已修改）
**文件**: `kgcnn/data/datasets/JarvisBulkModulusKvDataset.py`

现在支持自动查找和自定义路径！

### 2. 基类默认路径
**文件**: `kgcnn/data/datasets/JarvisBenchDataset2021.py:129`
```python
data_main_dir : str = os.path.join(os.path.expanduser("/home"), "datasets")
```

## 🔧 修改路径的方法

### 方法 1：使用环境中已有的数据目录（推荐）

新版本会自动检测以下路径（按优先级）：
1. `~/datasets` (用户主目录下的 datasets)
2. `/home/datasets`
3. `./datasets` (当前工作目录下的 datasets)

只需确保数据在以下任一位置：
```bash
# 选项 1: 用户主目录
~/datasets/jarvis_dft_3d_bulk_modulus_kv/
    ├── bulk_modulus_kv.csv
    └── bulk_modulus_kv/
        ├── JVASP-*.cif
        └── ...

# 选项 2: /home/datasets
/home/datasets/jarvis_dft_3d_bulk_modulus_kv/
    ├── bulk_modulus_kv.csv
    └── bulk_modulus_kv/

# 选项 3: 项目目录下
./datasets/jarvis_dft_3d_bulk_modulus_kv/
    ├── bulk_modulus_kv.csv
    └── bulk_modulus_kv/
```

### 方法 2：在超参数配置中指定路径

编辑 `training/hyper/hyper_jarvis_bulk_modulus_kv.py`：

```python
"data": {
    "dataset": {
        "class_name": "JarvisBulkModulusKvDataset",
        "module_name": "kgcnn.data.datasets.JarvisBulkModulusKvDataset",
        "config": {
            "data_main_dir": "/your/custom/path/datasets"  # 添加这行
        },
        "methods": [...]
    }
}
```

### 方法 3：直接在代码中指定

如果你直接使用 Python 代码：

```python
from kgcnn.data.datasets.JarvisBulkModulusKvDataset import JarvisBulkModulusKvDataset

# 使用自定义路径
dataset = JarvisBulkModulusKvDataset(
    data_main_dir="/your/custom/path/datasets"
)
```

### 方法 4：修改基类默认路径（不推荐）

编辑 `kgcnn/data/datasets/JarvisBenchDataset2021.py:129`：

```python
# 修改前
data_main_dir : str = os.path.join(os.path.expanduser("/home"), "datasets"),

# 修改后（改为你的路径）
data_main_dir : str = "/your/custom/path/datasets",
```

## 📂 完整数据目录结构

无论使用哪种方法，确保数据目录结构正确：

```
{data_main_dir}/
└── jarvis_dft_3d_bulk_modulus_kv/
    ├── bulk_modulus_kv.csv               # CSV 标签文件
    ├── bulk_modulus_kv/                   # CIF 文件目录
    │   ├── JVASP-1.cif
    │   ├── JVASP-2.cif
    │   └── ...
    └── bulk_modulus_kv.pymatgen.json      # PyMatGen 序列化文件（自动生成）
```

## 🚀 快速设置（推荐流程）

### 步骤 1: 确定数据路径

```bash
# 查看你的实际路径
echo $HOME/datasets
# 或
pwd
```

### 步骤 2: 创建数据目录

```bash
# 在用户主目录下创建（推荐）
mkdir -p ~/datasets/jarvis_dft_3d_bulk_modulus_kv/bulk_modulus_kv

# 或在当前项目下创建
mkdir -p ./datasets/jarvis_dft_3d_bulk_modulus_kv/bulk_modulus_kv
```

### 步骤 3: 下载数据

```bash
# 修改下载脚本中的路径
python download_jarvis_bulk_modulus.py
```

或手动修改 `download_jarvis_bulk_modulus.py` 中的路径：

```python
# 找到这一行（约第 29 行）
output_dir = '/home/datasets/jarvis_dft_3d_bulk_modulus_kv'

# 改为你的路径
output_dir = os.path.expanduser('~/datasets/jarvis_dft_3d_bulk_modulus_kv')
# 或
output_dir = '/your/custom/path/datasets/jarvis_dft_3d_bulk_modulus_kv'
```

### 步骤 4: 验证数据

```bash
# 检查文件是否存在
ls -la ~/datasets/jarvis_dft_3d_bulk_modulus_kv/bulk_modulus_kv.csv
ls ~/datasets/jarvis_dft_3d_bulk_modulus_kv/bulk_modulus_kv/ | head -5
```

### 步骤 5: 运行训练

```bash
# 现在应该可以正常运行了
python training/train_crystal.py \
  --hyper training/hyper/hyper_jarvis_bulk_modulus_kv.py \
  --category DenseGNN \
  --model DenseGNN \
  --make make_model \
  --dataset JarvisBulkModulusKvDataset \
  --seed 42
```

## 🔍 故障排查

### 错误: FileNotFoundError: /home/datasets/jarvis_dft_3d_bulk_modulus_kv/bulk_modulus_kv.csv

**原因**: 数据文件不存在

**解决方案**:
```bash
# 1. 检查数据是否存在
find ~ -name "bulk_modulus_kv.csv" 2>/dev/null

# 2. 如果找到了，记下路径，然后：
# - 方法 A: 移动数据到默认位置
mv /找到的路径/jarvis_dft_3d_bulk_modulus_kv ~/datasets/

# - 方法 B: 在配置中指定实际路径
# 编辑 training/hyper/hyper_jarvis_bulk_modulus_kv.py
# 在 "config": {} 中添加 "data_main_dir": "实际路径"

# 3. 如果没找到，需要重新下载
python download_jarvis_bulk_modulus.py
```

### 验证路径配置

```python
# 测试脚本
python -c "
import os
from kgcnn.data.datasets.JarvisBulkModulusKvDataset import JarvisBulkModulusKvDataset

# 方式 1: 使用默认路径
print('方式 1: 使用默认路径')
try:
    dataset = JarvisBulkModulusKvDataset()
    print(f'✓ 成功! 数据目录: {dataset.data_directory}')
except Exception as e:
    print(f'✗ 失败: {e}')

# 方式 2: 使用自定义路径
print('\n方式 2: 使用自定义路径')
custom_path = os.path.expanduser('~/datasets')
try:
    dataset = JarvisBulkModulusKvDataset(data_main_dir=custom_path)
    print(f'✓ 成功! 数据目录: {dataset.data_directory}')
except Exception as e:
    print(f'✗ 失败: {e}')
"
```

## 📝 常见路径示例

| 环境 | 推荐路径 |
|------|---------|
| Linux 服务器 | `~/datasets` 或 `/data/datasets` |
| Windows | `C:\Users\YourName\datasets` |
| Mac | `~/datasets` |
| Docker 容器 | `/workspace/datasets` 或 `/data/datasets` |
| Google Colab | `/content/datasets` |
| Jupyter | `./datasets` (相对路径) |

## ✅ 检查清单

使用前请确认：

- [ ] 数据目录已创建
- [ ] CSV 文件存在: `{path}/jarvis_dft_3d_bulk_modulus_kv/bulk_modulus_kv.csv`
- [ ] CIF 文件目录存在: `{path}/jarvis_dft_3d_bulk_modulus_kv/bulk_modulus_kv/`
- [ ] 有足够的磁盘空间 (至少 1GB)
- [ ] 有读写权限

## 💡 推荐配置

**生产环境**（服务器）:
```bash
export JARVIS_DATA_DIR="/data/datasets"
mkdir -p $JARVIS_DATA_DIR/jarvis_dft_3d_bulk_modulus_kv
```

**开发环境**（本地）:
```bash
mkdir -p ~/datasets/jarvis_dft_3d_bulk_modulus_kv
```

**临时测试**:
```bash
mkdir -p ./datasets/jarvis_dft_3d_bulk_modulus_kv
```
