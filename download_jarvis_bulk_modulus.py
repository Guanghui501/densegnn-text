#!/usr/bin/env python
"""
下载 JARVIS-DFT bulk_modulus_kv 数据集

使用方法:
    python download_jarvis_bulk_modulus.py
"""

import os
import pandas as pd
from jarvis.db.figshare import data as jdata


def download_jarvis_bulk_modulus_kv():
    """下载 JARVIS bulk_modulus_kv 数据集"""

    print("正在下载 JARVIS-DFT 3D 数据集...")
    dataset_name = 'bulk_modulus_kv'

    try:
        # 下载 JARVIS-DFT 数据
        dft_3d = jdata(dataset='dft_3d')
        print(f"成功下载 JARVIS-DFT 数据，共 {len(dft_3d)} 条记录")

        # 创建目录
        output_dir = '/home/datasets/jarvis_dft_3d_bulk_modulus_kv'
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建目录: {output_dir}")

        # 提取 bulk_modulus_kv 数据和结构文件
        data_list = []
        struct_dir = os.path.join(output_dir, 'bulk_modulus_kv')
        os.makedirs(struct_dir, exist_ok=True)

        for i, entry in enumerate(dft_3d):
            if dataset_name in entry:
                value = entry[dataset_name]
                jid = entry.get('jid', f'jid_{i}')

                # 只保存有效数据
                if value is not None and value != 'na':
                    data_list.append({'index': jid, dataset_name: value})

                    # 保存结构文件 (如果有的话)
                    if 'atoms' in entry:
                        from jarvis.core.atoms import Atoms
                        atoms = Atoms.from_dict(entry['atoms'])
                        cif_file = os.path.join(struct_dir, f'{jid}.cif')
                        atoms.write_cif(cif_file)

                # 打印进度
                if (i + 1) % 1000 == 0:
                    print(f"已处理 {i + 1}/{len(dft_3d)} 条记录, 找到 {len(data_list)} 条有效数据")

        # 保存 CSV 文件
        df = pd.DataFrame(data_list)
        output_csv = os.path.join(output_dir, f'{dataset_name}.csv')
        df.to_csv(output_csv, index=False)

        print(f"\n数据下载完成!")
        print(f"CSV 文件保存至: {output_csv}")
        print(f"结构文件保存至: {struct_dir}")
        print(f"共有 {len(df)} 条有效数据")
        print("\n前几行数据:")
        print(df.head())
        print("\n数据统计:")
        print(df[dataset_name].describe())

        return df

    except Exception as e:
        print(f"下载失败: {e}")
        print("\n备选方案:")
        print("1. 检查网络连接")
        print("2. 如果在代理后面，请设置 HTTP_PROXY 和 HTTPS_PROXY 环境变量")
        print("3. 或者从 https://jarvis.nist.gov/ 手动下载数据")
        raise


if __name__ == "__main__":
    download_jarvis_bulk_modulus_kv()
