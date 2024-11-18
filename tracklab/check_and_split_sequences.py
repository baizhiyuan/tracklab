#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称: check_and_split_missing_sequences.py

描述:
    该脚本执行以下两个主要任务：
    1. 遍历指定的 `video_bzy_dir` 目录下的每个联赛（league）目录，加载其中的 `sequences_info.json` 文件。
    2. 检查每个序列条目对应的数据集目录中是否存在 `2d_bboxpitch_*.json` 文件。
       对于缺失 `2d_bboxpitch_*.json` 文件的序列，将其收集并按指定的最大条目数拆分成多个 JSON 文件（例如，`sequence_info_last_01.json`、`sequence_info_last_02.json` 等），并保存到与 `sequences_info.json` 同级的目录中。

使用方法:
    1. 确保安装了所需的依赖库：
        pip install tqdm

    2. 修改脚本中的配置部分，设置 `video_bzy_dir` 的路径和 `max_entries_per_file`。

    3. 运行脚本：
        python -m tracklab.check_and_split_sequences
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import logging

# ---------------------------- 配置部分 ---------------------------- #

# 定义包含联赛子目录的根目录路径
video_bzy_dir = "/garage#2/projects/data/video_fps_1"

# 每个拆分 JSON 文件中包含的最大序列条目数
max_entries_per_file = 2

# ---------------------------- 日志设置 ---------------------------- #

def setup_logging():
    """
    配置脚本的日志设置。
    日志将以 INFO 级别及以上的信息打印到控制台。
    若需要更多调试信息，可将级别改为 DEBUG。
    """
    logging.basicConfig(
        level=logging.INFO,  # 可改为 logging.DEBUG 以获取更详细的信息
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# ---------------------------- 辅助函数 ---------------------------- #

def load_sequences_info(sequences_info_path):
    """
    从 JSON 文件中加载序列信息。

    参数:
        sequences_info_path (Path): `sequences_info.json` 文件的路径。

    返回:
        dict: 解析后的 JSON 数据。
    """
    try:
        with open(sequences_info_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"成功加载序列信息文件: '{sequences_info_path}'.")
        return data
    except Exception as e:
        logging.error(f"加载序列信息文件 '{sequences_info_path}' 时出错: {e}")
        return None

def check_2d_bboxpitch_exists(dataset_dir):
    """
    检查给定目录中是否存在任何 `2d_bboxpitch_*.json` 文件。

    参数:
        dataset_dir (Path): 数据集目录的路径。

    返回:
        bool: 如果存在至少一个 `2d_bboxpitch_*.json` 文件，返回 True；否则返回 False。
    """
    exists = any(dataset_dir.glob("2d_bboxpitch_*.json"))
    if exists:
        logging.debug(f"在 '{dataset_dir}' 中找到 '2d_bboxpitch_*.json' 文件。")
    else:
        logging.debug(f"在 '{dataset_dir}' 中未找到 '2d_bboxpitch_*.json' 文件。")
    return exists

def split_sequences(sequences, batch_size):
    """
    将序列列表拆分成多个批次。

    参数:
        sequences (list): 要拆分的序列列表。
        batch_size (int): 每个批次的最大条目数。

    生成:
        list: 拆分后的序列批次。
    """
    for i in range(0, len(sequences), batch_size):
        yield sequences[i:i + batch_size]

def process_league(league_path, video_bzy_dir, max_entries):
    """
    处理单个联赛目录，检查缺失的 `2d_bboxpitch_*.json` 文件，并将缺失的序列拆分保存。

    参数:
        league_path (Path): 联赛目录的路径。
        video_bzy_dir (Path): `video_bzy_dir` 根目录的路径。
        max_entries (int): 每个拆分 JSON 文件中的最大序列条目数。
    """
    sequences_info_path = league_path / "sequences_info.json"
    if not sequences_info_path.exists():
        logging.warning(f"在 '{league_path}' 中未找到 'sequences_info.json' 文件。跳过该联赛。")
        return

    # 加载序列信息
    data = load_sequences_info(sequences_info_path)
    if data is None:
        logging.warning(f"无法加载 '{sequences_info_path}'。跳过该联赛。")
        return

    missing_sequences = {}

    # 遍历 JSON 文件中的每个联赛键
    for league_key, sequences in data.items():
        if not isinstance(sequences, list):
            logging.warning(f"跳过键 '{league_key}'，因为其不包含序列列表。")
            continue

        logging.info(f"正在处理联赛 '{league_key}'，共有 {len(sequences)} 个序列。")
        missing_sequences[league_key] = []

        # 遍历每个序列条目
        for entry in tqdm(sequences, desc=f"检查联赛 '{league_key}' 的序列", unit="条目"):
            entry_id = entry.get('id', 'N/A')
            name = entry.get('name')
            if not name:
                logging.warning(f"序列 ID '{entry_id}' 缺少 'name' 字段。跳过该序列。")
                continue

            # 构建数据集目录路径，确保包含联赛目录
            dataset_dir = league_path / Path(name)
            if not dataset_dir.exists() or not dataset_dir.is_dir():
                logging.warning(f"数据集目录 '{dataset_dir}' 不存在。将序列 ID '{entry_id}' 标记为缺失。")
                missing_sequences[league_key].append(entry)
                continue

            # 检查是否存在 `2d_bboxpitch_*.json` 文件
            if not check_2d_bboxpitch_exists(dataset_dir):
                missing_sequences[league_key].append(entry)
            else:
                logging.debug(f"序列 ID '{entry_id}' 在 '{dataset_dir}' 中已存在 '2d_bboxpitch_*.json' 文件。")

    # 移除没有缺失序列的联赛键
    missing_sequences = {k: v for k, v in missing_sequences.items() if v}

    if not missing_sequences:
        logging.info(f"联赛 '{league_path.name}' 中所有序列均已生成对应的 '2d_bboxpitch_*.json' 文件。无需拆分。")
        return

    # 拆分并保存缺失的序列信息
    file_index = 1
    for league_key, sequences in missing_sequences.items():
        logging.info(f"联赛 '{league_key}' 中找到 {len(sequences)} 个缺失的序列。")
        batches = list(split_sequences(sequences, max_entries))
        for batch in batches:
            output_filename = f"sequence_info_last_{file_index:02d}.json"
            output_path = league_path / output_filename
            output_data = {
                league_key: batch
            }
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)
                logging.info(f"已保存 {len(batch)} 个序列到 '{output_path}'。")
                assert len(batch) <= max_entries, f"批次大小 {len(batch)} 超过最大限制 {max_entries}。"
                file_index += 1
            except AssertionError as ae:
                logging.error(ae)
            except Exception as e:
                logging.error(f"保存到 '{output_path}' 时出错: {e}")

def main():
    """
    主函数，执行检查和拆分缺失的序列信息的操作。
    """
    # 配置日志
    setup_logging()

    # 将字符串路径转换为 Path 对象
    video_bzy_dir_path = Path(video_bzy_dir)

    # 验证输入路径
    if not video_bzy_dir_path.is_dir():
        logging.error(f"指定的 'video_bzy_dir' 目录 '{video_bzy_dir}' 不存在。")
        return

    logging.info(f"输出文件将保存到与 'sequences_info.json' 同级的联赛目录中。")

    # 遍历每个联赛目录并处理
    league_dirs = [d for d in video_bzy_dir_path.iterdir() if d.is_dir()]
    if not league_dirs:
        logging.warning(f"在 '{video_bzy_dir_path}' 中未找到任何联赛目录。退出脚本。")
        return

    logging.info(f"找到 {len(league_dirs)} 个联赛目录需要处理。")

    for league_dir in league_dirs:
        logging.info(f"正在处理联赛目录: '{league_dir.name}'。")
        process_league(league_dir, video_bzy_dir_path, max_entries_per_file)

    logging.info("所有联赛目录的处理已完成。")

if __name__ == "__main__":
    main()