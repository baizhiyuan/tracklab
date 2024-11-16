import os
import json
import math

def split_sequences_info_files(video_bzy_dir, num_splits=16):
    # 遍历 video_bzy_dir 下的子目录（联赛级别目录）
    for league_dir in os.listdir(video_bzy_dir):
        league_path = os.path.join(video_bzy_dir, league_dir)
        if os.path.isdir(league_path):
            # 找到该联赛目录下的 sequences_info.json 文件
            json_path = os.path.join(league_path, "sequences_info.json")
            if not os.path.exists(json_path):
                print(f"在 {league_path} 中未找到 'sequences_info.json' 文件，跳过该联赛。")
                continue

            # 读取 sequences_info.json 文件
            with open(json_path, 'r') as json_file:
                sequences_info = json.load(json_file)

            # 获取版本号和序列列表
            version = sequences_info.get("version", "1.3")
            # 注意，这里获取联赛名称，因为序列信息保存在联赛名称的键下
            league_name = league_dir  # 或者从 sequences_info.keys() 中获取

            sequences = sequences_info.get(league_name, [])
            total_sequences = len(sequences)
            if total_sequences == 0:
                print(f"{league_name} 中没有序列信息，跳过。")
                continue

            # 计算每个拆分文件中的序列数量
            split_size = math.ceil(total_sequences / num_splits)

            # 开始拆分并保存
            for i in range(num_splits):
                split_sequences = sequences[i * split_size:(i + 1) * split_size]
                if not split_sequences:
                    break  # 如果没有更多的序列，提前退出

                split_sequences_info = {
                    "version": version,
                    league_name: split_sequences
                }

                split_filename = os.path.join(league_path, f"sequences_info_{i+1:02d}.json")
                with open(split_filename, 'w') as split_file:
                    json.dump(split_sequences_info, split_file, indent=4, ensure_ascii=False)

                print(f"已创建 {split_filename}")

# video_bzy_dir 的路径
video_bzy_dir = "/garage#2/projects/data/video_fps_1"

# 调用函数，将 sequences_info.json 拆分成 16 个部分
split_sequences_info_files(video_bzy_dir, num_splits=16)