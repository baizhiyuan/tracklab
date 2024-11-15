import os
import json

# video_bzy 目录的路径
video_bzy_dir = "/garage#2/projects/data/video_fps_1"

# 遍历 video_bzy 下的子目录（联赛级别目录）
for league_dir in os.listdir(video_bzy_dir):
    league_path = os.path.join(video_bzy_dir, league_dir)
    if os.path.isdir(league_path):
        sequences = []
        id_counter = 1  # ID 计数器起始值为 1

        # 收集 league_dir 目录中所有赛事（赛季和比赛）的序列信息
        for season_dir in os.listdir(league_path):
            season_path = os.path.join(league_path, season_dir)
            if os.path.isdir(season_path):
                for match_dir in os.listdir(season_path):
                    match_path = os.path.join(season_path, match_dir)
                    if os.path.isdir(match_path):
                        # 对于每个比赛目录，我们现在预期存在 '1_224p' 和 '2_224p' 子目录（因为目录结构改变）
                        for video_subdir in os.listdir(match_path):
                            video_path = os.path.join(match_path, video_subdir)
                            if os.path.isdir(video_path):
                                img1_path = os.path.join(video_path, "img1")
                                if os.path.exists(img1_path):
                                    # 统计 img1 文件夹中的帧（图像）数
                                    n_frames = len([f for f in os.listdir(img1_path) if f.endswith(".jpg")])
                                    sequences.append({
                                        "id": id_counter,  # 使用唯一 ID
                                        "name": os.path.join(season_dir, match_dir, video_subdir),  # 包含赛季、比赛和视频子目录的信息
                                        "n_frames": n_frames
                                    })
                                    id_counter += 1  # 增加 ID 计数器以确保唯一性

        # 为当前联赛目录创建 sequences_info.json
        sequences_info = {
            "version": "1.3",
            league_dir: sequences
        }

        # 将序列信息保存到联赛目录中的 sequences_info.json
        json_path = os.path.join(league_path, "sequences_info.json")
        with open(json_path, 'w') as json_file:
            json.dump(sequences_info, json_file, indent=4, ensure_ascii=False)

        print(f"已创建 {json_path}")