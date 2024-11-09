import os
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from SoccerNetMOT import SoccerNetMOT  # 假设 SoccerNetMOT 在相应路径已导入

log = logging.getLogger(__name__)

class SoccerNetPositionExtractor:
    def __init__(self, dataset_path, save_root, split=["train", "test"]):
        self.dataset_path = Path(dataset_path)
        self.save_root = Path(save_root)
        self.split = split
        self.soccer_dataset = SoccerNetMOT(dataset_path)  # 初始化 SoccerNetMOT

        assert self.dataset_path.exists(), f"'{self.dataset_path}' directory does not exist."
        log.info(f"Initializing SoccerNet Position Extractor from {self.dataset_path}.")

    def extract_position_from_set(self, tracking_set):
        """从 TrackingSet 提取 2D 位置数据并保存"""
        for video_id in tracking_set.video_metadata.index:
            video_metadata = tracking_set.video_metadata.loc[video_id]
            video_name = video_metadata['name']
            save_dir = self.save_root / video_name
            save_dir.mkdir(parents=True, exist_ok=True)

            # 获取当前视频的所有检测信息
            video_detections = tracking_set.detections[tracking_set.detections['video_id'] == video_id]
            frame_positions = {}

            for frame_id in sorted(video_detections['image_id'].unique()):
                frame_detections = video_detections[video_detections['image_id'] == frame_id]
                positions = frame_detections['bbox_ltwh'].apply(lambda x: [(x[0] + x[2] / 2), (x[1] + x[3] / 2)]).tolist()
                frame_positions[int(frame_id)] = positions

            # 保存当前视频的 2D 位置数据
            save_path = save_dir / f"{video_name}_positions.json"
            with open(save_path, 'w') as f:
                json.dump(frame_positions, f, indent=4)

    def extract_all_positions(self):
        """从所有 split 数据集提取位置数据"""
        for split_name in self.split:
            tracking_set = self.soccer_dataset.sets[split_name]
            if tracking_set:
                log.info(f"Extracting positions for {split_name} set.")
                self.extract_position_from_set(tracking_set)

if __name__ == "__main__":
    dataset_path = "/path/to/soccernet/dataset"
    save_root = "/path/to/save_position_data"

    extractor = SoccerNetPositionExtractor(dataset_path, save_root)
    extractor.extract_all_positions()