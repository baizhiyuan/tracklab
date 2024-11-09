import os
import rich.logging
import torch
import hydra
import warnings
import logging

from tracklab.utils import monkeypatch_hydra, \
    progress  # needed to avoid complex hydra stacktraces when errors occur in "instantiate(...)"
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

import sys
sys.path.append("/garage/projects/video-LLM/soccernet_bzy")  # 主项目路径
sys.path.append("/garage/projects/video-LLM/soccernet_bzy/sn-gamestate/plugins/calibration")  # 主项目路径
sys.path.append("/garage/projects/video-LLM/soccernet_bzy/tracklab")  # tracklab 路径

@hydra.main(version_base=None, config_path="pkg://tracklab.configs", config_name="config")
def main(cfg):
# @hydra.main(config_path="/garage/projects/video-LLM/soccernet_bzy/sn-gamestate/sn_gamestate/configs", config_name="soccernet")
# def main(cfg: DictConfig):
    device = init_environment(cfg)
    video_dir = "/garage#2/projects/data/videos/"
    save_root = "/garage/projects/video-LLM/position_dataset_bzy/"
    # Instantiate all modules
    tracking_dataset = instantiate(cfg.dataset)
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)

    modules = []
    if cfg.pipeline is not None:
        for name in cfg.pipeline:
            module = cfg.modules[name]
            inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
            modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    # Train tracking modules
    for module in modules:
        if module.training_enabled:
            module.train()

    # Test tracking
    if cfg.test_tracking:
        log.info(f"Starting tracking operation on {cfg.dataset.eval_set} set.")

        # Init tracker state and tracking engine
        tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
        tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
        )

        # Run tracking and visualization
        tracking_engine.track_dataset()

        # Evaluation
        evaluate(cfg, evaluator, tracker_state)

        # Save tracker state
        if tracker_state.save_file is not None:
            log.info(f"Saved state at : {tracker_state.save_file.resolve()}")

    close_enviroment()

    return 0


def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )


def init_environment(cfg):
    # For Hydra and Slurm compatibility
    progress.use_rich = cfg.use_rich
    set_sharing_strategy()  # Do not touch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: '{device}'.")
    wandb.init(cfg)
    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))
    if cfg.use_rich:
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        # TODO : Fix for mmcv fix. This should be done in a nicer way
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.INFO)
    return device


def close_enviroment():
    wandb.finish()


def evaluate(cfg, evaluator, tracker_state):
    if cfg.get("eval_tracking", True) and cfg.dataset.nframes == -1:
        log.info("Starting evaluation.")
        evaluator.run(tracker_state)
    elif cfg.get("eval_tracking", True) == False:
        log.warning("Skipping evaluation because 'eval_tracking' was set to False.")
    else:
        log.warning(
            "Skipping evaluation because only part of video was tracked (i.e. 'cfg.dataset.nframes' was not set "
            "to -1)"
        )

# """从 TrackingSet 提取 2D 位置数据并保存"""
# def extract_position_from_set(self, tracking_set):
#     for video_id in tracking_set.video_metadata.index:
#         video_metadata = tracking_set.video_metadata.loc[video_id]
#         video_name = video_metadata['name']
#         save_dir = self.save_root / video_name
#         save_dir.mkdir(parents=True, exist_ok=True)

#         # 获取当前视频的所有检测信息
#         video_detections = tracking_set.detections[tracking_set.detections['video_id'] == video_id]
#         frame_positions = {}

#         for frame_id in sorted(video_detections['image_id'].unique()):
#             frame_detections = video_detections[video_detections['image_id'] == frame_id]
#             positions = frame_detections['bbox_ltwh'].apply(lambda x: [(x[0] + x[2] / 2), (x[1] + x[3] / 2)]).tolist()
#             frame_positions[int(frame_id)] = positions

#         # 保存当前视频的 2D 位置数据
#         save_path = save_dir / f"{video_name}_positions.json"
#         with open(save_path, 'w') as f:
#             json.dump(frame_positions, f, indent=4)

# def extract_all_positions(self):
#     """从所有 split 数据集提取位置数据"""
#     for split_name in self.split:
#         tracking_set = self.soccer_dataset.sets[split_name]
#         if tracking_set:
#             log.info(f"Extracting positions for {split_name} set.")
#             self.extract_position_from_set(tracking_set)

if __name__ == "__main__":
    main()
