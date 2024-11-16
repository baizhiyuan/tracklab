import os
import sys
import pickle
import rich.logging
import torch
import hydra
import warnings
import logging
import json
import pandas as pd
from pathlib import Path
import zipfile
from tracklab.utils import monkeypatch_hydra, progress  # needed to avoid complex hydra stacktraces when errors occur in "instantiate(...)"
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb
from omegaconf import DictConfig
from torch.cuda.amp import autocast, GradScaler

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="pkg://tracklab.configs", config_name="config")
def main(cfg):
    device = init_environment(cfg)
    # 设置CuDNN加速
    torch.backends.cudnn.benchmark = True

    # 获取 Hydra 配置
    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.run.dir)
    log.info(f"Hydra run directory: {run_dir}")
    log.info(f"State save file: {cfg.state.save_file}")
    log.info(f"sequences_info_filename: {cfg.dataset.sequences_info_filename}")

    # Instantiate all modules
    # tracking_dataset = instantiate(cfg.dataset)
    tracking_dataset = instantiate(cfg.dataset, sequences_info_filename=cfg.dataset.sequences_info_filename)
    # 在主函数中
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)

    modules = []
    if cfg.pipeline is not None:
        for name in cfg.pipeline:
            module_cfg = cfg.modules[name]
            # 传递device参数，确保模块在GPU上运行
            inst_module = instantiate(module_cfg, device=device, tracking_dataset=tracking_dataset)
            modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    # Train tracking modules
    for module in modules:
        if module.training_enabled:
            module.train()

    # Test tracking
    if cfg.test_tracking:
        # 确保 cfg.dataset.eval_set 是一个列表
        if isinstance(cfg.dataset.eval_set, str):
            eval_sets = [cfg.dataset.eval_set]
        else:
            eval_sets = cfg.dataset.eval_set

        for split_name in eval_sets:
            if split_name not in tracking_dataset.sets:
                raise KeyError(f"Trying to access a '{split_name}' split of the dataset that is not available. Available splits are {list(tracking_dataset.sets.keys())}. Make sure this split name is correct or is available in the dataset folder.")

            log.info(f"Starting tracking operation on '{split_name}' set.")
            # Init tracker state and tracking engine
            tracking_set = tracking_dataset.sets[split_name]
            print(tracking_set.video_metadatas.loc[tracking_set.video_metadatas['id']==3])
            tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
            tracker_state.cfg = cfg
            tracker_state.eval_name = split_name
            tracking_engine = instantiate(
                cfg.engine,
                modules=pipeline,
                tracker_state=tracker_state,
            )

            # 调整批量大小，避免显存溢出
            # adjust_batch_size(tracker_state, device)
            # 清空 GPU 缓存
            torch.cuda.empty_cache()
            # Run tracking and visualization
            tracking_engine.track_dataset()

            # Evaluation
            evaluate(cfg, evaluator, tracker_state)
            log.info(f"Tracking operation on '{split_name}' set complete.")

            # 所有视频处理完成后，保存 tracker_state 并调用提取函数
            if tracker_state.save_file is not None:
                tracker_state.save()
                # log.info(f"Saved state at : {tracker_state.save_file.resolve()}")

    close_environment()

    return 0


def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )


def init_environment(cfg):
    # For Hydra and Slurm compatibility
    progress.use_rich = cfg.use_rich
    set_sharing_strategy()  # Do not touch

    # 设置设备为GPU（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: '{device}'.")

    wandb.init(cfg)
    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))
    if cfg.use_rich:
        for handler in log.root.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.ERROR)
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        # TODO : Fix for mmcv fix. This should be done in a nicer way
        for handler in log.root.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.INFO)
    return device


def close_environment():
    wandb.finish()


def evaluate(cfg, evaluator, tracker_state):
    if cfg.get("eval_tracking", True) and cfg.dataset.nframes == -1:
        log.info("Starting evaluation.")
        evaluator.run(tracker_state)
    elif not cfg.get("eval_tracking", True):
        log.warning("Skipping evaluation because 'eval_tracking' was set to False.")
    else:
        log.warning(
            "Skipping evaluation because only part of video was tracked (i.e. 'cfg.dataset.nframes' was not set "
            "to -1)"
        )


def adjust_batch_size(tracker_state, device, reduce=False):
    """
    通过尝试不同的批处理大小来避免显存溢出。如果遇到 OOM 错误，将批处理大小减半并重试。
    """
    if device == "cuda":
        for module in tracker_state.pipeline.models:
            if hasattr(module, 'batch_size'):
                original_batch_size = module.batch_size
                current_batch_size = module.batch_size

                if reduce:
                    current_batch_size = max(1, current_batch_size // 2)
                else:
                    # 初始调整，可以基于可用显存进行估算
                    try:
                        # 获取可用显存
                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        reserved_memory = torch.cuda.memory_reserved(0)
                        allocated_memory = torch.cuda.memory_allocated(0)
                        available_memory = total_memory - (reserved_memory + allocated_memory)
                        log.info(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")
                        log.info(f"Reserved GPU memory: {reserved_memory / (1024 ** 3):.2f} GB")
                        log.info(f"Allocated GPU memory: {allocated_memory / (1024 ** 3):.2f} GB")
                        log.info(f"Available GPU memory: {available_memory / (1024 ** 3):.2f} GB")

                        # 根据可用显存，计算允许的最大批量大小
                        approximate_memory_per_sample = 40 * 1024 * 1024  # 40MB
                        max_allowed_batch_size = int(available_memory / approximate_memory_per_sample)
                        max_allowed_batch_size = max(1, max_allowed_batch_size)  # 保证至少为1
                        log.info(f"Calculated max allowed batch size: {max_allowed_batch_size}")

                        # 为了避免某些模块的 batch_size 太大，可以设置一个合理的上限
                        current_batch_size = min(original_batch_size, max_allowed_batch_size)
                    except Exception as e:
                        log.error(f"Failed to calculate max allowed batch size: {e}")
                        current_batch_size = original_batch_size  # 使用原始批量大小

                if current_batch_size < original_batch_size:
                    log.warning(f"Reducing batch size for module {module.__class__.__name__} from {original_batch_size} to {current_batch_size}.")
                module.batch_size = current_batch_size
                log.info(f"Set batch size to {current_batch_size} for module {module.__class__.__name__}")
    else:
        # 如果使用CPU，设置批处理大小为1
        for module in tracker_state.pipeline.models:
            if hasattr(module, 'batch_size'):
                module.batch_size = 1
                log.info(f"Set batch size to 1 for module {module.__class__.__name__} on CPU")




if __name__ == "__main__":
    main()