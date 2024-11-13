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

# 确保 tracklab 模块路径已添加
sys.path.append("/garage/projects/video-LLM/soccernet_bzy")  # 主项目路径
sys.path.append("/garage/projects/video-LLM/soccernet_bzy/sn-gamestate/plugins/calibration")  # 主项目路径
sys.path.append("/garage/projects/video-LLM/soccernet_bzy/tracklab")  # tracklab 路径

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

    # Instantiate all modules
    tracking_dataset = instantiate(cfg.dataset)
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
            # tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
            tracker_state.cfg = cfg
            tracking_engine = instantiate(
                cfg.engine,
                modules=pipeline,
                tracker_state=tracker_state,
            )

            # 调整批量大小，避免显存溢出
            adjust_batch_size(tracker_state, device)

            # Run tracking and visualization
            tracking_engine.track_dataset()
            # Evaluation
            evaluate(cfg, evaluator, tracker_state)
            
            log.info(f"Tracking operation on '{split_name}' set complete.")

            # 所有视频处理完成后，保存 tracker_state 并调用提取函数
            if tracker_state.save_file is not None:
                tracker_state.save(tracker_state.save_file)
                log.info(f"Saved state at : {tracker_state.save_file.resolve()}")
            #     # 调用提取和删除函数
            #     extract_image_id_and_bbox_pitch_from_pklz(cfg)

    close_environment()

    return 0


def extract_image_id_and_bbox_pitch_from_pklz(cfg):
    """
    根据配置文件中的路径，读取 pklz 文件，提取所有视频的 image_id 和 bbox_pitch, 保存为 json 文件，然后删除该 pklz 文件。
    """
    import zipfile
    import pickle
    import json
    import os
    from pathlib import Path
    from hydra.core.hydra_config import HydraConfig

    log.info("Starting extract_image_id_and_bbox_pitch_from_pklz function.")

    # 获取 Hydra 运行目录
    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.run.dir)
    log.info(f"Run directory: {run_dir}")

    pklz_file_path = run_dir / cfg.state.save_file
    log.info(f"pklz file path: {pklz_file_path}")

    if not pklz_file_path.exists():
        log.error(f"pklz file not found at {pklz_file_path}")
        return

    try:
        # 打开 pklz 文件
        with zipfile.ZipFile(pklz_file_path, 'r') as zf:
            namelist = zf.namelist()
            log.info(f"Files inside pklz: {namelist}")
            # 找到所有的 detections pkl 文件，通常命名为 '{video_id}.pkl'
            pkl_files = [name for name in namelist if name.endswith('.pkl') and not name.endswith('_image.pkl') and name != 'summary.json']
            log.info(f"Detection pkl files: {pkl_files}")

            all_data = []
            for pkl_file in pkl_files:
                log.info(f"Processing file: {pkl_file}")
                with zf.open(pkl_file, 'r') as fp:
                    detections_pred = pickle.load(fp)
                    log.info(f"detections_pred columns: {detections_pred.columns}")

                    # 检查是否包含所需的列
                    if 'image_id' in detections_pred.columns and 'bbox_pitch' in detections_pred.columns:
                        # 提取 image_id 和 bbox_pitch
                        data_to_save = detections_pred[['image_id', 'bbox_pitch']].to_dict(orient='records')
                        all_data.extend(data_to_save)
                    else:
                        log.warning(f"'image_id' or 'bbox_pitch' not found in {pkl_file}")

        # 保存为一个整体的 JSON 文件
        if all_data:
            json_file_name = "all_videos_imageid_bboxpitch.json"
            json_file_path = run_dir / json_file_name
            log.info(f"Saving JSON to {json_file_path}")
            with open(json_file_path, 'w') as json_fp:
                json.dump(all_data, json_fp, indent=4)
            log.info(f"Saved image_id and bbox_pitch to {json_file_path}")
        else:
            log.warning("No data found to save.")

    except Exception as e:
        log.error(f"Error reading pklz file: {e}")
        return

    # 删除 pklz 文件
    try:
        os.remove(pklz_file_path)
        log.info(f"Deleted pklz file {pklz_file_path}")
    except Exception as e:
        log.error(f"Error deleting pklz file: {e}")


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


def adjust_batch_size(tracker_state, device):
    """
    根据可用的显存大小，调整批量大小，避免显存溢出
    """
    if device == "cuda":
        try:
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
            # 设定调整后的批量大小为 min(original, max_allowed_batch_size)
            for module in tracker_state.pipeline.models:
                if hasattr(module, 'batch_size'):
                    # 针对特定模块调整批量大小，例如 TVCalibrationModule
                    if module.__class__.__name__ == 'TVCalibrationModule':
                        module.batch_size = 1
                        log.info(f"Set batch size to 1 for module {module.__class__.__name__} to prevent OOM.")
                    else:
                        original_batch_size = module.batch_size
                        adjusted_batch_size = min(original_batch_size, max_allowed_batch_size)
                        if adjusted_batch_size < original_batch_size:
                            log.warning(f"Reducing batch size for module {module.__class__.__name__} from {original_batch_size} to {adjusted_batch_size} to fit GPU memory.")
                        module.batch_size = adjusted_batch_size
                        log.info(f"Set batch size to {adjusted_batch_size} for module {module.__class__.__name__}")
        except Exception as e:
            log.error(f"Failed to adjust batch size: {e}")
            # 在发生错误时，可以选择不调整批量大小
    else:
        # 如果使用CPU，设置批量大小为1
        for module in tracker_state.pipeline.models:
            if hasattr(module, 'batch_size'):
                module.batch_size = 1
                log.info(f"Set batch size to 1 for module {module.__class__.__name__} on CPU")


if __name__ == "__main__":
    main()