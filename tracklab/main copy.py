import os
import rich.logging
import torch
import hydra
import warnings
import logging

from tracklab.utils import monkeypatch_hydra, progress  # needed to avoid complex hydra stacktraces when errors occur in "instantiate(...)"
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# import sys
# sys.path.append("/garage/projects/video-LLM/soccernet_bzy")  # 主项目路径
# sys.path.append("/garage/projects/video-LLM/soccernet_bzy/sn-gamestate/plugins/calibration")  # 主项目路径
# sys.path.append("/garage/projects/video-LLM/soccernet_bzy/tracklab")  # tracklab 路径

@hydra.main(version_base=None, config_path="pkg://tracklab.configs", config_name="config")
def main(cfg):
    device = init_environment(cfg)
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

if __name__ == "__main__":
    main()