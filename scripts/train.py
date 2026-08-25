import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

import os
import sys

current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(scripts_dir)
src_dir = os.path.join(project_root, "src")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import argparse
import datetime
import random

import numpy as np
import torch
from torch import optim

import trainer as Trainer
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import losses, networks, optims
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    """Seed the random generators used by the training entry point."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cfg: Config):
    seed_everything(cfg.seed)
    logging.info("Random seed: %d", cfg.seed)
    logging.info("Initializing model...")
    try:
        network = getattr(networks, cfg.model_type)(cfg)
        network.to(device)

        if getattr(cfg, "resume", False):
            if cfg.resume_path and os.path.exists(cfg.resume_path):
                logging.info("Resume checkpoint found: %s", cfg.resume_path)
            else:
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {cfg.resume_path}"
                )
        else:
            if getattr(cfg, "finetune_base_path", ""):
                if os.path.exists(cfg.finetune_base_path):
                    logging.info(
                        "Loading Stage-1 initialization from %s",
                        cfg.finetune_base_path,
                    )
                    checkpoint = torch.load(
                        cfg.finetune_base_path,
                        map_location=device,
                    )
                    state_dict = checkpoint.get(
                        "state_dict_network",
                        checkpoint.get(
                            "state_dict",
                            checkpoint.get("model_state_dict", checkpoint),
                        ),
                    )
                    msg = network.load_state_dict(state_dict, strict=False)
                    logging.info("Stage-1 initialization loaded: %s", msg)
                else:
                    raise FileNotFoundError(
                        "Stage-1 initialization not found: "
                        f"{cfg.finetune_base_path}"
                    )
            else:
                logging.info("Starting from the configured pretrained encoders.")

    except AttributeError:
        if not hasattr(networks, cfg.model_type):
            raise NotImplementedError(
                "Model {} is not implemented".format(cfg.model_type)
            )
        raise

    logging.info("Initializing checkpoint directory and dataset...")
    cfg.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(cfg.checkpoint_dir),
        cfg.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    cfg.save(cfg)

    try:
        criterion = getattr(losses, cfg.loss_type)(cfg)
        criterion.to(device)
        logging.info("Initialized loss: %s", cfg.loss_type)
    except AttributeError:
        raise NotImplementedError("Loss {} is not implemented".format(cfg.loss_type))

    try:
        trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            network=network,
            criterion=criterion,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError("Trainer {} is not implemented".format(cfg.trainer))

    train_ds, test_ds = build_train_test_dataset(cfg)
    logging.info("Initializing trainer...")
    logging.info("Start training...")

    optimizer = optims.get_optim(cfg, network)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
        eta_min=getattr(cfg, "learning_rate_eta_min", 1e-8),
    )
    logging.info("Initialized cosine annealing scheduler.")

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        best_metric=getattr(cfg, "checkpoint_metric", "acc"),
        early_stopping_patience=getattr(
            cfg,
            "early_stopping_patience",
            0,
        ),
        save_periodic=getattr(
            cfg,
            "save_periodic_checkpoints",
            True,
        ),
        save_each_epoch=getattr(cfg, "save_each_epoch", False),
        save_all_states=cfg.save_all_states,
    )

    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)

    if getattr(cfg, "resume", False):
        logging.info("Restoring model, optimizer, scheduler, and AMP state.")
        trainer.load_all_states(cfg.resume_path)

    trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback])
    if getattr(cfg, "train_without_validation", False):
        final_path = trainer.save_all_states(
            weight_dir,
            cfg.num_epochs,
            trainer.global_step,
        )
        logging.info("Saved final outer-fold model to %s", final_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default=os.path.join(src_dir, "configs", "base.py"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = get_options(args.config)

    if getattr(cfg, "resume", False) and getattr(cfg, "cfg_path", None) is not None:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.cfg_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    main(cfg)
