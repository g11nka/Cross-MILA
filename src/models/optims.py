from torch import nn, optim
from configs.base import Config
from typing import Union, List, Dict
import logging


def get_parameter_groups(cfg: Config, network: nn.Module) -> List[Dict]:
    """Separate pretrained encoders from newly initialized task modules."""
    base_params = []
    fusion_params = []

    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue

        if "audio_encoder" in name or "text_encoder" in name:
            base_params.append(param)
        else:
            fusion_params.append(param)

    lr_base = getattr(cfg, 'learning_rate_base', cfg.learning_rate)
    lr_fusion = getattr(cfg, 'learning_rate_fusion', cfg.learning_rate)

    optimizer_grouped_parameters = []

    if len(base_params) > 0:
        optimizer_grouped_parameters.append({"params": base_params, "lr": lr_base})

    if len(fusion_params) > 0:
        optimizer_grouped_parameters.append({"params": fusion_params, "lr": lr_fusion})

    logging.info(
        "Optimizer groups: pretrained encoders=%d tensors (lr=%g), "
        "task modules=%d tensors (lr=%g)",
        len(base_params),
        lr_base,
        len(fusion_params),
        lr_fusion,
    )

    return optimizer_grouped_parameters


def adamw(cfg: Config, network: nn.Module) -> optim.AdamW:
    param_groups = get_parameter_groups(cfg, network)
    return optim.AdamW(
        params=param_groups,
        betas=(cfg.adam_beta_1, cfg.adam_beta_2),
        eps=cfg.adam_eps,
        weight_decay=getattr(cfg, "adam_weight_decay", 0.01),
    )


def adam(cfg: Config, network: nn.Module) -> optim.Adam:
    param_groups = get_parameter_groups(cfg, network)
    return optim.Adam(
        params=param_groups,
        betas=(cfg.adam_beta_1, cfg.adam_beta_2),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_weight_decay,
    )


def sgd(cfg: Config, network: nn.Module) -> optim.SGD:
    param_groups = get_parameter_groups(cfg, network)
    return optim.SGD(
        params=param_groups,
        momentum=cfg.momemtum,
        weight_decay=cfg.sdg_weight_decay,
    )


def get_optim(cfg: Config, network: nn.Module) -> Union[optim.SGD, optim.Adam]:
    optim_fn = {
        "SGD": sgd,
        "Adam": adam,
        "AdamW": adamw,
    }
    assert cfg.optimizer_type in optim_fn.keys(), (
            "Invalid optimizer_type. The valid optim is ["
            + " ".join(list(optim_fn.keys()))
            + "]"
    )

    return optim_fn[cfg.optimizer_type](cfg, network)
