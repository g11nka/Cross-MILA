import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(env_name: str, relative_default: str) -> str:
    """Resolve a path from an environment variable or the repository root."""
    configured = os.environ.get(env_name)
    path = Path(configured).expanduser() if configured else PROJECT_ROOT / relative_default
    return str(path.resolve())


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: 1/0, true/false, yes/no, or on/off."
    )


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self, cfg):
        pass


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info("%s: %s", key, value)

    def save(self, cfg):
        message = "\n"
        for key, value in sorted(vars(cfg).items()):
            message += f"{str(key):>30}: {str(value):<40}\n"

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        output_path = os.path.join(cfg.checkpoint_dir, "cfg.log")
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(message)
            output_file.write("\n")
        logging.info(message)

    def load(self, cfg_path: str):
        def decode_value(value: str):
            value = value.strip()
            if "." in value and value.replace(".", "").isdigit():
                return float(value)
            if value.isdigit():
                return int(value)
            if value == "True":
                return True
            if value == "False":
                return False
            if value == "None":
                return None
            if (
                value.startswith("'")
                and value.endswith("'")
                or value.startswith('"')
                and value.endswith('"')
            ):
                return value[1:-1]
            return value

        with open(cfg_path, "r", encoding="utf-8") as cfg_file:
            rows = list(filter(None, cfg_file.read().split("\n")))

        data = {}
        for row in rows:
            key, value = row.split(":", 1)
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = [decode_value(x) for x in value[1:-1].split(",")]
            else:
                value = decode_value(value)
            data[key.strip()] = value

        # Older checkpoints used MMD terminology for the same B=1 paired
        # RBF objective. Keep those cfg.log files loadable without exposing
        # the obsolete names in new runs.
        if "alpha_pra" not in data and "alpha_mmd" in data:
            data["alpha_pra"] = data["alpha_mmd"]
        if "pra_sigma" not in data and "mmd_sigma" in data:
            data["pra_sigma"] = data["mmd_sigma"]
        if "cross_mila_kernel_size" not in data and "mlla_kernel_size" in data:
            data["cross_mila_kernel_size"] = data["mlla_kernel_size"]
        data.pop("alpha_mmd", None)
        data.pop("mmd_sigma", None)
        data.pop("mlla_kernel_size", None)

        for key, value in data.items():
            setattr(self, key, value)


class Config(BaseConfig):
    """HuBERT-based Cross-MILA v2 configuration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_args(**kwargs)

    def set_args(self, **kwargs):
        self.validate_paths: bool = bool(kwargs.get("validate_paths", True))
        # Stage 1 freezes both encoders. Stage 2 loads the Stage-1 best
        # checkpoint and fully fine-tunes BERT and HuBERT.
        self.dataset_name: str = os.environ.get(
            "DATASET_NAME",
            kwargs.get("dataset_name", "ESD"),
        ).strip().upper()
        if self.dataset_name not in {"IEMOCAP", "ESD", "MELD"}:
            raise ValueError(
                "DATASET_NAME must be IEMOCAP, ESD, or MELD."
            )

        # IEMOCAP and ESD use five-fold evaluation. MELD keeps its official
        # train/dev/test split and therefore has no fold identifier.
        if self.dataset_name == "MELD":
            self.fold: Union[int, None] = None
        else:
            fold_env_name = (
                "IEMOCAP_FOLD"
                if self.dataset_name == "IEMOCAP"
                else "ESD_FOLD"
            )
            self.fold = int(os.environ.get(fold_env_name, "1"))
            if self.fold not in range(1, 6):
                raise ValueError(
                    f"{fold_env_name} must be an integer from 1 to 5."
                )

        # Retained for compatibility with helper scripts that still read it.
        self.iemocap_fold: Union[int, None] = self.fold

        self.training_stage: str = os.environ.get(
            "TRAINING_STAGE",
            kwargs.get("training_stage", "stage1"),
        ).strip().lower()
        self.stage2_finetune_path: str = os.environ.get(
            "STAGE2_FINETUNE_PATH",
            kwargs.get("stage2_finetune_path", ""),
        ).strip()
        if self.training_stage not in {"stage1", "stage2"}:
            raise ValueError(
                "training_stage must be either 'stage1' or 'stage2'."
            )

        self.model_variant: str = os.environ.get(
            "MODEL_VARIANT",
            kwargs.get("model_variant", "cross_mila"),
        ).strip().lower()
        valid_model_variants = {
            "cross_mila",
            "hubert_only",
            "bert_only",
            "concat",
            "late_fusion",
        }
        if self.model_variant not in valid_model_variants:
            raise ValueError(
                "MODEL_VARIANT must be one of: "
                + ", ".join(sorted(valid_model_variants))
            )

        self.trainer: str = "Trainer"
        self.num_epochs: int = int(os.environ.get("NUM_EPOCHS", "100"))
        self.checkpoint_dir: str = _configured_path(
            "CROSS_MILA_OUTPUT_ROOT",
            "outputs/checkpoints",
        )
        variant_names = {
            "cross_mila": "CrossMILA",
            "hubert_only": "HuBERTOnly",
            "bert_only": "BERTOnly",
            "concat": "SimpleConcat",
            "late_fusion": "LateFusion",
        }
        variant_name = variant_names[self.model_variant]
        if self.dataset_name == "MELD":
            if self.model_variant == "cross_mila":
                self.name = (
                    "MELD_V2_HuBERT_CrossMILA_OfficialSplit_"
                    f"{self.training_stage.capitalize()}"
                )
            else:
                self.name = (
                    f"MELD_V2_{variant_name}_CE_OfficialSplit_"
                    f"{self.training_stage.capitalize()}"
                )
        else:
            if self.model_variant == "cross_mila":
                self.name = (
                    f"{self.dataset_name}_V2_HuBERT_CrossMILA_5Fold_"
                    f"Fold{self.fold}_{self.training_stage.capitalize()}"
                )
            else:
                self.name = (
                    f"{self.dataset_name}_V2_{variant_name}_CE_5Fold_"
                    f"Fold{self.fold}_{self.training_stage.capitalize()}"
                )
        # Public defaults retain only the best validation weights. Resumable
        # optimizer/scheduler/AMP states can be enabled explicitly when a
        # long run is likely to be interrupted.
        self.save_all_states: bool = _env_bool("SAVE_ALL_STATES", False)
        self.save_best_val: bool = True
        self.max_to_keep: int = 1
        self.save_freq: int = 10**12
        self.save_periodic_checkpoints: bool = False
        self.save_each_epoch: bool = _env_bool("SAVE_EACH_EPOCH", False)
        self.checkpoint_metric: str = "acc"
        self.early_stopping_patience: int = 0
        self.batch_size: int = int(os.environ.get("BATCH_SIZE", "1"))
        self.eval_batch_size: int = int(os.environ.get("EVAL_BATCH_SIZE", "1"))
        self.num_workers: int = int(os.environ.get("NUM_WORKERS", "4"))
        self.seed: int = int(os.environ.get("SEED", "0"))
        self.use_amp: bool = True
        self.warmup_epochs: int = 5

        self.learning_rate: float = 5e-5
        self.learning_rate_base: float = 5e-6
        self.learning_rate_fusion: float = 1e-4
        self.learning_rate_step_size: int = 30
        self.learning_rate_gamma: float = 0.1
        self.learning_rate_eta_min: float = 1e-8
        self.optimizer_type: str = "AdamW"
        self.adam_beta_1: float = 0.9
        self.adam_beta_2: float = 0.999
        self.adam_eps: float = 1e-8
        # The original V2 optimizer hard-coded 0.01 even though old cfg.log
        # files displayed zero. Keep the effective training behavior here.
        self.adam_weight_decay: float = 0.01
        self.momemtum: float = 0.99
        self.sdg_weight_decay: float = 1e-6

        self.resume_path: str = os.environ.get("RESUME_PATH", "").strip()
        self.resume: bool = _env_bool("RESUME", bool(self.resume_path))
        if self.validate_paths and self.resume and not self.resume_path:
            raise ValueError("Set RESUME_PATH when RESUME is enabled.")
        if (
            self.validate_paths
            and self.resume
            and not os.path.isfile(self.resume_path)
        ):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {self.resume_path}"
            )
        self.finetune_base_path: str = (
            ""
            if self.training_stage == "stage1"
            else self.stage2_finetune_path
        )
        if (
            self.validate_paths
            and self.training_stage == "stage2"
            and not self.finetune_base_path
        ):
            raise ValueError(
                "Set STAGE2_FINETUNE_PATH to this fold's Stage-1 best_acc "
                "checkpoint before running Stage 2."
            )
        if (
            self.validate_paths
            and self.training_stage == "stage2"
            and not os.path.isfile(self.finetune_base_path)
        ):
            raise FileNotFoundError(
                "Stage-1 best_acc checkpoint not found: "
                f"{self.finetune_base_path}"
            )
        self.cfg_path: Union[str, None] = None
        self.use_lora: bool = False
        self.text_unfreeze: bool = (
            self.training_stage == "stage2"
            and self.model_variant != "hubert_only"
        )
        self.audio_unfreeze: bool = (
            self.training_stage == "stage2"
            and self.model_variant != "bert_only"
        )

        self.data_name: str = self.dataset_name
        if self.data_name == "IEMOCAP":
            self.split_protocol: str = "5-fold leave-one-session-out"
            self.test_session: str = f"Session{self.fold}"
            self.iemocap_5fold_root: str = os.environ.get(
                "IEMOCAP_5FOLD_ROOT",
                _configured_path(
                    "IEMOCAP_5FOLD_ROOT",
                    "data/processed/IEMOCAP",
                ),
            )
            self.data_root: str = os.path.join(
                self.iemocap_5fold_root,
                f"fold_{self.fold}",
            )
            self.num_classes: int = 4
        elif self.data_name == "ESD":
            self.split_protocol: str = (
                "5-fold speaker-independent evaluation on the English "
                "subset"
            )
            self.test_session: str = f"SpeakerPair{self.fold}"
            self.esd_5fold_root: str = os.environ.get(
                "ESD_5FOLD_ROOT",
                _configured_path(
                    "ESD_5FOLD_ROOT",
                    "data/processed/ESD",
                ),
            )
            self.data_root: str = os.path.join(
                self.esd_5fold_root,
                f"fold_{self.fold}",
            )
            self.num_classes: int = 5
        else:
            self.split_protocol: str = "official train/dev/test split"
            self.test_session: str = "official test set"
            self.meld_root: str = os.environ.get(
                "MELD_ROOT",
                _configured_path("MELD_ROOT", "data/processed/MELD"),
            )
            self.data_root: str = self.meld_root
            self.num_classes: int = 7

        self.data_valid: str = "val.pkl"
        required_split_files = (
            "train.pkl",
            "val.pkl",
            "test.pkl",
            "classes.json",
        )
        missing_split_files = [
            filename
            for filename in required_split_files
            if not os.path.isfile(os.path.join(self.data_root, filename))
        ]
        if self.validate_paths and missing_split_files:
            raise FileNotFoundError(
                "Missing {} split files in {}: {}".format(
                    self.data_name,
                    self.data_root,
                    ", ".join(missing_split_files),
                )
            )
        self.text_max_length: int = 297
        self.audio_max_length: int = 546220

        self.model_type: str = "MemoCMT"
        self.text_encoder_type: str = "bert"
        self.text_encoder_dim: int = 768
        self.audio_encoder_type: str = "hubert_base"
        self.audio_encoder_dim: int = 768
        self.fusion_dim: int = 768
        self.num_attention_head: int = 8
        self.cross_mila_kernel_size: int = 4
        self.dropout: float = 0.05
        default_pooling = "mean" if self.data_name == "ESD" else "min"
        self.fusion_head_output_type: str = os.environ.get(
            "POOLING_STRATEGY",
            default_pooling,
        ).strip().lower()
        if self.fusion_head_output_type not in {"cls", "min", "max", "mean"}:
            raise ValueError(
                "POOLING_STRATEGY must be cls, min, max, or mean."
            )
        self.linear_layer_output: List = [128]
        self.linear_layer_last_dim: int = 64

        self.loss_type: str = "CrossEntropyLoss"
        self.alpha_pra: float = float(os.environ.get("ALPHA_PRA", "0.1"))
        self.beta_dec: float = float(os.environ.get("BETA_DEC", "0.05"))
        self.pra_sigma: float = float(os.environ.get("PRA_SIGMA", "1.0"))
        if self.model_variant != "cross_mila":
            # Unimodal and simple-fusion baselines use CE only. This keeps
            # the comparison independent of the auxiliary objectives.
            self.alpha_pra = 0.0
            self.beta_dec = 0.0
        self.class_weights = None
        self.focal_gamma: float = 2.0

        # These generic loader switches stay disabled in V2.
        self.use_masked_batching: bool = False
        self.use_balanced_batches: bool = False
        self.use_precomputed_features: bool = False
        self.train_without_validation: bool = False

        for key, value in kwargs.items():
            setattr(self, key, value)
