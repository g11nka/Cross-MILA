import logging
import math
from typing import Dict

import torch
from torch import Tensor

from configs.base import Config
from models.networks import MemoCMT
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: MemoCMT,
        criterion: torch.nn.Module = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network.to(self.device)
        self.internal_epoch_counter = 1
        self.start_epoch = 1

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.use_amp = getattr(cfg, "use_amp", True)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and torch.cuda.is_available(),
        )
        if self.use_amp:
            logging.info("AMP is enabled.")

    def load_all_states(self, resume_path: str):
        logging.info("Restoring full training state from %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=self.device)
        state_dict = checkpoint.get(
            "state_dict_network",
            checkpoint.get(
                "state_dict",
                checkpoint.get("model_state_dict", checkpoint),
            ),
        )
        self.network.load_state_dict(state_dict, strict=False)

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        if "epoch" in checkpoint:
            self.start_epoch = int(checkpoint["epoch"])
            self.internal_epoch_counter = self.start_epoch + 1
            if hasattr(self, "current_epoch"):
                self.current_epoch = self.start_epoch

        optimizer_state = checkpoint.get(
            "state_optimizer",
            checkpoint.get(
                "optimizer",
                checkpoint.get("optimizer_state_dict"),
            ),
        )
        if optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = checkpoint.get(
            "state_lr_scheduler",
            checkpoint.get(
                "lr_scheduler",
                checkpoint.get("scheduler_state_dict"),
            ),
        )
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get(
            "scaler",
            checkpoint.get("scaler_state_dict"),
        )
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

    def train_epoch(self, *args, **kwargs):
        result = super().train_epoch(*args, **kwargs)
        self.internal_epoch_counter += 1
        return result

    def _prepare_batch(self, batch):
        if not isinstance(batch, dict):
            input_text, input_audio, label = batch
            return {
                "precomputed_features": False,
                "input_text": input_text.to(self.device),
                "input_audio": input_audio.to(self.device),
                "label": label.to(self.device).view(-1),
                "text_mask": None,
                "audio_mask": None,
                "audio_lengths": None,
                "domain_ids": None,
                "session_ids": None,
            }

        prepared = {
            "precomputed_features": bool(
                batch.get("precomputed_features", False)
            ),
            "label": batch["label"].to(self.device).view(-1),
            "domain_ids": batch.get("domain_id", batch.get("session_id")),
            "session_ids": batch.get("session_id"),
        }
        if prepared["domain_ids"] is not None:
            prepared["domain_ids"] = prepared["domain_ids"].to(self.device)
        if prepared["session_ids"] is not None:
            prepared["session_ids"] = prepared["session_ids"].to(self.device)

        if prepared["precomputed_features"]:
            prepared.update(
                text_features=batch["text_features"].to(self.device),
                audio_features=batch["audio_features"].to(self.device),
                text_mask=batch["text_mask"].to(self.device),
                audio_feature_mask=batch["audio_feature_mask"].to(
                    self.device
                ),
            )
        else:
            prepared.update(
                input_text=batch["input_text"].to(self.device),
                input_audio=batch["input_audio"].to(self.device),
                text_mask=batch["text_mask"].to(self.device),
                audio_mask=batch["audio_mask"].to(self.device),
                audio_lengths=batch["audio_lengths"].to(self.device),
            )
        return prepared

    def _forward_batch(self, batch, grl_scale):
        if batch["precomputed_features"]:
            return self.network.forward_precomputed(
                batch["text_features"],
                batch["audio_features"],
                text_mask=batch["text_mask"],
                audio_feature_mask=batch["audio_feature_mask"],
                domain_grl_scale=grl_scale,
            )
        return self.network(
            batch["input_text"],
            batch["input_audio"],
            text_mask=batch["text_mask"],
            audio_mask=batch["audio_mask"],
            audio_lengths=batch["audio_lengths"],
            domain_grl_scale=grl_scale,
        )

    def _compute_loss(
        self,
        output,
        label,
        domain_ids,
        session_ids,
    ):
        if domain_ids is None and session_ids is None:
            return self.criterion(output, label)
        try:
            return self.criterion(
                output,
                label,
                domain_ids=domain_ids,
                session_ids=session_ids,
            )
        except TypeError:
            try:
                return self.criterion(
                    output,
                    label,
                    session_ids=session_ids,
                )
            except TypeError:
                return self.criterion(output, label)

    def _grl_scale(self):
        warmup = max(
            1,
            getattr(
                self.cfg,
                "domain_grl_warmup_epochs",
                getattr(self.cfg, "session_grl_warmup_epochs", 10),
            ),
        )
        progress = min(
            1.0,
            max(0.0, (self.internal_epoch_counter - 1) / warmup),
        )
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    def _autocast(self):
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        return torch.amp.autocast(
            device_type="cuda",
            enabled=self.use_amp and torch.cuda.is_available(),
            dtype=dtype,
        )

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()
        prepared = self._prepare_batch(batch)
        label = prepared["label"]

        with self._autocast():
            output = self._forward_batch(prepared, self._grl_scale())
            logits = output[0] if isinstance(output, (tuple, list)) else output
            label = label.view(-1)
            loss = self._compute_loss(
                output,
                label,
                prepared["domain_ids"],
                prepared["session_ids"],
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        preds = torch.argmax(logits, dim=1)
        result = {
            "loss": loss.item(),
            "acc": torch.mean((preds == label).float()).item(),
        }
        for name, value in getattr(
            self.criterion,
            "last_components",
            {},
        ).items():
            result[f"loss_{name}"] = float(value.detach().cpu())
        return result

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        prepared = self._prepare_batch(batch)
        label = prepared["label"]

        with torch.no_grad(), self._autocast():
            output = self._forward_batch(prepared, 0.0)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            loss = self._compute_loss(
                output,
                label,
                prepared["domain_ids"],
                prepared["session_ids"],
            )

        preds = torch.argmax(logits, dim=1)
        accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
            "preds": preds.detach().cpu().tolist(),
            "labels": label.detach().cpu().tolist(),
        }
