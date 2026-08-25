import logging

import torch
import torch.nn as nn

from configs.base import Config
from .modules import (
    CrossMILAFusion,
    build_audio_encoder,
    build_text_encoder,
)


class MemoCMT(nn.Module):
    """HuBERT/BERT model with controlled unimodal and fusion variants."""

    SUPPORTED_VARIANTS = {
        "cross_mila",
        "hubert_only",
        "bert_only",
        "concat",
        "late_fusion",
    }

    def __init__(self, cfg: Config, device: str = "cpu"):
        super().__init__()
        self.model_variant = getattr(
            cfg,
            "model_variant",
            "cross_mila",
        ).lower()
        if self.model_variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported MODEL_VARIANT: {self.model_variant}"
            )

        needs_text = self.model_variant != "hubert_only"
        needs_audio = self.model_variant != "bert_only"
        self.text_encoder = (
            build_text_encoder(cfg.text_encoder_type).to(device)
            if needs_text
            else None
        )
        self.audio_encoder = (
            build_audio_encoder(cfg).to(device)
            if needs_audio
            else None
        )

        self.text_proj = (
            nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
            if needs_text
            else None
        )
        self.audio_proj = (
            nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
            if needs_audio
            else None
        )
        self._configure_encoder_trainability(cfg)

        self.cross_mila = (
            CrossMILAFusion(
                dim=cfg.fusion_dim,
                kernel_size=getattr(
                    cfg,
                    "cross_mila_kernel_size",
                    getattr(cfg, "mlla_kernel_size", 4),
                ),
            )
            if self.model_variant == "cross_mila"
            else None
        )
        self.concat_proj = (
            nn.Linear(cfg.fusion_dim * 2, cfg.fusion_dim)
            if self.model_variant == "concat"
            else None
        )

        self.dropout = nn.Dropout(cfg.dropout)
        self.fusion_head_output_type = (
            cfg.fusion_head_output_type.lower()
        )
        self.linear_layer_output = cfg.linear_layer_output
        self.fusion_dim = cfg.fusion_dim

        if self.model_variant == "late_fusion":
            self.classifier = None
            self.audio_late_layers, audio_dim = self._build_head_layers(cfg)
            self.text_late_layers, text_dim = self._build_head_layers(cfg)
            self.audio_late_classifier = nn.Linear(
                audio_dim,
                cfg.num_classes,
            )
            self.text_late_classifier = nn.Linear(
                text_dim,
                cfg.num_classes,
            )
        else:
            previous_dim = cfg.fusion_dim
            for index, output_dim in enumerate(cfg.linear_layer_output):
                setattr(
                    self,
                    f"linear_{index}",
                    nn.Linear(previous_dim, output_dim),
                )
                previous_dim = output_dim
            self.classifier = nn.Linear(previous_dim, cfg.num_classes)

        logging.info("Model variant: %s", self.model_variant)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load public checkpoints and legacy V2 checkpoints transparently."""
        if any(key.startswith("cross_mlla.") for key in state_dict):
            state_dict = state_dict.copy()
            for key in list(state_dict.keys()):
                if key.startswith("cross_mlla."):
                    new_key = key.replace("cross_mlla.", "cross_mila.", 1)
                    state_dict[new_key] = state_dict.pop(key)
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    @staticmethod
    def _build_head_layers(cfg):
        layers = nn.ModuleList()
        previous_dim = cfg.fusion_dim
        for output_dim in cfg.linear_layer_output:
            layers.append(nn.Linear(previous_dim, output_dim))
            previous_dim = output_dim
        return layers, previous_dim

    def _configure_encoder_trainability(self, cfg):
        text_unfreeze = bool(getattr(cfg, "text_unfreeze", False))
        audio_unfreeze = bool(getattr(cfg, "audio_unfreeze", False))

        if self.text_encoder is not None:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = text_unfreeze
        if self.audio_encoder is not None:
            for parameter in self.audio_encoder.parameters():
                parameter.requires_grad = audio_unfreeze

        text_count = (
            sum(
                parameter.numel()
                for parameter in self.text_encoder.parameters()
                if parameter.requires_grad
            )
            if self.text_encoder is not None
            else 0
        )
        audio_count = (
            sum(
                parameter.numel()
                for parameter in self.audio_encoder.parameters()
                if parameter.requires_grad
            )
            if self.audio_encoder is not None
            else 0
        )
        logging.info(
            "Trainable encoder parameters: text=%d, audio=%d",
            text_count,
            audio_count,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.text_encoder is not None and not any(
                parameter.requires_grad
                for parameter in self.text_encoder.parameters()
            ):
                self.text_encoder.eval()
            if self.audio_encoder is not None and not any(
                parameter.requires_grad
                for parameter in self.audio_encoder.parameters()
            ):
                self.audio_encoder.eval()
        return self

    def _pool_sequence(self, features: torch.Tensor) -> torch.Tensor:
        if self.fusion_head_output_type == "min":
            return features.min(dim=1)[0]
        if self.fusion_head_output_type == "max":
            return features.max(dim=1)[0]
        if self.fusion_head_output_type == "mean":
            return features.mean(dim=1)
        if self.fusion_head_output_type == "cls":
            return features[:, 0, :]
        raise ValueError(
            "Unsupported pooling strategy: "
            f"{self.fusion_head_output_type}"
        )

    def _pool_fusion(
        self,
        fusion_features: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        if self.fusion_head_output_type == "cls":
            return fusion_features[:, audio_length, :]
        return self._pool_sequence(fusion_features)

    @staticmethod
    def _pool_text(text_features: torch.Tensor) -> torch.Tensor:
        # BERT's first position is the standard CLS sequence summary.
        return text_features[:, 0, :]

    def _extract_text_features(self, input_text, text_mask=None):
        if self.text_encoder is None:
            raise RuntimeError("The text encoder is disabled for this variant.")
        kwargs = {"input_ids": input_text}
        if text_mask is not None:
            kwargs["attention_mask"] = text_mask.long()
        output = self.text_encoder(**kwargs)
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, tuple):
            return output[0]
        return output

    def _extract_audio_features(self, input_audio, audio_lengths=None):
        if self.audio_encoder is None:
            raise RuntimeError("The audio encoder is disabled for this variant.")
        try:
            output = self.audio_encoder(
                input_audio,
                lengths=audio_lengths,
            )
        except TypeError:
            try:
                output = self.audio_encoder(input_audio, audio_lengths)
            except TypeError:
                output = self.audio_encoder(input_audio)

        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, tuple):
            return output[0]
        if isinstance(output, dict) and "encoder_out" in output:
            return output["encoder_out"]
        return output

    def _classify_common(self, pooled_features):
        output = self.dropout(pooled_features)
        for index in range(len(self.linear_layer_output)):
            output = getattr(self, f"linear_{index}")(output)
            output = nn.functional.leaky_relu(output)
        logits = self.classifier(self.dropout(output))
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits

    def _classify_late_branch(self, features, layers, classifier):
        output = self.dropout(features)
        for layer in layers:
            output = nn.functional.leaky_relu(layer(output))
        return classifier(self.dropout(output))

    def _classify_projected(self, text_projected, audio_projected):
        if text_projected.dim() == 2:
            text_projected = text_projected.unsqueeze(1)
        if audio_projected.dim() == 2:
            audio_projected = audio_projected.unsqueeze(1)

        if self.model_variant == "hubert_only":
            return self._classify_common(
                self._pool_sequence(audio_projected)
            )

        if self.model_variant == "bert_only":
            return self._classify_common(
                self._pool_text(text_projected)
            )

        audio_global = self._pool_sequence(audio_projected)
        text_global = self._pool_text(text_projected)

        if self.model_variant == "concat":
            joint = torch.cat([audio_global, text_global], dim=-1)
            joint = nn.functional.leaky_relu(self.concat_proj(joint))
            return self._classify_common(joint)

        if self.model_variant == "late_fusion":
            audio_logits = self._classify_late_branch(
                audio_global,
                self.audio_late_layers,
                self.audio_late_classifier,
            )
            text_logits = self._classify_late_branch(
                text_global,
                self.text_late_layers,
                self.text_late_classifier,
            )
            return 0.5 * (audio_logits + text_logits)

        fusion_features = self.cross_mila(
            audio_projected,
            text_projected,
        )
        fusion_features = self.dropout(fusion_features)
        pooled_features = self._pool_fusion(
            fusion_features,
            audio_length=audio_projected.size(1),
        )
        return self._classify_common(pooled_features)

    def forward(
        self,
        input_text,
        input_audio,
        text_mask=None,
        audio_mask=None,
        audio_lengths=None,
        session_grl_scale=1.0,
        domain_grl_scale=None,
        output_attentions=False,
    ):
        del (
            audio_mask,
            session_grl_scale,
            domain_grl_scale,
            output_attentions,
        )

        text_projected = None
        audio_projected = None
        if self.text_encoder is not None:
            text_features = self._extract_text_features(
                input_text,
                text_mask=text_mask,
            )
            text_projected = self.text_proj(text_features)
        if self.audio_encoder is not None:
            audio_features = self._extract_audio_features(
                input_audio,
                audio_lengths=audio_lengths,
            )
            audio_projected = self.audio_proj(audio_features)

        if text_projected is None:
            text_projected = audio_projected.new_zeros(
                audio_projected.size(0),
                1,
                self.fusion_dim,
            )
        if audio_projected is None:
            audio_projected = text_projected.new_zeros(
                text_projected.size(0),
                1,
                self.fusion_dim,
            )

        logits = self._classify_projected(
            text_projected,
            audio_projected,
        )
        return logits, text_projected, audio_projected
