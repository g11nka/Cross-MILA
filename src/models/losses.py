import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as CELoss

from configs.base import Config


class CrossEntropyLoss(CELoss):
    """Cross entropy with paired RBF alignment and decorrelation."""

    def __init__(self, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        self.alpha_pra = getattr(
            cfg,
            "alpha_pra",
            getattr(cfg, "alpha_mmd", 0.1),
        )
        self.beta_dec = getattr(cfg, "beta_dec", 0.05)
        self.pra_sigma = getattr(
            cfg,
            "pra_sigma",
            getattr(cfg, "mmd_sigma", 1.0),
        )

    def compute_paired_rbf_loss(self, x, y):
        """Penalize each paired audio-text distance through an RBF kernel."""
        squared_distance = (x - y).pow(2).sum(dim=-1)
        similarity = torch.exp(
            -squared_distance / (2.0 * self.pra_sigma**2)
        )
        return (2.0 - 2.0 * similarity).mean()

    @staticmethod
    def compute_decoupling_loss(x, y):
        return F.cosine_similarity(x, y, dim=-1).abs().mean()

    def forward(self, inputs, target):
        if not isinstance(inputs, (tuple, list)):
            return super().forward(inputs, target.view(-1))

        logits = inputs[0]
        ce_loss = super().forward(logits, target.view(-1))
        if len(inputs) < 3:
            return ce_loss

        text_features = inputs[1]
        audio_features = inputs[2]
        if text_features.dim() == 3:
            text_features = text_features.mean(dim=1)
        if audio_features.dim() == 3:
            audio_features = audio_features.mean(dim=1)

        paired_rbf_loss = (
            self.compute_paired_rbf_loss(text_features, audio_features)
            if self.alpha_pra > 0
            else logits.new_zeros(())
        )
        dec_loss = (
            self.compute_decoupling_loss(text_features, audio_features)
            if self.beta_dec > 0
            else logits.new_zeros(())
        )
        return (
            ce_loss
            + self.alpha_pra * paired_rbf_loss
            + self.beta_dec * dec_loss
        )
