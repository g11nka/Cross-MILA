import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import (
    BertConfig,
    BertModel,
)

from configs.base import Config

def build_bert_encoder() -> nn.Module:
    config = BertConfig.from_pretrained(
        "bert-base-uncased", output_hidden_states=True, output_attentions=True
    )
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


class HuBertBase(nn.Module):
    def __init__(self, **kwargs):
        super(HuBertBase, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_hubert_base_encoder(cfg: Config) -> nn.Module:
    return HuBertBase()


def build_audio_encoder(cfg: Config) -> nn.Module:
    encoder_type = cfg.audio_encoder_type
    encoders = {
        "hubert_base": build_hubert_base_encoder,
    }
    if encoder_type not in encoders:
        raise ValueError(f"Invalid audio encoder type: {encoder_type}")
    return encoders[encoder_type](cfg)


def build_text_encoder(encoder_type: str = "bert") -> nn.Module:
    encoders = {
        "bert": build_bert_encoder,
    }
    if encoder_type not in encoders:
        raise ValueError(f"Invalid text encoder type: {encoder_type}")
    return encoders[encoder_type]()

class CrossLinearAttention1D(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_x, kv_x, q_mask=None, kv_mask=None):
        B, N, C = q_x.shape
        _, M, _ = kv_x.shape

        Q = self.q_proj(q_x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_x).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_x).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        Q = F.relu(Q) + 1e-6
        K = F.relu(K) + 1e-6

        if kv_mask is not None:
            source_mask = kv_mask[:, None, :, None].to(dtype=K.dtype)
            K = K * source_mask
            V = V * source_mask

        K_sum = K.sum(dim=2)
        den = torch.einsum("bhnd,bhd->bhn", Q, K_sum) + 1e-6

        KV = torch.einsum("bhmd,bhme->bhde", K, V)
        num = torch.einsum("bhnd,bhde->bhne", Q, KV)

        out = num / den.unsqueeze(-1)
        out = out.transpose(1, 2).reshape(B, N, C)
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).to(dtype=out.dtype)

        return self.out_proj(out)


class CrossMILABlock1D(nn.Module):
    """Original Cross-MILA v2 interaction block.

    The gate is the raw channel produced by ``in_proj`` and the residual has
    unit scale, matching the model used before the V3/V4 experiments.
    """

    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv1d = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.act = nn.SiLU()
        self.cross_attn = CrossLinearAttention1D(dim)
        self.out_proj = nn.Linear(dim, dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, cross_kv, x_mask=None, cross_mask=None):
        sequence_length = x.size(1)
        target_mask = None
        if x_mask is not None:
            target_mask = x_mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * target_mask

        local_channel, gate_channel = self.in_proj(x).chunk(2, dim=-1)
        local_channel = self.conv1d(
            local_channel.transpose(1, 2)
        )[:, :, :sequence_length].transpose(1, 2)
        query = self.act(local_channel)
        if target_mask is not None:
            query = query * target_mask
            gate_channel = gate_channel * target_mask

        attention = self.cross_attn(
            query,
            cross_kv,
            q_mask=x_mask,
            kv_mask=cross_mask,
        )
        output = x + self.out_proj(attention * gate_channel)
        if target_mask is not None:
            output = output * target_mask
        return output


class CrossMILAFusion(nn.Module):
    """Bidirectional Cross-MILA fusion used by the original 2.0 model."""

    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.audio_guided_by_text = CrossMILABlock1D(
            dim,
            kernel_size=kernel_size,
        )
        self.text_guided_by_audio = CrossMILABlock1D(
            dim,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        audio_feat,
        text_feat,
        audio_mask=None,
        text_mask=None,
        return_branches=False,
    ):
        audio_branch = self.audio_guided_by_text(
            audio_feat,
            text_feat,
            x_mask=audio_mask,
            cross_mask=text_mask,
        )
        text_branch = self.text_guided_by_audio(
            text_feat,
            audio_feat,
            x_mask=text_mask,
            cross_mask=audio_mask,
        )
        if return_branches:
            return audio_branch, text_branch
        return torch.cat([audio_branch, text_branch], dim=1)


# Legacy aliases keep existing checkpoints and older benchmark scripts usable.
MLLABlock1DV2 = CrossMILABlock1D
Cross_MLLA_FusionV2 = CrossMILAFusion
Cross_MLLA_Fusion = CrossMILAFusion
