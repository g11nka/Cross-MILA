"""Reproducible micro-benchmark for audio-text fusion modules.

The benchmark deliberately excludes HuBERT, BERT, and the DataLoader. It
measures the fusion operator under identical synthetic projected features so
that sequence-length scaling can be compared fairly.

For publication-quality memory measurements, invoke one model per process.
The ``--model all`` option is intended only as a quick local check.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.modules import CrossLinearAttention1D, CrossMILAFusion


MODEL_NAMES = (
    "explicit_standard",
    "sdpa",
    "linear",
    "cross_mila",
)

RESULT_FIELDS = (
    "timestamp",
    "status",
    "error",
    "device",
    "gpu",
    "python_version",
    "torch_version",
    "cuda_version",
    "model",
    "mode",
    "precision",
    "tf32",
    "batch_size",
    "audio_len",
    "text_len",
    "dim",
    "heads",
    "head_dim",
    "kernel_size",
    "warmup",
    "repeat",
    "runs",
    "params_m",
    "trainable_params_m",
    "param_memory_mib",
    "estimated_macs_g_per_sample",
    "estimated_flops_g_per_sample",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "throughput_samples_s",
    "peak_allocated_mib",
    "incremental_peak_allocated_mib",
    "peak_reserved_mib",
    "output_shape",
)


class CrossAttentionDirection(nn.Module):
    """One cross-attention direction with explicit or optimized attention."""

    def __init__(self, dim: int, heads: int, backend: str):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if backend not in {"explicit", "sdpa"}:
            raise ValueError(f"Unsupported backend: {backend}")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.backend = backend

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_out_proj = nn.Linear(dim, dim)
        # This post-attention projection and normalization match the original
        # standalone CMT comparison used by this project.
        self.post_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query_x: torch.Tensor, source_x: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.q_proj(query_x))
        key = self._heads(self.k_proj(source_x))
        value = self._heads(self.v_proj(source_x))

        if self.backend == "explicit":
            scores = torch.matmul(query, key.transpose(-2, -1))
            scores = scores * (self.head_dim ** -0.5)
            attention = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention, value)
        else:
            # need_weights=False MultiheadAttention normally dispatches to
            # this optimized SDPA path. It may use flash/memory-efficient
            # kernels, so it is reported separately from explicit attention.
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )

        output = output.transpose(1, 2).contiguous().view(
            query_x.size(0), query_x.size(1), self.dim
        )
        output = self.attn_out_proj(output)
        return self.norm(self.post_proj(output))


class BidirectionalStandardFusion(nn.Module):
    def __init__(self, dim: int, heads: int, backend: str):
        super().__init__()
        self.audio_guided_by_text = CrossAttentionDirection(dim, heads, backend)
        self.text_guided_by_audio = CrossAttentionDirection(dim, heads, backend)

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        audio_branch = self.audio_guided_by_text(audio_feat, text_feat)
        text_branch = self.text_guided_by_audio(text_feat, audio_feat)
        return torch.cat([audio_branch, text_branch], dim=1)


class BidirectionalLinearFusion(nn.Module):
    """Original bidirectional linear cross-attention without Conv or Gate."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.audio_guided_by_text = CrossLinearAttention1D(dim, num_heads=heads)
        self.text_guided_by_audio = CrossLinearAttention1D(dim, num_heads=heads)

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        audio_branch = self.audio_guided_by_text(audio_feat, text_feat)
        text_branch = self.text_guided_by_audio(text_feat, audio_feat)
        return torch.cat([audio_branch, text_branch], dim=1)


def build_module(name: str, dim: int, heads: int, kernel_size: int) -> nn.Module:
    if name == "explicit_standard":
        return BidirectionalStandardFusion(dim, heads, backend="explicit")
    if name == "sdpa":
        return BidirectionalStandardFusion(dim, heads, backend="sdpa")
    if name == "linear":
        return BidirectionalLinearFusion(dim, heads)
    if name == "cross_mila":
        if heads != 8:
            raise ValueError(
                "The current Cross-MILA implementation fixes num_heads=8; "
                "use --heads 8 for a fair comparison."
            )
        return CrossMILAFusion(dim, kernel_size=kernel_size)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(module: nn.Module) -> tuple[int, int, float]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    memory_mib = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    ) / (1024**2)
    return total, trainable, memory_mib


def estimate_forward_macs_per_sample(
    model: str,
    audio_len: int,
    text_len: int,
    dim: int,
    heads: int,
    kernel_size: int,
) -> float:
    """Estimate forward MACs, excluding bias, norm, and elementwise kernels.

    The estimate includes the projection layers and the dominant attention
    matrix products. It is intended for transparent comparison, not as a
    replacement for measured latency.
    """
    length_sum = audio_len + text_len
    head_dim = dim // heads

    if model in {"explicit_standard", "sdpa"}:
        # Q/K/V, attention output and post projection in both directions,
        # followed by QK^T and Attn*V in both directions.
        return 5 * length_sum * dim**2 + 4 * audio_len * text_len * dim

    if model == "linear":
        # Four projections per direction and the two associated linear
        # attention products K^T V and Q(K^T V).
        return 4 * length_sum * dim**2 + 2 * length_sum * dim * head_dim

    if model == "cross_mila":
        # in_proj, Q/K/V/out projections, block out_proj, depthwise Conv1D,
        # and the linear attention products in both directions.
        return (
            7 * length_sum * dim**2
            + 2 * length_sum * dim * head_dim
            + kernel_size * length_sum * dim
        )

    raise ValueError(f"Unknown model: {model}")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def build_step(
    module: nn.Module,
    audio_feat: torch.Tensor,
    text_feat: torch.Tensor,
    mode: str,
    precision: str,
    optimizer: torch.optim.Optimizer | None,
    scaler,
) -> Callable[[], tuple[int, ...]]:
    device = audio_feat.device

    if mode == "inference":
        def inference_step() -> tuple[int, ...]:
            with torch.inference_mode():
                with autocast_context(device, precision):
                    output = module(audio_feat, text_feat)
            return tuple(output.shape)

        return inference_step

    if optimizer is None:
        raise ValueError("Training mode requires an optimizer")

    def train_step() -> tuple[int, ...]:
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            output = module(audio_feat, text_feat)
            loss = output.float().square().mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return tuple(output.shape)

    return train_step


def time_cuda_step(
    step: Callable[[], tuple[int, ...]], repeat: int
) -> tuple[list[float], tuple[int, ...]]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    output_shape: tuple[int, ...] = ()

    for index in range(repeat):
        starts[index].record()
        output_shape = step()
        ends[index].record()

    torch.cuda.synchronize()
    timings = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return timings, output_shape


def time_cpu_step(
    step: Callable[[], tuple[int, ...]], repeat: int
) -> tuple[list[float], tuple[int, ...]]:
    timings: list[float] = []
    output_shape: tuple[int, ...] = ()
    for _ in range(repeat):
        start = time.perf_counter()
        output_shape = step()
        timings.append((time.perf_counter() - start) * 1000)
    return timings, output_shape


def measure_memory(
    step: Callable[[], tuple[int, ...]], device: torch.device
) -> tuple[float, float, float, tuple[int, ...]]:
    if device.type != "cuda":
        output_shape = step()
        return 0.0, 0.0, 0.0, output_shape

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output_shape = step()
    torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    mib = 1024**2
    return (
        peak_allocated / mib,
        (peak_allocated - baseline) / mib,
        peak_reserved / mib,
        output_shape,
    )


def benchmark_one(args, model_name: str, device: torch.device) -> dict[str, object]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    module = build_module(
        model_name,
        args.dim,
        args.heads,
        args.kernel_size,
    ).to(device)
    module.train(args.mode == "train")

    audio_feat = torch.randn(
        args.batch_size,
        args.audio_len,
        args.dim,
        device=device,
    )
    text_feat = torch.randn(
        args.batch_size,
        args.text_len,
        args.dim,
        device=device,
    )

    optimizer = None
    if args.mode == "train":
        optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate)
    scaler = make_grad_scaler(device, args.precision)
    step = build_step(
        module,
        audio_feat,
        text_feat,
        args.mode,
        args.precision,
        optimizer,
        scaler,
    )

    all_timings: list[float] = []
    run_means: list[float] = []
    output_shape: tuple[int, ...] = ()
    timer = time_cuda_step if device.type == "cuda" else time_cpu_step

    for _ in range(args.runs):
        for _ in range(args.warmup):
            output_shape = step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings, output_shape = timer(step, args.repeat)
        all_timings.extend(timings)
        run_means.append(statistics.mean(timings))

    peak_allocated, incremental_peak, peak_reserved, output_shape = measure_memory(
        step,
        device,
    )

    total_params, trainable_params, param_memory = count_parameters(module)
    macs = estimate_forward_macs_per_sample(
        model_name,
        args.audio_len,
        args.text_len,
        args.dim,
        args.heads,
        args.kernel_size,
    )
    latency_mean = statistics.mean(run_means)
    latency_std = statistics.stdev(run_means) if len(run_means) > 1 else 0.0

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "error": "",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "model": model_name,
        "mode": args.mode,
        "precision": args.precision,
        "tf32": not args.disable_tf32,
        "batch_size": args.batch_size,
        "audio_len": args.audio_len,
        "text_len": args.text_len,
        "dim": args.dim,
        "heads": args.heads,
        "head_dim": args.dim // args.heads,
        "kernel_size": args.kernel_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "runs": args.runs,
        "params_m": round(total_params / 1e6, 6),
        "trainable_params_m": round(trainable_params / 1e6, 6),
        "param_memory_mib": round(param_memory, 4),
        "estimated_macs_g_per_sample": round(macs / 1e9, 6),
        "estimated_flops_g_per_sample": round(2 * macs / 1e9, 6),
        "latency_mean_ms": round(latency_mean, 6),
        "latency_std_ms": round(latency_std, 6),
        "latency_p50_ms": round(percentile(all_timings, 0.50), 6),
        "latency_p95_ms": round(percentile(all_timings, 0.95), 6),
        "throughput_samples_s": round(args.batch_size * 1000 / latency_mean, 4),
        "peak_allocated_mib": round(peak_allocated, 4),
        "incremental_peak_allocated_mib": round(incremental_peak, 4),
        "peak_reserved_mib": round(peak_reserved, 4),
        "output_shape": str(output_shape),
    }


def error_row(args, model_name: str, device: torch.device, error: Exception) -> dict[str, object]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "OOM" if "out of memory" in str(error).lower() else "ERROR",
            "error": str(error).replace("\n", " "),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "",
            "model": model_name,
            "mode": args.mode,
            "precision": args.precision,
            "tf32": not args.disable_tf32,
            "batch_size": args.batch_size,
            "audio_len": args.audio_len,
            "text_len": args.text_len,
            "dim": args.dim,
            "heads": args.heads,
            "head_dim": args.dim // args.heads,
            "kernel_size": args.kernel_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "runs": args.runs,
        }
    )
    return row


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_row(row: dict[str, object]) -> None:
    if row["status"] != "OK":
        print(
            f"{row['model']}: {row['status']} | "
            f"B={row['batch_size']} La={row['audio_len']} Lt={row['text_len']} | "
            f"{row['error']}"
        )
        return

    print(
        f"{row['model']}: Params={row['params_m']}M | "
        f"MACs={row['estimated_macs_g_per_sample']}G/sample | "
        f"Mean={row['latency_mean_ms']} ms | P50={row['latency_p50_ms']} ms | "
        f"P95={row['latency_p95_ms']} ms | "
        f"Throughput={row['throughput_samples_s']} samples/s | "
        f"Peak={row['peak_allocated_mib']} MiB | "
        f"IncrementalPeak={row['incremental_peak_allocated_mib']} MiB"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark standard, linear, and Cross-MILA fusion modules."
    )
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--mode", choices=("inference", "train"), default="inference")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--audio_len", type=int, default=256)
    parser.add_argument("--text_len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


def validate_args(args) -> None:
    for name in ("batch_size", "audio_len", "text_len", "dim", "heads", "warmup", "repeat", "runs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.dim % args.heads != 0:
        raise ValueError("--dim must be divisible by --heads")
    if args.precision == "fp16" and not torch.cuda.is_available():
        raise ValueError("FP16 benchmarking requires CUDA")
    if args.precision == "bf16" and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The current CUDA device does not support BF16")


def main() -> None:
    args = parse_args()
    validate_args(args)

    tf32_enabled = not args.disable_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = MODEL_NAMES if args.model == "all" else (args.model,)

    print(
        f"Device={device} | GPU="
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'} | "
        f"PyTorch={torch.__version__} | CUDA={torch.version.cuda}"
    )
    print(
        f"Mode={args.mode} | Precision={args.precision} | B={args.batch_size} | "
        f"La={args.audio_len} | Lt={args.text_len} | d={args.dim} | "
        f"H={args.heads} | warmup={args.warmup} | repeat={args.repeat} | runs={args.runs}"
    )
    if args.model == "all":
        print(
            "Note: --model all is a convenience check. For publication memory "
            "results, invoke one model per process."
        )

    for model_name in models:
        try:
            row = benchmark_one(args, model_name, device)
        except (RuntimeError, ValueError) as error:
            row = error_row(args, model_name, device, error)
        print_row(row)
        if args.output_csv is not None:
            append_csv(args.output_csv, row)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
