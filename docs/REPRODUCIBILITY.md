# Reproducibility checklist

## Protocol

- IEMOCAP: five outer folds, each holding out one complete session for test.
- ESD: English subset, five outer folds, each holding out one speaker pair.
- Development data: the remaining four groups; validation is a stratified
  utterance subset generated with seed 0 (`--mode paper_like`).
- Node selection: highest validation accuracy within each fold.
- Reporting: arithmetic mean and sample standard deviation over five held-out
  folds. Fold variation is not presented as random-seed variation.
- MELD: official train/dev/test split.

Every public run should preserve its `cfg.log`, split `manifest.json`, best
checkpoint name, Git commit hash, GPU model, PyTorch/CUDA versions, and command.

## Two-stage training

Stage 1 freezes HuBERT and BERT and learns projections, Cross-MILA, and the
classifier. Stage 2 initializes from the same fold's Stage-1 `best_acc`
checkpoint and enables encoder fine-tuning with a smaller encoder learning
rate. A checkpoint from another fold must never be reused.

## Controlled objectives

Use environment variables to avoid editing the formal configuration:

| Objective | `ALPHA_PRA` | `BETA_DEC` |
|---|---:|---:|
| CE | 0 | 0 |
| CE + PRA | 0.1 | 0 |
| CE + Dec | 0 | 0.05 |
| CE + PRA + Dec | 0.1 | 0.05 |

PRA is the paired RBF alignment loss with `PRA_SIGMA=1.0`. It is not described
as minibatch distribution MMD because the formal runs use a per-step batch
size of one.

## Efficiency protocol

Compare operators with identical GPU, batch size, hidden width, head count,
precision, sequence lengths, warm-up count, and repetitions. Run each operator
in a fresh process for peak memory. Report both the explicit pairwise
implementation and optimized SDPA: Cross-MILA reduces memory relative to the
explicit implementation at long sequence lengths, but optimized SDPA can be
more memory-efficient in the measured regime.

## Weight release

Only final `best_acc` model weights should be uploaded outside Git, together
with the matching configuration and SHA-256 checksum. Optimizer/all-state
checkpoints are useful for resuming private training but are unnecessary for
inference releases.

