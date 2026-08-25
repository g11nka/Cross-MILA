# GitHub publication checklist

## Commit to the repository

- Source code under `src/` and `scripts/`.
- Split-construction utilities and protocol documentation.
- Manuscript figures under `assets/`.
- Numeric results and raw efficiency exports under `results/`.
- `README.md`, `LICENSE`, `NOTICE`, `.gitignore`, and dependency metadata.

## Keep outside Git

- Raw or processed IEMOCAP, ESD, and MELD data.
- `train.pkl`, `val.pkl`, and `test.pkl` files containing local media paths.
- Training logs, TensorBoard/MLflow runs, caches, and IDE metadata.
- Periodic and full-state checkpoints containing optimizer, scheduler, or AMP state.

## Optional weight release

Release only selected validation-best inference weights through GitHub Releases
or an archival service such as Zenodo. For every weight file, include:

1. the matching `cfg.log`;
2. dataset, fold, split-manifest hash, and selection metric;
3. the Git commit used for training;
4. a SHA-256 checksum;
5. the exact evaluation command.

Do not place weight binaries in normal Git history. If files exceed the host's
release limits, publish them on Zenodo and link the DOI from the repository.

Before publishing, replace `Ginka` in `LICENSE` and `NOTICE` with the preferred
public author or copyright-holder name if necessary.
