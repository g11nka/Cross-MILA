import argparse
import json
import os
import pickle
import random
import re
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


SPLIT_FILES = ("train.pkl", "val.pkl", "test.pkl")

IEMOCAP_GROUPS = [f"Session{i}" for i in range(1, 6)]
ESD_GROUPS = [
    ("0011", "0016"),
    ("0012", "0017"),
    ("0013", "0018"),
    ("0014", "0019"),
    ("0015", "0020"),
]

CLASS_NAMES = {
    "IEMOCAP": ["Anger", "Happiness", "Sadness", "Neutral"],
    "ESD": ["Angry", "Happy", "Sad", "Neutral", "Surprise"],
}


def load_unique_samples(source_dir):
    samples_by_path = {}
    for filename in SPLIT_FILES:
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing source split: {path}")
        with path.open("rb") as handle:
            samples = pickle.load(handle)
        for sample in samples:
            if not isinstance(sample, (tuple, list)) or len(sample) < 3:
                raise ValueError(f"Unexpected sample format in {path}: {sample!r}")
            audio_path = os.path.normcase(os.path.normpath(str(sample[0])))
            if audio_path in samples_by_path and samples_by_path[audio_path] != sample:
                raise ValueError(f"Conflicting duplicate sample: {audio_path}")
            samples_by_path[audio_path] = tuple(sample)
    return list(samples_by_path.values())


def sample_group(dataset, sample):
    audio_path = str(sample[0])
    if dataset == "IEMOCAP":
        match = re.search(r"[\\/]Session([1-5])[\\/]", audio_path, re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot extract IEMOCAP session from: {audio_path}")
        return f"Session{match.group(1)}"

    match = re.search(r"[\\/](00(?:1[1-9]|20))[\\/]", audio_path)
    if not match:
        raise ValueError(f"Cannot extract ESD English speaker from: {audio_path}")
    return match.group(1)


def label_counts(samples):
    return {str(k): v for k, v in sorted(Counter(int(x[2]) for x in samples).items())}


def shuffle_samples(samples, seed):
    result = list(samples)
    random.Random(seed).shuffle(result)
    return result


def build_strict_fold(dataset, samples, fold_index, seed):
    if dataset == "IEMOCAP":
        test_groups = {IEMOCAP_GROUPS[fold_index]}
        val_groups = {IEMOCAP_GROUPS[(fold_index + 1) % 5]}
        all_groups = set(IEMOCAP_GROUPS)
    else:
        test_groups = set(ESD_GROUPS[fold_index])
        val_groups = set(ESD_GROUPS[(fold_index + 1) % 5])
        all_groups = {speaker for pair in ESD_GROUPS for speaker in pair}

    train_groups = all_groups - test_groups - val_groups
    train = [x for x in samples if sample_group(dataset, x) in train_groups]
    val = [x for x in samples if sample_group(dataset, x) in val_groups]
    test = [x for x in samples if sample_group(dataset, x) in test_groups]
    return (
        shuffle_samples(train, seed + fold_index),
        shuffle_samples(val, seed + 100 + fold_index),
        shuffle_samples(test, seed + 200 + fold_index),
        train_groups,
        val_groups,
        test_groups,
    )


def build_paper_like_fold(dataset, samples, fold_index, seed):
    if dataset == "IEMOCAP":
        test_groups = {IEMOCAP_GROUPS[fold_index]}
    else:
        test_groups = set(ESD_GROUPS[fold_index])

    test = [x for x in samples if sample_group(dataset, x) in test_groups]
    development = [x for x in samples if sample_group(dataset, x) not in test_groups]
    train, val = train_test_split(
        development,
        test_size=0.1,
        random_state=seed + fold_index,
        stratify=[int(x[2]) for x in development],
    )
    train_groups = {sample_group(dataset, x) for x in train}
    val_groups = {sample_group(dataset, x) for x in val}
    return (
        shuffle_samples(train, seed + fold_index),
        shuffle_samples(val, seed + 100 + fold_index),
        shuffle_samples(test, seed + 200 + fold_index),
        train_groups,
        val_groups,
        test_groups,
    )


def verify_split(dataset, train, val, test, strict):
    path_sets = [
        {os.path.normcase(os.path.normpath(str(x[0]))) for x in split}
        for split in (train, val, test)
    ]
    if path_sets[0] & path_sets[1] or path_sets[0] & path_sets[2] or path_sets[1] & path_sets[2]:
        raise RuntimeError("Audio-path overlap detected between generated splits")

    group_sets = [
        {sample_group(dataset, x) for x in split}
        for split in (train, val, test)
    ]
    if group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("Test-speaker overlap detected")
    if strict and (group_sets[0] & group_sets[1]):
        raise RuntimeError("Train/validation speaker overlap detected in strict mode")


def write_fold(dataset, output_dir, fold_number, split_data, mode, seed, source_dir):
    train, val, test, train_groups, val_groups, test_groups = split_data
    verify_split(dataset, train, val, test, strict=(mode == "strict"))

    fold_dir = output_dir / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for filename, samples in zip(SPLIT_FILES, (train, val, test)):
        with (fold_dir / filename).open("wb") as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    classes = {name: index for index, name in enumerate(CLASS_NAMES[dataset])}
    with (fold_dir / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump(classes, handle, ensure_ascii=False, indent=2)

    manifest = {
        "dataset": dataset,
        "mode": mode,
        "seed": seed,
        "source_dir": str(source_dir.resolve()),
        "fold": fold_number,
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_label_counts": label_counts(train),
        "val_label_counts": label_counts(val),
        "test_label_counts": label_counts(test),
    }
    with (fold_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Build five speaker-independent folds from existing MemoCMT PKL files."
    )
    parser.add_argument("--dataset", required=True, choices=["IEMOCAP", "ESD"])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["strict", "paper_like"],
        default="strict",
        help=(
            "strict: train/val/test groups are disjoint; "
            "paper_like: four outer folds form the development pool and validation "
            "is a stratified utterance split within that pool."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples = load_unique_samples(args.source)
    expected_groups = (
        set(IEMOCAP_GROUPS)
        if args.dataset == "IEMOCAP"
        else {speaker for pair in ESD_GROUPS for speaker in pair}
    )
    observed_groups = {sample_group(args.dataset, x) for x in samples}
    if observed_groups != expected_groups:
        raise RuntimeError(
            f"Unexpected groups. Expected {sorted(expected_groups)}, "
            f"observed {sorted(observed_groups)}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    builder = build_strict_fold if args.mode == "strict" else build_paper_like_fold
    for fold_index in range(5):
        split_data = builder(args.dataset, samples, fold_index, args.seed)
        write_fold(
            args.dataset,
            args.output,
            fold_index + 1,
            split_data,
            args.mode,
            args.seed,
            args.source,
        )


if __name__ == "__main__":
    main()
