import hashlib
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import BertTokenizer

from configs.base import Config
from models.networks import MemoCMT


def _sample_fields(sample) -> Tuple[str, str, int]:
    if isinstance(sample, dict):
        audio_path = sample.get("audio_path", sample.get("audio"))
        return str(audio_path), str(sample.get("text", "")), int(sample["label"])
    if len(sample) < 3:
        raise ValueError("Each item must contain audio path, text, and label.")
    return str(sample[0]), str(sample[1]), int(sample[2])


def infer_domain_key(
    audio_path: str,
    data_name: str,
    domain_type: str = "speaker",
) -> Optional[str]:
    normalized = audio_path.replace("\\", "/")
    domain_type = domain_type.lower()

    if data_name == "IEMOCAP":
        if domain_type == "speaker":
            match = re.search(r"Ses(\d{2})([FM])", normalized, re.IGNORECASE)
            return (
                f"Ses{match.group(1)}{match.group(2).upper()}"
                if match
                else None
            )
        match = re.search(r"/Session(\d+)/", normalized, re.IGNORECASE)
        return f"Session{match.group(1)}" if match else None

    if data_name == "ESD":
        pair_lookup = {
            "0011": "Pair1",
            "0016": "Pair1",
            "0012": "Pair2",
            "0017": "Pair2",
            "0013": "Pair3",
            "0018": "Pair3",
            "0014": "Pair4",
            "0019": "Pair4",
            "0015": "Pair5",
            "0020": "Pair5",
        }
        for part in normalized.split("/"):
            if re.fullmatch(r"\d{4}", part):
                return pair_lookup.get(part, part) if domain_type == "session" else part
        return None

    if data_name == "MELD":
        match = re.search(r"dia(\d+)_utt\d+", normalized, re.IGNORECASE)
        return f"dialogue_{match.group(1)}" if match else None

    return None


def feature_cache_key(audio_path: str, text: str) -> str:
    normalized = os.path.normcase(os.path.abspath(os.path.normpath(audio_path)))
    payload = f"{normalized}\n{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def feature_cache_path(cache_root: str, audio_path: str, text: str) -> str:
    key = feature_cache_key(audio_path, text)
    return os.path.join(cache_root, "items", key[:2], f"{key}.pt")


def infer_training_domain_mapping(cfg: Config) -> Dict[str, int]:
    train_path = os.path.join(cfg.data_root, "train.pkl")
    with open(train_path, "rb") as handle:
        samples = pickle.load(handle)
    domain_type = getattr(cfg, "adversarial_domain", "speaker")
    keys = {
        infer_domain_key(_sample_fields(sample)[0], cfg.data_name, domain_type)
        for sample in samples
    }
    keys.discard(None)
    return {key: index for index, key in enumerate(sorted(keys))}


def _load_torch_file(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class MaskedBatchCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        text = [sample["input_text"].long() for sample in samples]
        audio = [sample["input_audio"].float() for sample in samples]
        text_lengths = torch.tensor([item.size(0) for item in text], dtype=torch.long)
        audio_lengths = torch.tensor([item.size(0) for item in audio], dtype=torch.long)
        input_text = pad_sequence(text, batch_first=True, padding_value=self.pad_token_id)
        input_audio = pad_sequence(audio, batch_first=True, padding_value=0.0)
        text_steps = torch.arange(input_text.size(1)).unsqueeze(0)
        audio_steps = torch.arange(input_audio.size(1)).unsqueeze(0)
        domain_ids = torch.tensor(
            [int(sample["domain_id"]) for sample in samples],
            dtype=torch.long,
        )
        session_ids = torch.tensor(
            [int(sample["session_id"]) for sample in samples],
            dtype=torch.long,
        )

        return {
            "input_text": input_text,
            "input_audio": input_audio,
            "label": torch.stack([sample["label"] for sample in samples]).long(),
            "text_mask": text_steps < text_lengths.unsqueeze(1),
            "audio_mask": audio_steps < audio_lengths.unsqueeze(1),
            "text_lengths": text_lengths,
            "audio_lengths": audio_lengths,
            "domain_id": domain_ids,
            "session_id": session_ids,
            "precomputed_features": False,
        }


class CachedFeatureCollator:
    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        text = [sample["text_features"] for sample in samples]
        audio = [sample["audio_features"] for sample in samples]
        text_lengths = torch.tensor([item.size(0) for item in text], dtype=torch.long)
        audio_lengths = torch.tensor([item.size(0) for item in audio], dtype=torch.long)
        text_features = pad_sequence(text, batch_first=True, padding_value=0.0)
        audio_features = pad_sequence(audio, batch_first=True, padding_value=0.0)
        text_steps = torch.arange(text_features.size(1)).unsqueeze(0)
        audio_steps = torch.arange(audio_features.size(1)).unsqueeze(0)
        domain_ids = torch.tensor(
            [int(sample["domain_id"]) for sample in samples],
            dtype=torch.long,
        )
        session_ids = torch.tensor(
            [int(sample["session_id"]) for sample in samples],
            dtype=torch.long,
        )

        return {
            "text_features": text_features,
            "audio_features": audio_features,
            "label": torch.stack([sample["label"] for sample in samples]).long(),
            "text_mask": text_steps < text_lengths.unsqueeze(1),
            "audio_feature_mask": audio_steps < audio_lengths.unsqueeze(1),
            "text_lengths": text_lengths,
            "audio_feature_lengths": audio_lengths,
            "domain_id": domain_ids,
            "session_id": session_ids,
            "precomputed_features": True,
        }


class EmotionBalancedBatchSampler(Sampler[List[int]]):
    """Builds emotion-balanced batches with cross-domain positives."""

    def __init__(self, dataset: "BaseDataset", batch_size: int, seed: int = 0):
        if batch_size < 2:
            raise ValueError("Balanced batches require batch_size >= 2.")

        self.batch_size = batch_size
        self.num_batches = max(1, len(dataset) // batch_size)
        self.seed = seed
        self.epoch = 0
        self.class_to_indices = defaultdict(list)
        self.class_domain_to_indices = defaultdict(lambda: defaultdict(list))
        for index, (label, domain_id) in enumerate(
            zip(dataset.labels, dataset.session_ids)
        ):
            label = int(label)
            domain_id = int(domain_id)
            self.class_to_indices[label].append(index)
            self.class_domain_to_indices[label][domain_id].append(index)
        self.classes = sorted(self.class_to_indices)

    def __len__(self) -> int:
        return self.num_batches

    @staticmethod
    def _sample_for_class(rng, domain_to_indices, fallback, count):
        valid_domains = [key for key in domain_to_indices if key >= 0]
        if not valid_domains:
            return (
                rng.sample(fallback, count)
                if len(fallback) >= count
                else rng.choices(fallback, k=count)
            )

        rng.shuffle(valid_domains)
        selected = []
        for offset in range(count):
            domain = valid_domains[offset % len(valid_domains)]
            candidates = domain_to_indices[domain]
            unused = [index for index in candidates if index not in selected]
            selected.append(rng.choice(unused or candidates))
        return selected

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        classes_per_batch = min(len(self.classes), max(2, self.batch_size // 2))

        for _ in range(self.num_batches):
            selected_classes = rng.sample(self.classes, classes_per_batch)
            per_class = self.batch_size // classes_per_batch
            remainder = self.batch_size % classes_per_batch
            batch = []
            for class_index, label in enumerate(selected_classes):
                count = per_class + int(class_index < remainder)
                batch.extend(
                    self._sample_for_class(
                        rng,
                        self.class_domain_to_indices[label],
                        self.class_to_indices[label],
                        count,
                    )
                )
            rng.shuffle(batch)
            yield batch


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        data_mode: str = "train.pkl",
        encoder_model: Union[MemoCMT, None] = None,
        domain_to_id: Optional[Dict[str, int]] = None,
        session_to_id: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.is_train = "train" in data_mode.lower()
        self.data_name = cfg.data_name
        self.domain_type = getattr(cfg, "adversarial_domain", "speaker")
        self.use_precomputed_features = getattr(
            cfg,
            "use_precomputed_features",
            False,
        )
        self.feature_cache_root = getattr(cfg, "feature_cache_root", "")
        self.feature_cache_in_memory = getattr(
            cfg,
            "feature_cache_in_memory",
            False,
        )

        with open(os.path.join(cfg.data_root, data_mode), "rb") as data_file:
            self.data_list = pickle.load(data_file)

        if cfg.text_encoder_type != "bert":
            raise NotImplementedError(
                f"Tokenizer {cfg.text_encoder_type} is not implemented"
            )
        self.tokenizer = (
            None
            if self.use_precomputed_features
            else BertTokenizer.from_pretrained("bert-base-uncased")
        )
        self.pad_token_id = 0 if self.tokenizer is None else self.tokenizer.pad_token_id

        self.audio_max_length = cfg.audio_max_length
        self.text_max_length = cfg.text_max_length
        self.use_masked_batching = getattr(cfg, "use_masked_batching", False)
        self.pad_to_max_length = not self.use_masked_batching
        self.audio_encoder_type = cfg.audio_encoder_type
        fields = [_sample_fields(item) for item in self.data_list]
        self.labels = [item[2] for item in fields]
        self.domain_keys = [
            infer_domain_key(item[0], self.data_name, self.domain_type)
            for item in fields
        ]
        if domain_to_id is None:
            unique_keys = sorted({key for key in self.domain_keys if key is not None})
            domain_to_id = {key: index for index, key in enumerate(unique_keys)}
        self.domain_to_id = dict(domain_to_id)
        self.domain_ids = [
            self.domain_to_id.get(key, -1) if key is not None else -1
            for key in self.domain_keys
        ]
        self.session_keys = [
            infer_domain_key(item[0], self.data_name, "session")
            for item in fields
        ]
        if session_to_id is None:
            unique_sessions = sorted(
                {
                    key
                    for key in self.session_keys
                    if key is not None
                }
            )
            session_to_id = {
                key: index for index, key in enumerate(unique_sessions)
            }
        self.session_to_id = dict(session_to_id)
        self.session_ids = [
            self.session_to_id.get(key, -1) if key is not None else -1
            for key in self.session_keys
        ]

        self.encode_data = False
        self.list_encode_audio_data = []
        self.list_encode_text_data = []
        if encoder_model is not None and not self.use_precomputed_features:
            self._encode_data(encoder_model)
            self.encode_data = True

        self.cached_items = None
        if self.use_precomputed_features:
            if not self.feature_cache_root:
                raise ValueError("feature_cache_root must be set for cached training.")
            missing = [
                feature_cache_path(self.feature_cache_root, audio_path, text)
                for audio_path, text, _ in fields
                if not os.path.isfile(
                    feature_cache_path(self.feature_cache_root, audio_path, text)
                )
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} cached feature files are missing. "
                    f"First missing file: {missing[0]}"
                )
            if self.feature_cache_in_memory:
                logging.info("Loading %d cached samples into host memory...", len(fields))
                self.cached_items = [
                    _load_torch_file(
                        feature_cache_path(self.feature_cache_root, audio_path, text)
                    )
                    for audio_path, text, _ in fields
                ]

    def _encode_data(self, encoder):
        logging.info("Encoding data with encoders in evaluation mode...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.eval()
        encoder.to(device)
        with torch.no_grad():
            for sample in tqdm(self.data_list):
                audio_path, text, _ = _sample_fields(sample)
                samples = self.__paudio__(audio_path)
                audio_embedding = (
                    encoder.encode_audio(samples.unsqueeze(0).to(device))
                    .squeeze(0)
                    .detach()
                    .cpu()
                )
                self.list_encode_audio_data.append(audio_embedding)
                input_ids = self.__ptext__(text)
                text_embedding = (
                    encoder.encode_text(input_ids.unsqueeze(0).to(device))
                    .squeeze(0)
                    .detach()
                    .cpu()
                )
                self.list_encode_text_data.append(text_embedding)

    def __getitem__(self, index: int):
        audio_path, text, label = _sample_fields(self.data_list[index])
        domain_id = self.domain_ids[index]
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.use_precomputed_features:
            cached = (
                self.cached_items[index]
                if self.cached_items is not None
                else _load_torch_file(
                    feature_cache_path(self.feature_cache_root, audio_path, text)
                )
            )
            if int(cached["label"]) != label:
                raise ValueError(f"Cached label mismatch for {audio_path}")
            return {
                "text_features": cached["text_features"],
                "audio_features": cached["audio_features"],
                "label": label_tensor,
                "domain_id": domain_id,
                "session_id": self.session_ids[index],
            }

        input_audio = (
            self.list_encode_audio_data[index]
            if self.encode_data
            else self.__paudio__(audio_path)
        )
        input_text = (
            self.list_encode_text_data[index]
            if self.encode_data
            else self.__ptext__(text)
        )
        if self.use_masked_batching:
            return {
                "input_text": input_text,
                "input_audio": input_audio,
                "label": label_tensor,
                "domain_id": domain_id,
                "session_id": self.session_ids[index],
            }
        return input_text, input_audio, label_tensor

    def __paudio__(self, file_path: str) -> torch.Tensor:
        samples, sample_rate = sf.read(file_path, dtype="float32")
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = torch.from_numpy(samples.astype(np.float32))
        if sample_rate != 16000:
            samples = torchaudio.functional.resample(samples, sample_rate, 16000)
        if self.audio_max_length is not None:
            samples = samples[: self.audio_max_length]
            if self.pad_to_max_length and samples.size(0) < self.audio_max_length:
                samples = torch.nn.functional.pad(
                    samples,
                    (0, self.audio_max_length - samples.size(0)),
                )
        return samples

    @staticmethod
    def _text_preprocessing(text):
        text = re.sub(r"[\(\[].*?[\)\]]", "", text)
        text = re.sub(" +", " ", text).strip()
        try:
            text = " ".join(text.split())
        except (AttributeError, TypeError):
            text = "NULL"
        return text if text.strip() else "NULL"

    def __ptext__(self, text: str) -> torch.Tensor:
        text = self._text_preprocessing(text)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if self.text_max_length is not None:
            input_ids = input_ids[: self.text_max_length]
            if self.pad_to_max_length and len(input_ids) < self.text_max_length:
                input_ids = np.pad(
                    input_ids,
                    (0, self.text_max_length - len(input_ids)),
                    "constant",
                    constant_values=self.tokenizer.pad_token_id,
                )
        return torch.from_numpy(np.asarray(input_ids, dtype=np.int64))

    def __len__(self):
        return len(self.data_list)


def build_train_test_dataset(
    cfg: Config,
    encoder_model: Union[MemoCMT, None] = None,
):
    dataset_map = {"IEMOCAP": BaseDataset, "ESD": BaseDataset, "MELD": BaseDataset}
    dataset = dataset_map.get(cfg.data_name)
    if dataset is None:
        raise NotImplementedError(
            f"Dataset {cfg.data_name} is not implemented, "
            f"available datasets: {dataset_map.keys()}"
        )

    train_data = dataset(cfg, data_mode="train.pkl", encoder_model=encoder_model)
    if encoder_model is not None:
        encoder_model.eval()

    skip_validation = getattr(cfg, "train_without_validation", False)
    eval_data = None
    if not skip_validation:
        eval_set = cfg.data_valid if cfg.data_valid is not None else "test.pkl"
        eval_data = dataset(
            cfg,
            data_mode=eval_set,
            encoder_model=encoder_model,
            domain_to_id=train_data.domain_to_id,
            session_to_id=train_data.session_to_id,
        )

    use_masked_batching = getattr(cfg, "use_masked_batching", False)
    if train_data.use_precomputed_features:
        collate_fn = CachedFeatureCollator()
    elif use_masked_batching:
        collate_fn = MaskedBatchCollator(train_data.pad_token_id)
    else:
        collate_fn = None

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    if use_masked_batching and getattr(cfg, "use_balanced_batches", False):
        train_dataloader = DataLoader(
            train_data,
            batch_sampler=EmotionBalancedBatchSampler(
                train_data,
                batch_size=cfg.batch_size,
                seed=getattr(cfg, "seed", 0),
            ),
            collate_fn=collate_fn,
            **loader_kwargs,
        )
    else:
        train_dataloader = DataLoader(
            train_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

    eval_dataloader = None
    if eval_data is not None:
        eval_dataloader = DataLoader(
            eval_data,
            batch_size=(
                getattr(cfg, "eval_batch_size", cfg.batch_size)
                if use_masked_batching
                else 1
            ),
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
    return train_dataloader, eval_dataloader
