import logging
import os
import sys

current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(scripts_dir)
src_dir = os.path.join(project_root, "src")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
import csv
import glob
import argparse
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from data.dataloader import build_train_test_dataset
from tqdm.auto import tqdm
from models import networks
from configs.base import Config
from collections import Counter
from typing import Tuple


def calculate_accuracy(y_true, y_pred) -> Tuple[float, float]:
    class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
    bacc = float(
        balanced_accuracy_score(
            y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true]
        )
    )
    acc = float(accuracy_score(y_true, y_pred))
    return bacc, acc


def calculate_f1_score(y_true, y_pred) -> Tuple[float, float]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    return macro_f1, weighted_f1


def eval(cfg, checkpoint_path, all_state_dict=True, cm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = getattr(networks, cfg.model_type)(cfg)
    network.to(device)

    # Build dataset
    # Final LOSO training disables validation, but evaluation must still build test.pkl.
    cfg.train_without_validation = False
    _, test_ds = build_train_test_dataset(cfg)
    weight = torch.load(checkpoint_path, map_location=torch.device(device))
    if all_state_dict:
        weight = weight["state_dict_network"]

    network.load_state_dict(weight)
    network.eval()
    network.to(device)

    y_actu = []
    y_pred = []
    use_amp = bool(getattr(cfg, "use_amp", False) and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    MAX_AUDIO_LEN = 546220  # MELD config 中的 audio_max_length，约 34.1 秒 @16kHz

    for every_test_list in tqdm(test_ds):
        is_cached = (
            isinstance(every_test_list, dict)
            and bool(every_test_list.get("precomputed_features", False))
        )
        if is_cached:
            text_features = every_test_list["text_features"].to(device)
            audio_features = every_test_list["audio_features"].to(device)
            label = every_test_list["label"].to(device)
            text_mask = every_test_list["text_mask"].to(device)
            audio_feature_mask = every_test_list[
                "audio_feature_mask"
            ].to(device)
        elif isinstance(every_test_list, dict):
            input_ids = every_test_list["input_text"]
            audio = every_test_list["input_audio"]
            label = every_test_list["label"]
            text_mask = every_test_list["text_mask"]
            audio_mask = every_test_list["audio_mask"]
            audio_lengths = every_test_list["audio_lengths"]
        else:
            input_ids, audio, label = every_test_list
            text_mask = None
            audio_mask = None
            audio_lengths = None

        if not is_cached:
            # Safety crop for abnormal MELD clips during evaluation.
            if audio.dim() == 2 and audio.size(1) > MAX_AUDIO_LEN:
                print(
                    f"[Eval Crop] audio length {audio.size(1)} "
                    f"-> {MAX_AUDIO_LEN}"
                )
                audio = audio[:, :MAX_AUDIO_LEN]
                if audio_mask is not None:
                    audio_mask = audio_mask[:, :MAX_AUDIO_LEN]
                if audio_lengths is not None:
                    audio_lengths = audio_lengths.clamp_max(MAX_AUDIO_LEN)
            elif audio.dim() == 1 and audio.size(0) > MAX_AUDIO_LEN:
                print(
                    f"[Eval Crop] audio length {audio.size(0)} "
                    f"-> {MAX_AUDIO_LEN}"
                )
                audio = audio[:MAX_AUDIO_LEN]

            input_ids = input_ids.to(device)
            audio = audio.to(device)
            label = label.to(device)
            if text_mask is not None:
                text_mask = text_mask.to(device)
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
            if audio_lengths is not None:
                audio_lengths = audio_lengths.to(device)

        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            if is_cached:
                output = network.forward_precomputed(
                    text_features,
                    audio_features,
                    text_mask=text_mask,
                    audio_feature_mask=audio_feature_mask,
                    domain_grl_scale=0.0,
                )[0]
            else:
                output = network(
                    input_ids,
                    audio,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                    audio_lengths=audio_lengths,
                    domain_grl_scale=0.0,
                )[0]
            _, preds = torch.max(output, 1)
            y_actu.extend(label.detach().cpu().view(-1).tolist())
            y_pred.extend(preds.detach().cpu().view(-1).tolist())
    bacc, acc = calculate_accuracy(y_actu, y_pred)
    macro_f1, weighted_f1 = calculate_f1_score(y_actu, y_pred)
    if cm:
        cm = confusion_matrix(y_actu, y_pred)
        print("Confusion Matrix: \n", cm)
        cmn = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100

        ax = plt.subplots(figsize=(8, 5.5))[1]
        sns.heatmap(
            cmn,
            cmap=sns.light_palette("#4A90C2", as_cmap=True),
            annot=True,
            square=True,
            vmin=0,
            vmax=100,
            linecolor="#C8DCEB",
            linewidths=0.75,
            ax=ax,
            fmt=".2f",
            annot_kws={"size": 16},
        )
        ax.set_xlabel("Predicted", fontsize=18, fontweight="bold")
        ax.xaxis.set_label_position("bottom")
        label_names = ["Anger", "Happiness", "Sadness", "Neutral"]
        if cfg.num_classes != 4:
            with open(os.path.join(cfg.data_root, "classes.json"), "r") as f:
                label_data = json.load(f)
                label_names = label_data.keys()

        ax.xaxis.set_ticklabels(label_names, fontsize=16)
        ax.set_ylabel("Ground Truth", fontsize=18, fontweight="bold")
        ax.yaxis.set_ticklabels(label_names, fontsize=16)
        plt.tight_layout()
        plt.savefig(
            "confusion_matrix_" + cfg.name + cfg.data_valid + ".png",
            format="png",
            dpi=1200,
        )

    return bacc, acc, macro_f1, weighted_f1


def find_checkpoint_folder(path):
    candidate = os.listdir(path)
    if "logs" in candidate and "weights" in candidate and "cfg.log" in candidate:
        return [path]
    list_candidates = []
    for c in candidate:
        list_candidates += find_checkpoint_folder(os.path.join(path, c))
    return list_candidates


def main(args):
    logging.info("Finding checkpoints")
    list_checkpoints = find_checkpoint_folder(args.checkpoint_path)
    test_set = args.test_set if args.test_set is not None else "test.pkl"
    csv_path = os.path.basename(args.checkpoint_path) + "{}.csv".format(test_set)
    # field names
    fields = ["BACC", "ACC", "MACRO_F1", "WEIGHTED_F1", "Time", "Model", "Settings"]
    with open(csv_path, "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for ckpt in list_checkpoints:
            meta_info = ckpt.split("/")
            time = meta_info[-1]
            settings = meta_info[-2]
            model_name = meta_info[-3]
            logging.info("Evaluating: {}/{}/{}".format(model_name, settings, time))
            cfg_path = os.path.join(ckpt, "cfg.log")
            if args.latest:
                ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pt"))
                if len(ckpt_path) != 0:
                    ckpt_path = ckpt_path[0]
                    all_state_dict = True
                else:
                    ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pth"))[0]
                    all_state_dict = False

            else:
                ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0_0.pt")
                all_state_dict = True
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0.pth")
                    all_state_dict = False

            cfg = Config(validate_paths=False)
            cfg.load(cfg_path)
            cfg.batch_size = 2
            cfg.audio_max_length = 546220
            cfg.text_max_length = 297
            cfg.num_workers = 0

            test_set = args.test_set if args.test_set is not None else "test.pkl"
            cfg.data_valid = test_set

            if args.data_root is not None:
                assert (
                    args.data_name is not None
                ), "Change validation dataset requires data_name"
                cfg.data_root = args.data_root
                cfg.data_name = args.data_name

            bacc, acc, macro_f1, weighted_f1 = eval(
                cfg, ckpt_path, all_state_dict=all_state_dict, cm=args.confusion_matrix
            )
            writer.writerows(
                [
                    {
                        "BACC": round(bacc * 100, 2),
                        "ACC": round(acc * 100, 2),
                        "MACRO_F1": round(macro_f1 * 100, 2),
                        "WEIGHTED_F1": round(weighted_f1 * 100, 2),
                        "Time": time,
                        "Model": model_name,
                        "Settings": settings,
                    }
                ]
            )
            logging.info(
                "\nBACC | ACC | MACRO_F1 | WEIGHTED_F1 \n{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(
                    round(bacc * 100, 2),
                    round(acc * 100, 2),
                    round(macro_f1 * 100, 2),
                    round(weighted_f1 * 100, 2),
                )
            )


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ckpt", "--checkpoint_path", type=str, help="path to checkpoint folder"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="optional direct path to an all-state .pt checkpoint",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="whether to travel child folder or not",
    )

    parser.add_argument(
        "-l",
        "--latest",
        action="store_true",
        help="whether to use latest weight or best weight",
    )

    parser.add_argument(
        "-t",
        "--test_set",
        type=str,
        default=None,
        help="name of testing set. Ex: test.pkl",
    )

    parser.add_argument(
        "-cm",
        "--confusion_matrix",
        action="store_true",
        help="whether to export consution matrix or not",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="If want to change the validation dataset",
    )
    parser.add_argument(
        "--data_name", type=str, default=None, help="for changing validation dataset"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if not args.recursive:
        cfg_path = os.path.join(args.checkpoint_path, "cfg.log")
        if args.checkpoint_file is not None:
            ckpt_path = args.checkpoint_file
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            all_state_dict = True
        else:
            all_state_dict = True
            ckpt_path = os.path.join(
                args.checkpoint_path, "weights/best_acc/checkpoint_0_0.pt"
            )
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(
                    args.checkpoint_path, "weights/best_acc/checkpoint_0.pth"
                )
                all_state_dict = False

        cfg = Config(validate_paths=False)
        cfg.load(cfg_path)
        # Change to test set
        test_set = args.test_set if args.test_set is not None else "test.pkl"
        cfg.data_valid = test_set
        if args.data_root is not None:
            assert (
                args.data_name is not None
            ), "Change validation dataset requires data_name"
            cfg.data_root = args.data_root
            cfg.data_name = args.data_name

        bacc, acc, macro_f1, weighted_f1 = eval(
            cfg,
            ckpt_path,
            cm=args.confusion_matrix,
            all_state_dict=all_state_dict,
        )
        logging.info(
            "\nBACC | ACC | MACRO_F1 | WEIGHTED_F1 \n{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(
                round(bacc * 100, 2),
                round(acc * 100, 2),
                round(macro_f1 * 100, 2),
                round(weighted_f1 * 100, 2),
            )
        )

    else:
        main(args)
