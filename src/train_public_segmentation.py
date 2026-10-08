"""Train/evaluate the public palpebral U-Net independently of anemia models.

Example::

    python -m src.train_public_segmentation \
      --validated data/processed/segmentation/public_palpebral_v2 \
      --output data/processed/segmentation/public_unet_mobilenetv3_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .public_segmentation_data import PublicPair, assign_subject_splits, letterbox_pair, sha256
from .segmentation_model import MobileNetSmallUNet, dice_bce_loss
from .validate_public_segmentation import write_if_same_or_new


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PublicSegmentationDataset(Dataset):
    """Preloaded, aspect-preserving RGB/binary pairs with train-only augmentation."""

    def __init__(self, records: list[dict[str, str]], size: int, augment: bool) -> None:
        self.records = records
        self.size = size
        self.augment = augment
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for row in records:
            original_bgr = cv2.imread(row["original_path"], cv2.IMREAD_COLOR)
            mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
            if original_bgr is None or mask is None:
                raise ValueError(f"Unable to decode validated pair {row['key']}")
            if set(np.unique(mask)) != {0, 255}:
                raise ValueError(f"Invalid binary mask values for {row['key']}")
            aspect_delta = abs(
                (original_bgr.shape[1] / original_bgr.shape[0]) /
                (mask.shape[1] / mask.shape[0]) - 1.0
            )
            if aspect_delta > 0.01:
                raise ValueError(f"Image/mask orientation differs for {row['key']}")
            image_rgb = cv2.cvtColor(
                cv2.resize(original_bgr, (mask.shape[1], mask.shape[0]),
                           interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            image_square, mask_square, _ = letterbox_pair(
                image_rgb, mask > 0, size
            )
            self.samples.append((image_square, mask_square))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.samples[index]
        if self.augment:
            image = image.copy()
            mask = mask.copy()
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            angle = random.uniform(-10.0, 10.0)
            scale = random.uniform(0.95, 1.05)
            matrix = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), angle, scale)
            image = cv2.warpAffine(
                image, matrix, (self.size, self.size), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            mask = cv2.warpAffine(
                mask, matrix, (self.size, self.size), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            image = np.clip(
                image.astype(np.float32) * random.uniform(0.90, 1.10), 0, 255
            ).astype(np.uint8)
        rgb_float = image.astype(np.float32) / 255.0
        normalized = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask.astype(np.float32)[None].copy())
        return image_tensor, mask_tensor


def binary_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Per-image positive-class metrics at the fixed 0.5 probability threshold."""
    predicted = predicted.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(
    model: MobileNetSmallUNet, loader: DataLoader, device: torch.device,
    timed: bool = False,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    metrics: list[dict[str, float]] = []
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            _synchronize(device)
            start = time.perf_counter()
            logits = model(images)
            _synchronize(device)
            seconds = (time.perf_counter() - start) / len(images)
            loss = dice_bce_loss(logits, masks)
            total_loss += float(loss.detach().cpu()) * len(images)
            predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
            truths = masks.cpu().numpy()
            for predicted, truth in zip(predictions, truths):
                sample = binary_metrics(predicted, truth)
                if timed:
                    sample["inference_seconds"] = seconds
                metrics.append(sample)
    summary = {name: float(np.mean([row[name] for row in metrics]))
               for name in ("dice", "iou", "precision", "recall")}
    summary["loss"] = total_loss / len(metrics)
    if timed:
        summary["mean_inference_seconds"] = float(np.mean([
            row["inference_seconds"] for row in metrics
        ]))
    return summary, metrics


def _choose_device(requested: str, size: int) -> tuple[torch.device, MobileNetSmallUNet]:
    if requested == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS is unavailable in this process")
    try:
        model = MobileNetSmallUNet(pretrained_encoder=True).to(device)
        model.train()
        dummy = torch.zeros((1, 3, size, size), device=device)
        target = torch.zeros((1, 1, size, size), device=device)
        dice_bce_loss(model(dummy), target).backward()
        _synchronize(device)
        model.zero_grad(set_to_none=True)
        return device, model
    except RuntimeError:
        if requested != "auto" or device.type != "mps":
            raise
        print("MPS smoke check failed; restarting model on CPU")
        device = torch.device("cpu")
        return device, MobileNetSmallUNet(pretrained_encoder=True).to(device)


def _read_validated(validated: Path) -> tuple[list[dict[str, str]], list[tuple[str, str, int]]]:
    with (validated / "validated_pairs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((validated / "summary.json").read_text())
    if len(rows) != int(summary["pair_count"]):
        raise ValueError("Validated pair ledger and summary disagree")
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate public subject keys")
    for row in rows:
        for path_column, hash_column in (
            ("original_path", "original_sha256"),
            ("segmented_path", "segmented_sha256"),
            ("mask_path", "mask_sha256"),
        ):
            path = Path(row[path_column])
            if not path.is_file() or sha256(path) != row[hash_column]:
                raise ValueError(f"Validated source hash mismatch: {row['key']}/{path_column}")
    similar = [
        (item["first"], item["second"], int(item["hamming_distance"]))
        for item in summary["near_duplicate_candidates"]
    ]
    return rows, similar


def train_public_unet(
    validated: Path, output: Path, size: int, batch_size: int,
    epochs: int, patience: int, learning_rate: float, seed: int,
    requested_device: str,
) -> dict[str, Any]:
    """Train on public train subjects; choose by validation; score test once."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}")
    if min(size, batch_size, epochs, patience) <= 0:
        raise ValueError("Size, batch, epochs, and patience must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows, similar = _read_validated(validated)
    pairs = [
        PublicPair(row["country"], row["subject_id"],
                   Path(row["original_path"]), Path(row["segmented_path"]))
        for row in rows
    ]
    assignment = assign_subject_splits(pairs, similar, seed)
    for first, second, _ in similar:
        if assignment[first] != assignment[second]:
            raise AssertionError("Similar public images crossed splits")
    split_rows = {
        split: [row for row in rows if assignment[row["key"]] == split]
        for split in ("train", "validation", "test")
    }
    datasets = {
        split: PublicSegmentationDataset(items, size, augment=(split == "train"))
        for split, items in split_rows.items()
    }
    loaders = {
        split: DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=0,
            generator=torch.Generator().manual_seed(seed + index),
        )
        for index, (split, dataset) in enumerate(datasets.items())
    }
    device, model = _choose_device(requested_device, size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    config: dict[str, Any] = {
        "model": "MobileNetV3-Small encoder U-Net",
        "encoder_weights": "IMAGENET1K_V1 cached locally",
        "validated_pairs_csv_sha256": sha256(validated / "validated_pairs.csv"),
        "validated_summary_sha256": sha256(validated / "summary.json"),
        "image_size": size,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "early_stopping_warmup_epochs": 5,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": device.type,
        "augmentation": "train-only horizontal flip, ±10-degree rotation, 0.95–1.05 scale, ±10% brightness",
        "mask_threshold": 0.5,
        "split_method": "subject-disjoint seeded 70/15/15 with pHash<=6 candidate grouping",
        "split_counts": {split: len(items) for split, items in split_rows.items()},
        "split_country_counts": {
            split: dict(Counter(row["country"] for row in items))
            for split, items in split_rows.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_if_same_or_new(output / "config.json", (json.dumps(config, indent=2) + "\n").encode())
    split_ledger = [
        {"key": row["key"], "split": assignment[row["key"]],
         "original_sha256": row["original_sha256"],
         "mask_sha256": row["mask_sha256"]}
        for row in rows
    ]
    write_if_same_or_new(
        output / "split.json", (json.dumps(split_ledger, indent=2) + "\n").encode()
    )
    best_dice = -1.0
    best_epoch = 0
    epochs_without_gain = 0
    logs: list[dict[str, float | int]] = []
    checkpoint = output / "best_model.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        for images, masks in loaders["train"]:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = dice_bce_loss(logits, masks)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach().cpu()) * len(images)
        validation, _ = evaluate(model, loaders["validation"], device)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": training_loss / len(datasets["train"]),
            "validation_loss": validation["loss"],
            "validation_dice": validation["dice"],
            "validation_iou": validation["iou"],
            "validation_precision": validation["precision"],
            "validation_recall": validation["recall"],
        }
        logs.append(record)
        print(
            f"epoch={epoch:02d} train_loss={record['train_loss']:.4f} "
            f"val_loss={record['validation_loss']:.4f} "
            f"val_dice={record['validation_dice']:.4f}", flush=True
        )
        if validation["dice"] > best_dice + 1e-4:
            best_dice = validation["dice"]
            best_epoch = epoch
            epochs_without_gain = 0
            torch.save({"model_state": model.state_dict(), "config": config,
                        "best_epoch": epoch}, checkpoint)
        elif epoch >= 5:
            epochs_without_gain += 1
        if epoch >= 5 and epochs_without_gain >= patience:
            print(f"early_stopping_epoch={epoch}", flush=True)
            break
    with (output / "training_log.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    validation, _ = evaluate(model, loaders["validation"], device, timed=True)
    test, per_image = evaluate(model, loaders["test"], device, timed=True)
    result = {
        "status": "public_segmentation_baseline_evaluated",
        "best_epoch": best_epoch,
        "epochs_run": len(logs),
        "validation": validation,
        "test": test,
        "test_subject_count": len(datasets["test"]),
        "test_scored_once_after_checkpoint_selection": True,
        "checkpoint_sha256": sha256(checkpoint),
        "note": "Segmentation metrics are not anemia-screening or clinical metrics.",
    }
    write_if_same_or_new(output / "metrics.json", (json.dumps(result, indent=2) + "\n").encode())
    test_rows = [
        {"key": row["key"], **metric}
        for row, metric in zip(split_rows["test"], per_image)
    ]
    with (output / "test_per_image.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    result = train_public_unet(
        args.validated, args.output, args.size, args.batch_size,
        args.epochs, args.patience, args.learning_rate, args.seed, args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
