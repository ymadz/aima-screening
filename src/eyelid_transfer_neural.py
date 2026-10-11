"""Deterministic, resumable MobileNetV3 anemia training on frozen prefix maps."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from torch import nn
from torchvision.models import mobilenet_v3_small

from .local_eyelid_hybrid_features import WEIGHTS_SHA256
from .local_eyelid_validation import sha256

SETTINGS = {
    "public": {"max_epochs": 24, "patience": 4, "head_lr": .001, "tail_lr": .0001},
    "A": {"max_epochs": 12, "patience": 4, "head_lr": .001, "tail_lr": 0.0},
    "B": {"max_epochs": 12, "patience": 4, "head_lr": .0003, "tail_lr": .00001},
}
DESIGN = {
    "architecture": "MobileNetV3-Small ImageNet1K_V1; features+global avgpool576; dropout0.2; Linear576->1 sigmoid",
    "public_adaptation": "Train binary head and last two features blocks; earlier blocks frozen ImageNet",
    "A": "Initialize public model, freeze all backbone parameters, adapt public-initialized binary head",
    "B": "Initialize same public model, adapt head plus final two backbone blocks at conservative LR",
    "batchnorm": "All BatchNorm running statistics fixed in eval, including trainable late blocks",
    "optimizer": "AdamW, weight_decay0.001, batch_size32, training-only negative/positive BCE pos_weight",
    "early_stopping": "Inner/public validation AP; AP ties lower weighted BCE; patience4",
    "outer_refit": "Use all outer training subjects for rounded median inner best epochs; no outer labels/checkpoint selection",
    "strategy_selection": "Pooled inner OOF AP; ties A; report A,B and selected separately",
    "threshold": "Max training-only OOF sensitivity with specificity>=0.80, ties specificity then higher threshold; public-only threshold uses public validation only",
    "calibration": "None; class-weighted neural scores are not calibrated clinical probabilities",
    "device": "CPU, one native thread, deterministic algorithms; label-independent frozen prefix maps cached for original and flipped inputs",
    "settings": SETTINGS,
}


def runtime(seed: int) -> None:
    torch.set_num_threads(1)
    threadpool_limits(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


class AnemiaMobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        self.features = base.features
        self.pool = base.avgpool
        self.dropout = nn.Dropout(.2)
        self.head = nn.Linear(576, 1)

    def forward(self, x: torch.Tensor, cached: bool = False) -> torch.Tensor:
        features = self.features[-2:](x) if cached else self.features(x)
        return self.head(self.dropout(self.pool(features).flatten(1))).flatten()

    def configure(self, strategy: str) -> None:
        if strategy not in SETTINGS:
            raise ValueError("Unknown transfer strategy")
        self.requires_grad_(False)
        self.head.requires_grad_(True)
        if strategy != "A":
            self.features[-2:].requires_grad_(True)

    def training_mode(self) -> None:
        self.train()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def initial_model(weights: Path, seed: int = 20261011) -> AnemiaMobileNet:
    runtime(seed)
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("Unexpected ImageNet checkpoint")
    original = mobilenet_v3_small(weights=None)
    original.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    model = AnemiaMobileNet()
    model.features.load_state_dict(original.features.state_dict())
    return model.eval()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode()); digest.update(str(tensor.shape).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prefix_cache(model: AnemiaMobileNet, inputs: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
    """Cache only fixed ImageNet early blocks, without labels or train fitting."""
    model.eval()
    prefix = model.features[:-2]
    views = []
    with torch.no_grad():
        for flipped in (False, True):
            chunks = []
            for start in range(0, len(inputs), batch_size):
                x = inputs[start:start + batch_size]
                if flipped:
                    x = x.flip(-1)
                chunks.append(prefix(x).detach().clone())
            views.append(torch.cat(chunks))
    return torch.stack(views, dim=1)


def predict(model: AnemiaMobileNet, cache: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probabilities = [model(cache[i:i + 32, 0], cached=True).sigmoid().numpy() for i in range(0, len(cache), 32)]
    return np.concatenate(probabilities).astype(np.float64)


def atomic_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> AnemiaMobileNet:
    payload = torch.load(path, weights_only=True, map_location="cpu")
    model = AnemiaMobileNet()
    model.load_state_dict(payload["state"])
    if state_hash(model.state_dict()) != payload["state_sha256"]:
        raise ValueError("Checkpoint state checksum mismatch")
    return model.eval()


def fit_model(initial_state: dict[str, torch.Tensor], cache: torch.Tensor, labels: np.ndarray,
              train: np.ndarray, validation: np.ndarray | None, subjects: list[str],
              strategy: str, directory: Path, seed: int, cache_hash: str,
              fixed_epochs: int | None = None) -> tuple[AnemiaMobileNet, dict[str, Any]]:
    """Training API validates isolation; outer refits accept no validation labels."""
    runtime(seed)
    train = np.asarray(train, dtype=int)
    validation = np.asarray(validation, dtype=int) if validation is not None else np.array([], dtype=int)
    labels = np.asarray(labels, dtype=int)
    if len(subjects) != len(cache) or len(set(subjects)) != len(subjects) or set(train) & set(validation):
        raise ValueError("Repeated subjects or train/validation overlap")
    if set(labels[train]) != {0, 1} or (len(validation) and set(labels[validation]) != {0, 1}):
        raise ValueError("Insufficient class support")
    if (fixed_epochs is None) != bool(len(validation)):
        raise ValueError("Early stopping requires inner/public validation; outer refit requires fixed epochs")
    config = SETTINGS[strategy]
    epochs = fixed_epochs or config["max_epochs"]
    signature = {"train_subjects": [subjects[i] for i in train], "validation_subjects": [subjects[i] for i in validation],
                 "training_labels": labels[train].tolist(), "validation_labels": labels[validation].tolist(),
                 "strategy": strategy, "settings": config, "epochs": epochs, "seed": seed,
                 "cache_sha256": cache_hash, "initializer_state_sha256": state_hash(initial_state)}
    signature_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    result_path, best_path, last_path = directory / "training.json", directory / "best.pt", directory / "last.pt"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["signature_sha256"] != signature_hash or sha256(best_path) != result["checkpoint_sha256"]:
            raise ValueError("Completed fit provenance changed; never silently retrain/overwrite")
        return load_checkpoint(best_path), result
    model = AnemiaMobileNet(); model.load_state_dict(initial_state); model.configure(strategy)
    parameters = [{"params": model.head.parameters(), "lr": config["head_lr"]}]
    if strategy != "A":
        parameters.append({"params": model.features[-2:].parameters(), "lr": config["tail_lr"]})
    optimizer = torch.optim.AdamW(parameters, weight_decay=.001)
    positive_weight = float(np.sum(labels[train] == 0) / np.sum(labels[train] == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    history, start, best_key, best_epoch, stale = [], 1, (-float("inf"), -float("inf")), 0, 0
    if last_path.exists():
        previous = torch.load(last_path, weights_only=True, map_location="cpu")
        if previous["signature_sha256"] != signature_hash:
            raise ValueError("Resume configuration/input/initializer mismatch")
        model.load_state_dict(previous["state"]); optimizer.load_state_dict(previous["optimizer"])
        history, start = previous["history"], previous["epoch"] + 1
        best_key, best_epoch, stale = tuple(previous["best_key"]), previous["best_epoch"], previous["stale"]
    for epoch in range(start, epochs + 1):
        if len(validation) and stale >= config["patience"]:
            break
        torch.manual_seed(seed + epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        indices = train[torch.randperm(len(train), generator=generator).numpy()]
        model.training_mode(); loss_sum = 0.0
        for offset in range(0, len(indices), 32):
            batch = indices[offset:offset + 32]
            flip = torch.randint(2, (len(batch),), generator=generator)
            x = cache[torch.tensor(batch), flip]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, cached=True), torch.tensor(labels[batch], dtype=torch.float32))
            loss.backward(); optimizer.step(); loss_sum += float(loss.detach()) * len(batch)
        row = {"epoch": epoch, "training_weighted_bce": loss_sum / len(train)}
        if len(validation):
            probabilities = predict(model, cache[validation])
            ap = float(average_precision_score(labels[validation], probabilities))
            with torch.no_grad():
                validation_loss = float(criterion(model(cache[validation, 0], cached=True), torch.tensor(labels[validation], dtype=torch.float32)))
            row.update(validation_ap=ap, validation_weighted_bce=validation_loss)
            key = (ap, -validation_loss)
        else:
            key = (float(epoch), 0.0)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            atomic_save({"state": model.state_dict(), "state_sha256": state_hash(model.state_dict()),
                         "epoch": epoch, "signature_sha256": signature_hash}, best_path)
        else:
            stale += 1
        history.append(row)
        atomic_save({"state": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "signature_sha256": signature_hash, "history": history, "best_key": best_key,
                     "best_epoch": best_epoch, "stale": stale}, last_path)
    model = load_checkpoint(best_path)
    result = {**signature, "signature_sha256": signature_hash, "best_epoch": best_epoch,
              "history": history, "pos_weight_training_only": positive_weight, "checkpoint_sha256": sha256(best_path),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "training_status": "complete"}
    # Count parameters on the configured training model, not the reloaded default.
    model.configure(strategy)
    result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result_path.write_text(json.dumps(result, indent=2))
    return model.eval(), result
