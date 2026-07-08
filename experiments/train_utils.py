"""Training and evaluation utilities."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

if __package__:
    from .model import TaskSensorModel, count_total_params, count_trainable_params
else:
    from model import TaskSensorModel, count_total_params, count_trainable_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    metadata = {k: v.to(device) for k, v in batch["metadata"].items()}
    return {
        "x": batch["x"].to(device),
        "y": batch["y"].to(device),
        "metadata": metadata,
        "subject": batch["subject"],
        "sensor": batch["sensor"],
    }


def classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def label_counts(values: List[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_backbone_checkpoint(model: TaskSensorModel, checkpoint: str) -> Dict[str, object]:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Backbone init checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    backbone_state = {}
    target_keys = set(model.backbone.state_dict().keys())
    for key, value in state.items():
        if key.startswith("backbone."):
            target_key = key.removeprefix("backbone.")
            if target_key in target_keys:
                backbone_state[target_key] = value
            else:
                for suffix in ("linear1", "linear2"):
                    mapped_key = target_key.replace(f".{suffix}.", f".{suffix}.base.")
                    if mapped_key in target_keys:
                        backbone_state[mapped_key] = value
                        break
    if not backbone_state:
        raise ValueError(f"Checkpoint has no backbone.* parameters: {path}")
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    return {
        "path": str(path),
        "loaded_tensors": len(backbone_state),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool, grad_clip: float):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits, tokens = model(batch["x"], batch["metadata"], return_tokens=True)
            ce = F.cross_entropy(logits, batch["y"])
            reg = model.regularization_loss(batch["metadata"], tokens)
            loss = ce + reg
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        batch_n = batch["y"].shape[0]
        total_loss += float(loss.detach().cpu()) * batch_n
        n += batch_n
    return {"loss": total_loss / max(1, n)}


@torch.no_grad()
def evaluate(model, loader, device, split: str = "val") -> Dict[str, float]:
    model.eval()
    losses, y_true, y_pred = [], [], []
    by_sensor: Dict[str, Tuple[List[int], List[int]]] = {}
    for batch in tqdm(loader, desc=split, leave=False):
        batch = move_batch(batch, device)
        logits = model(batch["x"], batch["metadata"])
        loss = F.cross_entropy(logits, batch["y"])
        pred = logits.argmax(dim=-1)
        losses.append(float(loss.detach().cpu()) * batch["y"].shape[0])
        yt = batch["y"].detach().cpu().tolist()
        yp = pred.detach().cpu().tolist()
        y_true.extend(yt)
        y_pred.extend(yp)
        for sensor, truth, guess in zip(batch["sensor"], yt, yp):
            if sensor not in by_sensor:
                by_sensor[sensor] = ([], [])
            by_sensor[sensor][0].append(truth)
            by_sensor[sensor][1].append(guess)

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = sum(losses) / max(1, len(y_true))
    metrics["true_counts"] = label_counts(y_true)
    metrics["pred_counts"] = label_counts(y_pred)
    for sensor, (truths, guesses) in by_sensor.items():
        sm = classification_metrics(truths, guesses)
        metrics[f"{sensor}_macro_f1"] = sm["macro_f1"]
        metrics[f"{sensor}_balanced_accuracy"] = sm["balanced_accuracy"]
    return metrics


def train_model(cfg, loaders: Dict[str, object], output_dir: Path) -> Dict[str, object]:
    set_seed(cfg.train.seed)
    device = get_device(cfg.device)
    model = TaskSensorModel(cfg).to(device)
    backbone_init_info = None
    if cfg.backbone.init_checkpoint:
        backbone_init_info = load_backbone_checkpoint(model, cfg.backbone.init_checkpoint)
        model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = None
    if cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp and device.type == "cuda")

    best_metric = -1.0
    best_state = None
    patience = 0
    history = []
    start = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            cfg.train.use_amp,
            cfg.train.grad_clip,
        )
        val_metrics = evaluate(model, loaders["val"], device, split="val")
        if scheduler is not None:
            scheduler.step()
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        score = val_metrics.get("macro_f1", val_metrics.get("balanced_accuracy", 0.0))
        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device, split="test")
    elapsed = time.time() - start
    result = {
        "method": cfg.method,
        "seed": cfg.train.seed,
        "holdout_sensor": cfg.data.holdout_sensor,
        "label_fraction": cfg.data.label_fraction,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
        "elapsed_sec": elapsed,
        "train_size": loaders.get("train_size"),
        "val_size": loaders.get("val_size"),
        "test_size": loaders.get("test_size"),
        "best_val_macro_f1": best_metric,
        "backbone_init": backbone_init_info,
        "random_frozen_backbone": bool(
            cfg.backbone.backbone == "patch_transformer"
            and cfg.backbone.freeze_backbone
            and not cfg.backbone.init_checkpoint
        ),
        "test": test_metrics,
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    state_dict = model.state_dict()
    omitted = []
    if cfg.backbone.backbone == "moment_prefix":
        # The official frozen MOMENT checkpoint is reloaded from Hugging Face.
        # Saving it for every method/holdout makes diagnostic artifacts enormous.
        keep = {}
        for key, value in state_dict.items():
            if key.startswith("backbone.moment."):
                omitted.append(key)
            else:
                keep[key] = value
        state_dict = keep
    torch.save(
        {
            "state_dict": state_dict,
            "omitted_frozen_backbone_tensors": len(omitted),
            "backbone": cfg.backbone.backbone,
        },
        output_dir / "model.pt",
    )
    return result
