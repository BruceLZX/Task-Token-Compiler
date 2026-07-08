"""Run reviewer-facing experiment suites.

Examples:

    python experiments/run_experiments.py --suite quick --epochs 2
    python experiments/run_experiments.py --suite wesad_loso --methods compiler_prefix,lora,metadata_mlp
    python experiments/run_experiments.py --suite few_label --label-fractions 0.01,0.05,0.1,1.0
    python experiments/run_experiments.py --suite ablations
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

if __package__:
    from .config import ExperimentConfig
    from .datasets import build_wesad_loso_loaders, get_available_wesad_sensors
    from .train_utils import train_model
else:
    from config import ExperimentConfig
    from datasets import build_wesad_loso_loaders, get_available_wesad_sensors
    from train_utils import train_model


WESAD_SENSORS = ["ecg", "eda", "bvp", "acc", "temp", "resp"]
DEFAULT_METHODS = [
    "compiler_prefix",
    "compiler_concat",
    "frozen_probe",
    "metadata_mlp",
    "mlp_token_compiler",
    "random_prefix",
    "per_sensor_prompt",
    "per_sensor_task_prompt",
    "lora",
    "adapter",
]


def parse_csv(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def base_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.device = args.device
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.data.batch_size = args.batch_size
    cfg.data.window_len = args.window_len
    cfg.data.num_workers = args.num_workers
    cfg.data.max_windows_per_subject = args.max_windows_per_subject
    cfg.backbone.freeze_backbone = not args.unfreeze_backbone
    cfg.backbone.lora_rank = args.lora_rank
    cfg.backbone.init_checkpoint = args.init_backbone_checkpoint or None
    cfg.data.synthetic_if_missing = not args.no_synthetic
    cfg.train.early_stopping_patience = args.patience
    cfg.compiler.structure_loss_weight = args.structure_loss_weight
    return cfg


def run_single(
    cfg: ExperimentConfig,
    method: str,
    holdout_sensor: str,
    label_fraction: float,
    seed: int,
    run_root: Path,
    ablation: str = "none",
) -> Dict[str, object]:
    cfg = copy.deepcopy(cfg)
    cfg.method = method
    cfg.data.holdout_sensor = holdout_sensor
    cfg.data.label_fraction = label_fraction
    cfg.train.seed = seed
    if cfg.backbone.init_checkpoint:
        cfg.backbone.init_checkpoint = cfg.backbone.init_checkpoint.format(
            holdout=holdout_sensor,
            method=method,
            seed=seed,
            label_fraction=label_fraction,
            ablation=ablation,
        )

    if ablation == "no_sensor_id":
        cfg.sensor_meta.drop_sensor_name = True
    elif ablation == "no_task_description":
        cfg.sensor_meta.drop_task_description = True
    elif ablation == "no_sampling_rate":
        cfg.sensor_meta.sampling_rate = False
    elif ablation == "no_body_location":
        cfg.sensor_meta.body_location = False
    elif ablation == "no_unit":
        cfg.sensor_meta.physical_unit = False
    elif ablation == "no_signal_summary":
        cfg.compiler.use_signal_summary = False
    elif ablation == "no_structure_loss":
        cfg.compiler.structure_loss_weight = 0.0
    elif ablation == "concat_control":
        cfg.method = "compiler_concat"

    available_sensors = get_available_wesad_sensors(
        cfg.data.data_dir, synthetic_if_missing=cfg.data.synthetic_if_missing
    )
    if holdout_sensor not in available_sensors:
        raise ValueError(
            f"holdout_sensor={holdout_sensor} is unavailable. Available WESAD sensors: {available_sensors}"
        )
    train_sensors = [s for s in available_sensors if s != holdout_sensor]
    loaders = build_wesad_loso_loaders(
        data_dir=cfg.data.data_dir,
        train_sensors=train_sensors,
        test_sensor=holdout_sensor,
        window_len=cfg.data.window_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_subject_frac=cfg.data.val_subject_frac,
        test_subject_frac=cfg.data.test_subject_frac,
        label_fraction=cfg.data.label_fraction,
        seed=cfg.train.seed,
        synthetic_if_missing=cfg.data.synthetic_if_missing,
        max_windows_per_subject=cfg.data.max_windows_per_subject,
    )
    cfg.data.num_classes = int(loaders.get("num_classes", cfg.data.num_classes))
    out_dir = run_root / f"method={cfg.method}" / f"holdout={holdout_sensor}" / f"labels={label_fraction}" / f"seed={seed}" / f"ablation={ablation}"
    return train_model(cfg, loaders, out_dir)


def write_summary(results: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "all_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    fields = [
        "method",
        "holdout_sensor",
        "label_fraction",
        "seed",
        "ablation",
        "trainable_params",
        "elapsed_sec",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_accuracy",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "method": result["method"],
                "holdout_sensor": result["holdout_sensor"],
                "label_fraction": result["label_fraction"],
                "seed": result["seed"],
                "ablation": result.get("ablation", "none"),
                "trainable_params": result["trainable_params"],
                "elapsed_sec": round(float(result["elapsed_sec"]), 2),
                "test_macro_f1": result["test"]["macro_f1"],
                "test_balanced_accuracy": result["test"]["balanced_accuracy"],
                "test_accuracy": result["test"]["accuracy"],
            }
            writer.writerow(row)


def run_suite(args) -> None:
    cfg = base_config(args)
    methods = parse_csv(args.methods) if args.methods else DEFAULT_METHODS
    seeds = parse_csv(args.seeds, int)
    if args.holdouts:
        holdouts = parse_csv(args.holdouts)
    else:
        holdouts = get_available_wesad_sensors(
            cfg.data.data_dir, synthetic_if_missing=cfg.data.synthetic_if_missing
        )
    label_fractions = parse_csv(args.label_fractions, float)
    run_root = Path(args.output_dir) / args.suite

    if args.suite == "quick":
        methods = ["compiler_prefix", "metadata_mlp", "per_sensor_prompt"]
        holdouts = [args.holdout]
        label_fractions = [1.0]
        seeds = [args.seed]
    elif args.suite == "few_label":
        holdouts = [args.holdout] if args.holdout else holdouts
    elif args.suite == "ablations":
        methods = ["compiler_prefix"]
        label_fractions = [args.ablation_label_fraction]
        holdouts = [args.holdout]
    elif args.suite == "wesad_loso":
        pass
    else:
        raise ValueError(f"Unknown suite: {args.suite}")

    ablations = ["none"]
    if args.suite == "ablations":
        ablations = [
            "none",
            "no_sensor_id",
            "no_task_description",
            "no_sampling_rate",
            "no_body_location",
            "no_unit",
            "no_signal_summary",
            "no_structure_loss",
            "concat_control",
        ]

    results: List[Dict[str, object]] = []
    for method in methods:
        for holdout in holdouts:
            for frac in label_fractions:
                for seed in seeds:
                    for ablation in ablations:
                        result = run_single(cfg, method, holdout, frac, seed, run_root, ablation)
                        result["ablation"] = ablation
                        results.append(result)
                        write_summary(results, run_root / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["quick", "wesad_loso", "few_label", "ablations"], default="quick")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", default="")
    parser.add_argument("--holdouts", default="")
    parser.add_argument("--holdout", default="eda")
    parser.add_argument("--label-fractions", default="1.0")
    parser.add_argument("--ablation-label-fraction", type=float, default=0.1)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-len", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-windows-per-subject", type=int, default=80)
    parser.add_argument("--structure-loss-weight", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--init-backbone-checkpoint", default="")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
