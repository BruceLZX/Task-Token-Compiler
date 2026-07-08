"""
Configuration objects for Task/Sensor Token Compiler experiments.

The defaults are intentionally small enough for pilot runs. Paper-grade runs
should override epochs, seeds, and backbone size from the CLI.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


MethodName = Literal[
    "compiler_prefix",
    "compiler_concat",
    "frozen_probe",
    "metadata_mlp",
    "mlp_token_compiler",
    "random_prefix",
    "per_sensor_prompt",
    "per_sensor_task_prompt",
    "sensor_id_prompt",
    "adapter",
    "lora",
]


@dataclass
class SensorMetaConfig:
    """Metadata fields compiled into sensor/task tokens."""

    sensor_type: bool = True
    measured_process: bool = True
    sampling_rate: bool = True
    body_location: bool = True
    physical_unit: bool = True
    channel_layout: bool = True
    window_duration: bool = True
    task_type: bool = True
    task_description: bool = True

    # Robustness controls.
    field_dropout: float = 0.10
    drop_sensor_name: bool = False
    drop_task_description: bool = False

    # Embedding dimensions.
    categorical_dim: int = 32
    numeric_dim: int = 16


@dataclass
class CompilerConfig:
    """Token compiler architecture."""

    num_layers: int = 2
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    num_sensor_tokens: int = 4
    num_task_tokens: int = 2
    use_signal_summary: bool = True
    structure_loss_weight: float = 0.01


@dataclass
class BackboneConfig:
    """Prefix-capable time-series backbone."""

    backbone: Literal["patch_transformer", "moment_features"] = "patch_transformer"
    d_model: int = 256
    patch_len: int = 16
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    freeze_backbone: bool = True
    lora_rank: int = 8
    init_checkpoint: Optional[str] = None

    # MOMENT feature mode is a baseline only; true prefix injection requires an
    # exposed patch embedding + encoder path.
    moment_variant: Literal["small", "base", "large"] = "base"


@dataclass
class DataConfig:
    """Dataset configuration."""

    data_dir: str = "./data"
    dataset: Literal["wesad", "sleep_edf", "ptbxl", "ppgdalia", "ucihar"] = "wesad"
    sensors: List[str] = field(
        default_factory=lambda: ["ecg", "eda", "bvp", "acc", "temp", "resp"]
    )
    holdout_sensor: str = "eda"
    task: str = "stress"
    num_classes: int = 3
    window_len: int = 512
    batch_size: int = 64
    num_workers: int = 0
    val_subject_frac: float = 0.2
    test_subject_frac: float = 0.2
    label_fraction: float = 1.0
    synthetic_if_missing: bool = True
    max_windows_per_subject: int = 80


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    scheduler: Literal["cosine", "none"] = "cosine"
    use_amp: bool = True
    early_stopping_patience: int = 6
    seed: int = 42
    grad_clip: float = 1.0


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str = "task_sensor_token_compiler"
    method: MethodName = "compiler_prefix"
    sensor_meta: SensorMetaConfig = field(default_factory=SensorMetaConfig)
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "./results"
    device: str = "cuda"


def make_config(**overrides) -> ExperimentConfig:
    """Create a config and apply shallow dotted overrides."""
    cfg = ExperimentConfig()
    for key, value in overrides.items():
        if "." not in key:
            setattr(cfg, key, value)
            continue
        obj_name, field_name = key.split(".", 1)
        setattr(getattr(cfg, obj_name), field_name, value)
    return cfg
