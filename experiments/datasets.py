"""
Dataset utilities for reviewer-facing Task/Sensor Token Compiler experiments.

Expected preprocessed WESAD format:

    data/wesad/S2/ECG.npy
    data/wesad/S2/EDA.npy
    data/wesad/S2/BVP.npy
    data/wesad/S2/ACC.npy
    data/wesad/S2/TEMP.npy
    data/wesad/S2/RESP.npy
    data/wesad/S2/labels.npy

Signals may be [T] or [T, C]. Labels should be aligned at signal timestep
resolution or be safely indexable at the window center. If real data is absent,
synthetic data is generated so the experiment pipeline can be smoke-tested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


SENSOR_REGISTRY: Dict[str, Dict[str, object]] = {
    "ecg": {
        "sensor_type": 0,
        "measured_process": 0,
        "sampling_rate": 700.0,
        "body_location": 0,
        "physical_unit": 0,
        "channel_layout": 0,
    },
    "eeg": {
        "sensor_type": 1,
        "measured_process": 1,
        "sampling_rate": 100.0,
        "body_location": 1,
        "physical_unit": 1,
        "channel_layout": 1,
    },
    "eog": {
        "sensor_type": 2,
        "measured_process": 2,
        "sampling_rate": 100.0,
        "body_location": 2,
        "physical_unit": 1,
        "channel_layout": 0,
    },
    "emg": {
        "sensor_type": 3,
        "measured_process": 3,
        "sampling_rate": 100.0,
        "body_location": 3,
        "physical_unit": 1,
        "channel_layout": 0,
    },
    "eda": {
        "sensor_type": 4,
        "measured_process": 4,
        "sampling_rate": 4.0,
        "body_location": 4,
        "physical_unit": 2,
        "channel_layout": 0,
    },
    "ppg": {
        "sensor_type": 5,
        "measured_process": 5,
        "sampling_rate": 64.0,
        "body_location": 4,
        "physical_unit": 3,
        "channel_layout": 0,
    },
    "bvp": {
        "sensor_type": 6,
        "measured_process": 5,
        "sampling_rate": 64.0,
        "body_location": 4,
        "physical_unit": 3,
        "channel_layout": 0,
    },
    "acc": {
        "sensor_type": 7,
        "measured_process": 6,
        "sampling_rate": 32.0,
        "body_location": 4,
        "physical_unit": 4,
        "channel_layout": 2,
    },
    "gyro": {
        "sensor_type": 8,
        "measured_process": 6,
        "sampling_rate": 50.0,
        "body_location": 5,
        "physical_unit": 5,
        "channel_layout": 2,
    },
    "temp": {
        "sensor_type": 9,
        "measured_process": 7,
        "sampling_rate": 4.0,
        "body_location": 4,
        "physical_unit": 6,
        "channel_layout": 0,
    },
    "resp": {
        "sensor_type": 10,
        "measured_process": 8,
        "sampling_rate": 4.0,
        "body_location": 0,
        "physical_unit": 3,
        "channel_layout": 0,
    },
}


def get_available_wesad_sensors(data_dir: str, synthetic_if_missing: bool = True) -> List[str]:
    """Infer available WESAD sensors from local files."""
    root = Path(data_dir) / "wesad"
    if sorted(root.glob("*.parquet")):
        return ["acc", "bvp", "eda", "temp"]
    sensors = []
    for sensor in ["ecg", "eda", "bvp", "acc", "temp", "resp"]:
        if any((subject_dir / f"{sensor.upper()}.npy").exists() for subject_dir in root.glob("S*")):
            sensors.append(sensor)
    if sensors:
        return sensors
    if (root / "WESAD_raw_data.csv").exists():
        return ["ecg"]
    return ["ecg", "eda", "bvp", "acc", "temp", "resp"] if synthetic_if_missing else []


TASK_REGISTRY: Dict[str, Dict[str, int]] = {
    "ecg_diagnosis": {"task_type": 0, "task_description": 0, "num_classes": 5},
    "stress": {"task_type": 0, "task_description": 1, "num_classes": 3},
    "sleep_stage": {"task_type": 0, "task_description": 2, "num_classes": 5},
    "heart_rate": {"task_type": 1, "task_description": 3, "num_classes": 1},
    "activity": {"task_type": 0, "task_description": 4, "num_classes": 6},
}


def make_metadata(
    sensor: str,
    task: str,
    window_duration: float,
    sampling_rate: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Create scalar metadata tensors for one sample."""
    info = SENSOR_REGISTRY[sensor]
    task_info = TASK_REGISTRY[task]
    rate = float(sampling_rate if sampling_rate is not None else info["sampling_rate"])
    return {
        "sensor_type": torch.tensor(int(info["sensor_type"]), dtype=torch.long),
        "measured_process": torch.tensor(int(info["measured_process"]), dtype=torch.long),
        "sampling_rate": torch.tensor(rate, dtype=torch.float32),
        "body_location": torch.tensor(int(info["body_location"]), dtype=torch.long),
        "physical_unit": torch.tensor(int(info["physical_unit"]), dtype=torch.long),
        "channel_layout": torch.tensor(int(info["channel_layout"]), dtype=torch.long),
        "window_duration": torch.tensor(float(window_duration), dtype=torch.float32),
        "task_type": torch.tensor(int(task_info["task_type"]), dtype=torch.long),
        "task_description": torch.tensor(
            int(task_info["task_description"]), dtype=torch.long
        ),
    }


@dataclass
class SampleIndex:
    subject: str
    start: int
    label: int


class SensorWindowDataset(Dataset):
    """Single-sensor windowed dataset with metadata and subject IDs."""

    def __init__(
        self,
        signals_by_subject: Dict[str, np.ndarray],
        labels_by_subject: Dict[str, np.ndarray],
        sensor: str,
        task: str,
        window_len: int,
        stride: int | None = None,
        normalize: bool = True,
        max_windows_per_subject: int | None = None,
    ):
        self.signals_by_subject = signals_by_subject
        self.labels_by_subject = labels_by_subject
        self.sensor = sensor
        self.task = task
        self.window_len = int(window_len)
        self.stride = int(stride or max(1, window_len // 2))
        self.normalize = normalize
        self.indices: List[SampleIndex] = []

        for subject, signal in sorted(signals_by_subject.items()):
            labels = labels_by_subject[subject]
            n = len(signal)
            subject_count = 0
            for start in range(0, max(1, n - self.window_len + 1), self.stride):
                center = min(start + self.window_len // 2, len(labels) - 1)
                label = int(labels[center])
                if label < 0:
                    continue
                self.indices.append(SampleIndex(subject, start, label))
                subject_count += 1
                if max_windows_per_subject and subject_count >= max_windows_per_subject:
                    break

    @property
    def subjects(self) -> List[str]:
        return [idx.subject for idx in self.indices]

    @property
    def num_classes(self) -> int:
        return int(TASK_REGISTRY[self.task]["num_classes"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        signal = self.signals_by_subject[idx.subject]
        x = signal[idx.start : idx.start + self.window_len]
        if x.ndim == 2:
            # Convert multichannel windows to one channel for current TSFM path.
            x = x.mean(axis=-1) if x.shape[0] == self.window_len else x.mean(axis=0)
        if len(x) < self.window_len:
            x = np.pad(x, (0, self.window_len - len(x)), mode="constant")
        x = x.astype(np.float32)
        if self.normalize:
            std = float(x.std())
            if std > 1e-6:
                x = (x - float(x.mean())) / std
        rate = float(SENSOR_REGISTRY[self.sensor]["sampling_rate"])
        duration = self.window_len / max(rate, 1.0)
        metadata = make_metadata(self.sensor, self.task, duration, rate)
        return {
            "x": torch.from_numpy(x).float().unsqueeze(0),
            "y": torch.tensor(idx.label, dtype=torch.long),
            "metadata": metadata,
            "subject": idx.subject,
            "sensor": self.sensor,
        }


class WESADParquetDataset(Dataset):
    """Window-level dataset for LouisSimon/wesad-parquet shards."""

    SENSOR_COLUMNS = {
        "acc": ["acc_x", "acc_y", "acc_z"],
        "bvp": ["bvp"],
        "eda": ["eda"],
        "temp": ["temp"],
    }

    def __init__(
        self,
        parquet_files: Sequence[Path],
        sensor: str,
        window_len: int,
        max_windows_per_subject: int | None = None,
    ):
        if sensor not in self.SENSOR_COLUMNS:
            raise ValueError(f"Parquet WESAD source does not contain sensor={sensor}")
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError("Install duckdb to load WESAD parquet shards") from exc

        self.sensor = sensor
        self.task = "stress"
        self.window_len = int(window_len)
        self.rows: List[Tuple[str, int, np.ndarray]] = []
        cols = self.SENSOR_COLUMNS[sensor]
        select_cols = ", ".join(cols + ["stress", "user"])
        files_sql = ", ".join([f"'{str(p)}'" for p in parquet_files])
        if max_windows_per_subject:
            # Keep smoke runs tractable without collapsing to a one-class shard:
            # sample up to N windows for each subject/class pair.
            base = (
                f"SELECT {select_cols}, "
                "row_number() OVER (PARTITION BY user, stress ORDER BY random()) AS _rn "
                f"FROM read_parquet([{files_sql}]) WHERE stress IS NOT NULL"
            )
            query = f"SELECT {select_cols} FROM ({base}) WHERE _rn <= {int(max_windows_per_subject)}"
        else:
            query = f"SELECT {select_cols} FROM read_parquet([{files_sql}]) WHERE stress IS NOT NULL"
        df = duckdb.connect().execute(query).fetchdf()
        for _, row in df.iterrows():
            arrays = [np.asarray(row[col], dtype=np.float32) for col in cols]
            if len(arrays) == 1:
                x = arrays[0]
            else:
                min_len = min(len(a) for a in arrays)
                x = np.stack([a[:min_len] for a in arrays], axis=1)
            raw_label = int(row["stress"])
            # Public WESAD parquet uses the original WESAD stress labels:
            # 1=baseline, 2=stress, 3=amusement. Match the numpy loader's
            # contiguous 0/1/2 class convention so the head has no empty class.
            label_map = {1: 0, 2: 1, 3: 2}
            if raw_label not in label_map:
                continue
            label = label_map[raw_label]
            user = str(row["user"])
            if not user.upper().startswith("S"):
                user = f"S{user}"
            self.rows.append((user, label, x))
        self.indices = [SampleIndex(subject=user, start=i, label=label) for i, (user, label, _) in enumerate(self.rows)]

    @property
    def subjects(self) -> List[str]:
        return [idx.subject for idx in self.indices]

    @property
    def num_classes(self) -> int:
        labels = [idx.label for idx in self.indices]
        return max(labels) + 1 if labels else 2

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        subject, label, x = self.rows[item]
        if x.ndim == 2:
            x = x.mean(axis=-1)
        if len(x) > self.window_len:
            x = x[: self.window_len]
        elif len(x) < self.window_len:
            x = np.pad(x, (0, self.window_len - len(x)), mode="constant")
        x = x.astype(np.float32)
        std = float(x.std())
        if std > 1e-6:
            x = (x - float(x.mean())) / std
        rate = float(SENSOR_REGISTRY[self.sensor]["sampling_rate"])
        duration = self.window_len / max(rate, 1.0)
        metadata = make_metadata(self.sensor, self.task, duration, rate)
        return {
            "x": torch.from_numpy(x).float().unsqueeze(0),
            "y": torch.tensor(label, dtype=torch.long),
            "metadata": metadata,
            "subject": subject,
            "sensor": self.sensor,
        }


def _synthetic_wesad_sensor(
    sensor: str,
    subject: str,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetic signal with label-correlated structure for smoke tests."""
    rate = float(SENSOR_REGISTRY[sensor]["sampling_rate"])
    t = np.arange(length, dtype=np.float32) / max(rate, 1.0)
    base_freq = {
        "ecg": 1.2,
        "bvp": 1.1,
        "ppg": 1.0,
        "eda": 0.08,
        "acc": 0.7,
        "temp": 0.02,
        "resp": 0.25,
    }.get(sensor, 0.5)
    subject_shift = (abs(hash(subject)) % 17) / 100.0
    return (
        np.sin(2 * np.pi * (base_freq + subject_shift) * t)
        + 0.25 * np.sin(2 * np.pi * (base_freq * 0.5) * t)
        + 0.15 * rng.normal(size=length)
    ).astype(np.float32)


def _synthetic_labels(length: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int64)
    thirds = np.array_split(np.arange(length), 3)
    for label, idx in enumerate(thirds):
        labels[idx] = label
    # Add small label noise to avoid an unrealistically clean toy task.
    noise_idx = rng.choice(length, size=max(1, length // 25), replace=False)
    labels[noise_idx] = rng.integers(0, 3, size=len(noise_idx))
    return labels


def load_wesad_sensor(
    data_dir: str,
    sensor: str,
    window_len: int,
    synthetic_if_missing: bool = True,
    max_windows_per_subject: int | None = None,
) -> SensorWindowDataset:
    """Load one WESAD sensor from preprocessed numpy files."""
    root = Path(data_dir) / "wesad"
    sensor_key = sensor.upper()
    signals: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}

    for subject_dir in sorted(root.glob("S*")):
        sig_path = subject_dir / f"{sensor_key}.npy"
        label_path = subject_dir / "labels.npy"
        if not sig_path.exists() or not label_path.exists():
            continue
        signal = np.load(sig_path)
        raw_label = np.load(label_path).astype(np.int64)
        # WESAD labels are commonly: 1=baseline, 2=stress, 3=amusement,
        # with 0/4/5/6/7 used for transient or other conditions. Keep the
        # standard 3-class setting and mark other labels as ignored.
        label = np.full_like(raw_label, fill_value=-1, dtype=np.int64)
        label[raw_label == 1] = 0
        label[raw_label == 2] = 1
        label[raw_label == 3] = 2
        if len(signal) == 0 or len(label) == 0:
            continue
        n = min(len(signal), len(label))
        signals[subject_dir.name] = signal[:n]
        labels[subject_dir.name] = label[:n]

    if not signals:
        parquet_files = sorted(root.glob("*.parquet"))
        if parquet_files:
            return WESADParquetDataset(
                parquet_files,
                sensor=sensor,
                window_len=window_len,
                max_windows_per_subject=max_windows_per_subject,
            )

    if not signals:
        csv_path = root / "WESAD_raw_data.csv"
        if csv_path.exists():
            signals, labels = load_wesad_csv_columns(csv_path, sensor)

    if not signals:
        if not synthetic_if_missing:
            raise FileNotFoundError(f"No preprocessed WESAD files found under {root}")
        logger.warning("WESAD %s not found. Using synthetic smoke-test data.", sensor)
        rng = np.random.default_rng(1234 + int(SENSOR_REGISTRY[sensor]["sensor_type"]))
        for subject_idx in range(1, 16):
            subject = f"S{subject_idx}"
            length = max(window_len * 12, 4096)
            signals[subject] = _synthetic_wesad_sensor(sensor, subject, length, rng)
            labels[subject] = _synthetic_labels(length, rng)

    return SensorWindowDataset(
        signals,
        labels,
        sensor=sensor,
        task="stress",
        window_len=window_len,
        max_windows_per_subject=max_windows_per_subject,
    )


def _condition_to_stress_label(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "baseline", "base"}:
        return 0
    if text in {"2", "stress"}:
        return 1
    if text in {"3", "amusement", "amuse"}:
        return 2
    # Treat meditation as non-stress baseline-like for binary/3-class fallback.
    if text in {"4", "meditation", "meditate"}:
        return 0
    return -1


def _candidate_columns(sensor: str, columns: Sequence[str]) -> List[str]:
    """Find plausible WESAD raw CSV columns for a sensor."""
    lower_to_original = {c.lower(): c for c in columns}
    candidates: List[str] = []
    aliases = {
        "ecg": ["ecg"],
        "eda": ["eda", "gsr"],
        "bvp": ["bvp", "ppg"],
        "acc": ["acc", "acc_x", "acc_y", "acc_z"],
        "temp": ["temp", "temperature"],
        "resp": ["resp", "respiration"],
    }[sensor]
    for col in columns:
        lc = col.lower()
        if any(alias in lc for alias in aliases):
            # Avoid selecting derived label/meta columns.
            if not any(skip in lc for skip in ["label", "condition", "subject", "time", "sssq"]):
                candidates.append(col)
    if sensor == "acc" and len(candidates) > 3:
        # Keep tri-axial raw-ish columns if available.
        axis_cols = [c for c in candidates if any(ax in c.lower() for ax in ["x", "y", "z"])]
        if axis_cols:
            candidates = axis_cols[:3]
    return candidates[:3] if sensor == "acc" else candidates[:1]


def load_wesad_csv_columns(csv_path: Path, sensor: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load the public Hugging Face WESAD_raw_data.csv feature table.

    This file is feature-level rather than raw waveform-level. It is valid for
    a real-data smoke/P0 adaptation test, but paper-grade results should prefer
    raw or uniformly preprocessed WESAD windows when available.
    """
    header = pd.read_csv(csv_path, nrows=0)
    columns = list(header.columns)
    subject_col = next((c for c in columns if c.lower() in {"subject_id", "subject", "sid"}), None)
    condition_col = next((c for c in columns if c.lower() in {"condition", "label", "class"}), None)
    if subject_col is None or condition_col is None:
        raise ValueError(f"Could not find subject/condition columns in {csv_path}")
    sensor_cols = _candidate_columns(sensor, columns)
    if not sensor_cols:
        raise ValueError(f"Could not find columns for sensor={sensor} in {csv_path}")
    usecols = [subject_col, condition_col] + sensor_cols
    df = pd.read_csv(csv_path, usecols=usecols)
    df["_label"] = df[condition_col].map(_condition_to_stress_label).astype(np.int64)
    df = df[df["_label"] >= 0]
    signals: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}
    for subject, group in df.groupby(subject_col, sort=True):
        values = group[sensor_cols].to_numpy(dtype=np.float32)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        subject_name = str(subject)
        if not subject_name.upper().startswith("S"):
            subject_name = f"S{subject_name}"
        signals[subject_name] = values
        labels[subject_name] = group["_label"].to_numpy(dtype=np.int64)
    logger.info("Loaded %s from %s columns=%s subjects=%d", sensor, csv_path, sensor_cols, len(signals))
    return signals, labels


def split_by_subject(
    dataset: SensorWindowDataset,
    val_frac: float,
    test_frac: float,
    seed: int,
    allow_row_split: bool = False,
) -> Tuple[Subset, Subset, Subset]:
    subjects = np.array(sorted(set(dataset.subjects)))
    if len(subjects) < 3:
        if not allow_row_split:
            raise ValueError(
                f"Need at least 3 subjects for subject-level splitting; found {len(subjects)}. "
                "Download a full WESAD parquet file instead of a converted smoke shard."
            )
        # Tiny downloaded shards may contain only one subject. This fallback is
        # for smoke tests only; paper-grade runs must use subject-level splits.
        rng = np.random.default_rng(seed)
        idx = np.arange(len(dataset))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        n_val = max(1, int(round(len(idx) * val_frac)))
        test_idx = idx[:n_test].tolist()
        val_idx = idx[n_test : n_test + n_val].tolist()
        train_idx = idx[n_test + n_val :].tolist()
        if not train_idx:
            train_idx = val_idx
        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(round(len(subjects) * test_frac)))
    n_val = max(1, int(round(len(subjects) * val_frac)))
    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test : n_test + n_val])
    train_subjects = set(subjects[n_test + n_val :])
    if not train_subjects:
        train_subjects = set(subjects[n_test + n_val - 1 : n_test + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for idx, sample in enumerate(dataset.indices):
        if sample.subject in test_subjects:
            test_idx.append(idx)
        elif sample.subject in val_subjects:
            val_idx.append(idx)
        elif sample.subject in train_subjects:
            train_idx.append(idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def _labels_and_subjects(dataset: Dataset) -> Tuple[List[int], List[str]]:
    labels: List[int] = []
    subjects: List[str] = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item["y"]))
        subjects.append(str(item["subject"]))
    return labels, subjects


def validate_split_quality(name: str, dataset: Dataset, min_classes: int = 2) -> None:
    labels, subjects = _labels_and_subjects(dataset)
    if not labels:
        raise ValueError(f"{name} split is empty")
    classes = sorted(set(labels))
    if len(classes) < min_classes:
        raise ValueError(
            f"{name} split has only {len(classes)} class(es), labels={classes}, "
            f"samples={len(labels)}, subjects={len(set(subjects))}. "
            "This run is not paper-grade; use a fuller dataset shard or adjust split fractions."
        )


def limit_label_fraction(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    """Subsample labels for few-label protocols while preserving class coverage."""
    if fraction >= 0.999:
        return dataset
    rng = np.random.default_rng(seed)
    labels = []
    for i in range(len(dataset)):
        labels.append(int(dataset[i]["y"]))
    labels_arr = np.array(labels)
    keep: List[int] = []
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels_arr == label)
        if len(idx) == 0:
            continue
        k = max(1, int(round(len(idx) * fraction)))
        keep.extend(rng.choice(idx, size=k, replace=False).tolist())
    keep = sorted(keep)
    return Subset(dataset, keep)


def collate_physio(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    x = torch.stack([item["x"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    metadata: Dict[str, torch.Tensor] = {}
    for key in batch[0]["metadata"].keys():
        metadata[key] = torch.stack([item["metadata"][key] for item in batch])
    return {
        "x": x,
        "y": y,
        "metadata": metadata,
        "subject": [item["subject"] for item in batch],
        "sensor": [item["sensor"] for item in batch],
    }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_physio,
        pin_memory=torch.cuda.is_available(),
    )


def build_wesad_loso_loaders(
    data_dir: str,
    train_sensors: Iterable[str],
    test_sensor: str,
    window_len: int,
    batch_size: int,
    num_workers: int,
    val_subject_frac: float,
    test_subject_frac: float,
    label_fraction: float,
    seed: int,
    synthetic_if_missing: bool = True,
    max_windows_per_subject: int | None = None,
) -> Dict[str, object]:
    """Build WESAD leave-one-sensor-out loaders."""
    train_parts, val_parts = [], []
    for sensor in train_sensors:
        ds = load_wesad_sensor(
            data_dir,
            sensor,
            window_len,
            synthetic_if_missing=synthetic_if_missing,
            max_windows_per_subject=max_windows_per_subject,
        )
        train_ds, val_ds, _ = split_by_subject(
            ds, val_subject_frac, test_subject_frac, seed
        )
        train_parts.append(limit_label_fraction(train_ds, label_fraction, seed))
        val_parts.append(val_ds)

    test_ds_full = load_wesad_sensor(
        data_dir,
        test_sensor,
        window_len,
        synthetic_if_missing=synthetic_if_missing,
        max_windows_per_subject=max_windows_per_subject,
    )
    _, _, test_ds = split_by_subject(
        test_ds_full, val_subject_frac, test_subject_frac, seed
    )

    train_ds = ConcatDataset(train_parts)
    val_ds = ConcatDataset(val_parts)
    validate_split_quality("train", train_ds)
    validate_split_quality("val", val_ds)
    validate_split_quality("test", test_ds)
    num_classes = max(
        [getattr(part.dataset if isinstance(part, Subset) else part, "num_classes", 2) for part in train_parts + [test_ds]]
    )
    return {
        "train": make_loader(train_ds, batch_size, True, num_workers),
        "val": make_loader(val_ds, batch_size * 2, False, num_workers),
        "test": make_loader(test_ds, batch_size * 2, False, num_workers),
        "num_classes": num_classes,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
