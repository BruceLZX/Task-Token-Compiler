# Task/Sensor Token Compiler Experiments

This folder implements the ICLR pilot for:

```text
sensor metadata + task description
    -> compiled sensor/task tokens
    -> frozen prefix-capable time-series backbone
    -> downstream prediction
```

The code is organized around reviewer challenges rather than only standard
benchmark accuracy.

## Install

```bash
cd /Users/zhexiangli/Research/AAAI27/task_token_compiler
python3 -m venv .venv
source .venv/bin/activate
pip install -r experiments/requirements.txt
```

## Data Format

The WESAD pilot expects preprocessed numpy files:

```text
data/wesad/S2/ECG.npy
data/wesad/S2/EDA.npy
data/wesad/S2/BVP.npy
data/wesad/S2/ACC.npy
data/wesad/S2/TEMP.npy
data/wesad/S2/RESP.npy
data/wesad/S2/labels.npy
```

If files are missing, the loader generates synthetic smoke-test data by default.
Use `--no-synthetic` to force real data.

As a fallback real-data format, the loader also accepts the public Hugging Face
feature table:

```text
data/wesad/WESAD_raw_data.csv
```

This CSV path is useful for fast real-data smoke tests. Paper-grade claims
should prefer raw or uniformly preprocessed waveform windows.

For a better real WESAD smoke test, download the public parquet shard:

```bash
python -m experiments.download_wesad_hf --source parquet_0000 --out-dir data/wesad
```

This provides `acc`, `bvp`, `eda`, and `temp` window arrays plus stress labels.

## Smoke Test

```bash
python -m experiments.run_experiments \
  --suite quick \
  --epochs 2 \
  --batch-size 16 \
  --window-len 128 \
  --device cpu
```

## Main Reviewer-Challenge Suites

### 1. WESAD Leave-One-Sensor-Out

Tests whether compiled tokens generalize to a held-out sensor under the same
stress-classification task.

```bash
python -m experiments.run_experiments \
  --suite wesad_loso \
  --methods compiler_prefix,metadata_mlp,per_sensor_prompt,lora,adapter \
  --holdouts ecg,eda,bvp,acc,temp,resp \
  --seeds 1,2,3,4,5 \
  --epochs 30 \
  --no-synthetic
```

### 2. Few-Label New Sensor

Tests label efficiency against LoRA, prompts, and metadata side features.

```bash
python -m experiments.run_experiments \
  --suite few_label \
  --holdout eda \
  --methods compiler_prefix,metadata_mlp,per_sensor_prompt,lora,adapter \
  --label-fractions 0.01,0.05,0.1,0.5,1.0 \
  --seeds 1,2,3,4,5 \
  --epochs 30 \
  --no-synthetic
```

### 3. Ablations and Mechanism Checks

Tests the likely reviewer objections:

- is it just sensor ID lookup?
- does task description matter?
- is sampling rate/body location/unit useful?
- is prefix injection better than post-backbone concatenation?
- does the token-structure regularizer matter?

```bash
python -m experiments.run_experiments \
  --suite ablations \
  --holdout eda \
  --ablation-label-fraction 0.1 \
  --seeds 1,2,3,4,5 \
  --epochs 30 \
  --no-synthetic
```

## Methods Implemented

| Method | Purpose |
|---|---|
| `compiler_prefix` | proposed metadata/task-to-token compiler with true prefix injection |
| `compiler_concat` | post-backbone concat control |
| `frozen_probe` | no adaptation |
| `metadata_mlp` | metadata as side feature |
| `mlp_token_compiler` | simple metadata-to-prefix baseline for over-engineering checks |
| `random_prefix` | parameter-count control |
| `per_sensor_prompt` | learned sensor prompt lookup |
| `per_sensor_task_prompt` | stronger learned sensor-task prompt lookup |
| `sensor_id_prompt` | alias for per-sensor prompt |
| `adapter` | bottleneck PEFT baseline |
| `lora` | LoRA on transformer feed-forward layers |

Current backbone note: the implemented backbone is a prefix-capable
PatchTransformer-style encoder so the proposed method can be tested with true
token injection. For paper-grade MOMENT experiments, do not use post-hoc MOMENT
features as the main method; expose MOMENT's patch embeddings and encoder and
insert compiled tokens before the transformer.

## Outputs

Each run writes:

```text
results/<suite>/.../result.json
results/<suite>/.../model.pt
results/<suite>/summary.csv
results/<suite>/all_results.json
```

Main paper decisions should use `summary.csv` plus the ablation tables described
in `research/iclr_success_criteria.md`.

Aggregate multi-seed results with:

```bash
python -m experiments.aggregate_results results/wesad_loso/summary.csv
```
