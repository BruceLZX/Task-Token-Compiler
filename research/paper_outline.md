# Paper Outline: Task/Sensor Token Compiler

**Primary title:** Task Tokens for Physiological Foundation Models: Compiling Sensor Metadata into Adaptation Tokens  
**Alternative title:** Token-Conditioned Adaptation for Heterogeneous Biomedical Time Series  
**Target:** ICLR main track only if P0 evidence in `iclr_revision_plan.md` is met; otherwise ICLR workshop / ML4H / UbiComp.

---

## 0. Central Thesis

This paper is not about training a new wearable foundation model, and not about attaching a standard adapter to every dataset.

The core idea is a **token compiler**:

```text
sensor description + sampling rate + body location + unit + task description
    -> learned sensor/task adaptation tokens
    -> frozen or lightly adapted time-series/audio foundation model
    -> downstream prediction
```

Different physiological signals should not require separate models or separate LoRA modules. Instead, the model receives a compact set of compiled tokens that tells a frozen foundation model how to interpret the current sensor and task.

The paper's central claim:

> A structured sensor/task-to-token interface can adapt a frozen physiological or time-series foundation model across heterogeneous biomedical signals more efficiently and more compositionally than ordinary PEFT or channel-text conditioning.

## 1. Introduction

Biomedical sensing is structurally heterogeneous:

- ECG, EEG, EOG, EMG, EDA, PPG/BVP, ACC/Gyro, temperature, and respiration measure different physiological processes.
- Sampling rates range from a few Hz to hundreds of Hz.
- Units and body locations vary across datasets and devices.
- Downstream tasks include diagnosis, stress classification, sleep staging, heart-rate regression, and activity recognition.

Existing solutions are limited:

- Training a wearable foundation model from scratch is expensive and tied to available pretraining data.
- Full fine-tuning and LoRA/adapters often learn dataset- or sensor-specific parameters.
- Channel-description methods use text or channel metadata, but they do not directly define a reusable adaptation-token interface for frozen physiological foundation models.

We propose **Task/Sensor Token Compiler**:

- compile structured sensor/task metadata into adaptation tokens;
- insert tokens into the foundation model token sequence;
- train only the compiler and lightweight heads, optionally with light backbone adaptation;
- evaluate whether compiled tokens generalize to missing sensors, unseen sensor combinations, and low-label new tasks.

## 2. Research Question

Main RQ:

> Can sensor metadata and task intent be compiled into adaptation tokens that let a frozen foundation model dynamically reinterpret heterogeneous biomedical time series?

Sub-questions:

1. Do compiled sensor/task tokens outperform frozen probes, LoRA/adapters, and channel-text embeddings under matched parameter budgets?
2. Do tokens generalize to unseen sensor combinations or missing sensor settings?
3. Does the learned token space reflect sensor/task structure rather than dataset shortcuts?
4. Are compiled tokens more stable and label-efficient than ordinary PEFT?

## 3. Related Work and Positioning

### Task Tokens

Task Tokens for behavior foundation models show that learned extra tokens can adapt a frozen transformer-style foundation model to new behavior-control tasks while preserving generalization. Our work imports this adaptation-interface idea into physiological time series, but the token source is different: structured sensor metadata and biomedical task intent rather than behavior task feedback.

### CHARM and Channel Descriptions

CHARM uses channel-level textual descriptions to improve multivariate time-series representation and channel-order invariance. Our difference is that we compile structured sensor/task metadata into trainable adaptation tokens for frozen or lightly adapted foundation models. CHARM-style channel text embedding is a required baseline.

### NormWear

NormWear motivates the problem: wearable physiological foundation models must handle heterogeneous sensing configurations. Our paper does not train a new wearable foundation model. It asks whether existing TSFM/audio FMs can be adapted through a small token interface.

### Gen-P-Tuning and Sensor-Prompt Tuning

Gen-P-Tuning adapts frozen univariate TSFMs to multivariate healthcare time series. Sensor-Prompt Tuning adapts MOMENT to motion sensors in few-shot HAR using sensor-friendly filters and gating. These works make the prompt-tuning baseline stronger; therefore our contribution must be more structured: a compiler from explicit sensor/task metadata to adaptation tokens, tested across physiological sensor heterogeneity.

## 4. Method

### 4.1 Metadata Input

The compiler input should be explicit and structured:

```text
sensor_description: ECG / EEG / EDA / BVP / PPG / ACC / Gyro / EMG / EOG / Temp / Resp
sampling_rate: continuous value, e.g. 4, 32, 64, 100, 500, 700 Hz
body_location: chest / wrist / scalp / eye / chin / waist / finger
unit: mV / uV / uS / g / rad/s / bpm / arbitrary optical
channel_layout: single-channel / 3-axis / 12-lead / multi-channel
task_description: stress classification / sleep staging / ECG diagnosis / HR regression / activity recognition
```

Important design rule:

- The method may include sensor IDs, but the main experiment must show that it is not just a sensor-ID lookup table.
- Numeric fields such as sampling rate and window length should use continuous or Fourier features, not only coarse bins.
- Task description should be part of the compiler, because the original idea is task/sensor token adaptation, not sensor-only prompting.

### 4.2 Token Compiler

Recommended architecture:

```text
field encoders for sensor/task metadata
    -> field dropout and metadata masking
    -> small transformer or cross-attention compiler
    -> K adaptation tokens in backbone hidden dimension
```

The compiler should output both:

- **sensor tokens**: describe how to interpret the measurement source;
- **task tokens**: describe prediction intent and label semantics.

The compiler must be compared against two simple alternatives:

1. **Metadata MLP compiler:** field embeddings -> MLP -> tokens.
2. **Per-sensor-task prompt table:** learned prompt for each observed sensor-task pair.

This is necessary because a reviewer can reasonably argue that a transformer compiler is overkill unless it beats simpler metadata-conditioned prompt generators.

To reach ICLR-level novelty, the compiler should not be trained as a plain supervised prompt module. Use a **leave-configuration-out episodic training objective**:

```text
For each training episode:
    sample a task/dataset
    sample observed sensors/configurations as support
    hold out one sensor/configuration as query
    train compiler tokens on support
    optimize query performance with the held-out sensor/configuration metadata
```

This directly trains the compiler for the generalization behavior claimed in the paper: unseen sensors, missing sensors, and new sensor combinations.

Add a structured token regularizer:

```text
metadata graph distance(sensor/task i, sensor/task j)
    should correlate with
token embedding distance(i, j)
```

This makes the compiler more than "learned prompts"; it encourages tokens to preserve physiological and task structure.

Two implementation variants:

1. **Prefix-token injection:** prepend compiled tokens to patch embeddings before the frozen transformer encoder.
2. **Layer-wise token conditioning:** insert or attend to tokens at selected layers, similar to prefix tuning.

Avoid making the main method post-backbone concatenation only; that would reduce the idea to metadata side features.

### 4.3 Backbone Choices

Primary:

- MOMENT with true pre-encoder token insertion, because the claim is foundation-model adaptation rather than a custom PatchTransformer result.

Secondary if feasible:

- one additional TSFM such as MOIRAI, Chronos, or TimesFM-style feature extraction where the task is appropriate;
- audio foundation model such as wav2vec/HuBERT only if high-frequency waveform preprocessing is defensible;
- NormWear only if code/weights and protocol access are practical.

PatchTransformer is allowed only as an engineering/debugging scaffold. It cannot be the main evidence for the paper.

### 4.4 Training

Main setting:

- frozen backbone;
- trainable token compiler;
- trainable task head;
- optional small projection layer.

Secondary setting:

- lightly adapt final transformer block or use LoRA with compiled tokens to test complementarity.

Regularization:

- field dropout to prevent shortcuts;
- sensor-name dropout to force use of sampling rate/location/unit/task;
- cross-dataset batching;
- task-balanced sampling.
- episodic leave-sensor/leave-configuration-out training;
- token-structure regularization aligned with metadata similarity;
- counterfactual consistency under metadata perturbations that preserve the task.

### 4.5 Why This Is Not Ordinary PEFT

LoRA/adapters learn parameters tied to a target dataset or task. Per-sensor prompts learn tokens tied to observed sensors. The compiler learns a **function from metadata to adaptation tokens**. This distinction matters only if the paper proves:

- the same compiler handles unseen sensor combinations;
- the compiler works when sensor ID/name is removed or masked;
- task tokens change behavior for the same sensor under different tasks;
- prefix-token conditioning beats post-backbone metadata concatenation.

The strongest version also shows **multi-task sharing**: one compiler shared across stress classification, sleep staging, HR regression, and activity recognition, compared against separate per-task LoRA/adapters.

## 5. Experiments

The experiments should prove the original idea:

> sensor/task tokens are a better adaptation interface than no token, ordinary PEFT, or channel-text conditioning.

### Experiment 1: Main Adaptation Benchmark

Datasets:

- WESAD: ECG / EDA / BVP / ACC / TEMP / RESP, stress classification.
- Sleep-EDF: EEG / EOG / EMG, sleep staging.
- PTB-XL: ECG, diagnosis classification.
- PPG-DaLiA: PPG / ACC / ECG, heart-rate regression.
- UCI HAR: ACC / Gyro, activity classification.

Backbone:

- MOMENT frozen as the primary backbone, with compiled tokens inserted before the transformer encoder.
- PatchTransformer results are reported only as implementation sanity checks or omitted from the main paper.

Compare:

| Method | Purpose |
|---|---|
| frozen TSFM probe | no adaptation-token baseline |
| metadata MLP side feature | tests whether tokens are more than side information |
| LoRA / adapter | ordinary PEFT baseline |
| FourierFT / VeRA / DoRA where feasible | stronger PEFT challenge beyond LoRA |
| per-sensor learned prompt | tests whether compiler beats lookup prompts |
| per-sensor-task learned prompt | stronger lookup-prompt baseline |
| metadata MLP token generator | tests whether transformer compiler is necessary |
| channel text embedding | CHARM-style baseline |
| Gen-P-Tuning / SPT-style baseline | strongest prompt-tuning competitors where feasible |
| compiled sensor/task tokens | proposed method |
| full fine-tune | upper bound |

Metrics:

- classification: macro-F1, balanced accuracy, AUROC where appropriate;
- sleep staging: macro-F1 and Cohen's kappa;
- regression: MAE/RMSE/correlation;
- efficiency: trainable parameters, training time, memory;
- stability: variance over seeds and label budgets.

### Experiment 2: Missing Sensor and New Sensor Combination

Goal:

- Show that compiled tokens generalize when sensor availability changes.

Protocols:

- WESAD leave-one-sensor-out: train with all but one sensor, test the held-out sensor for the same stress task.
- WESAD missing-sensor subsets: train on multiple sensor combinations, test unseen combinations.
- PPG-DaLiA: evaluate PPG/ACC/ECG subsets for HR estimation.
- Sleep-EDF: evaluate EEG/EOG/EMG channel removal where signal validity permits.

Key comparisons:

- LoRA and per-sensor prompts should struggle with unseen combinations because their parameters are tied to seen configurations.
- Compiled tokens should handle new combinations by composing metadata fields.

### Experiment 3: Few-Label New Task/Sensor Adaptation

Goal:

- Show label efficiency and parameter efficiency.

Protocol:

- For each target dataset/sensor/task, train with 1%, 5%, 10%, 50%, 100% labels.
- Compare compiled tokens against LoRA, adapters, frozen probe, and channel text embedding.

Expected publishable pattern:

- compiled tokens win most clearly at 1%-10%;
- LoRA may catch up with full labels but uses more parameters and is less transferable;
- compiled tokens have lower variance across seeds.

### Experiment 3B: Multi-Task Shared Compiler

Goal:

- Show the main advantage of a compiler over per-task PEFT: one shared metadata-to-token function can adapt across tasks.

Protocol:

- Jointly train one compiler over at least two task families, preferably:
  - WESAD stress classification;
  - Sleep-EDF sleep staging;
  - PPG-DaLiA heart-rate regression or UCI HAR activity recognition.
- Use task-specific heads but a shared compiler and shared frozen backbone.
- Compare against separate LoRA/adapters trained per task and per-task prompt tables.

Required result:

- The shared compiler should match or beat separate PEFT in low-label target adaptation while using fewer total trainable parameters across tasks.

### Experiment 4: Cross-Dataset Same-Sensor Transfer

Goal:

- Prove tokens are not dataset shortcuts.

Protocols:

- ECG: PTB-XL ECG -> WESAD ECG / PPG-DaLiA ECG with new task heads.
- ACC: WESAD/PPG-DaLiA wrist ACC -> UCI HAR ACC/Gyro with new task heads.
- Optical pulse: WESAD BVP -> PPG-DaLiA PPG if preprocessing supports it.

Interpretation:

- Do not claim direct zero-shot label transfer when tasks differ.
- Claim that the sensor/task token interface transfers representation adaptation with low-label target heads.

### Experiment 5: Token Space and Mechanism Analysis

Goal:

- Show the compiler learns structured tokens, not arbitrary prompts.

Analyses:

- token similarity by sensor family, measured process, body location, unit, and task;
- nearest-neighbor retrieval in token space;
- token distance correlation with metadata distance;
- counterfactual edits: change sampling rate/body location/task text and observe representation or output changes;
- attention/attribution from sensor/task tokens to signal patches;
- ablation of each metadata field.

Avoid relying only on t-SNE; use quantitative token-space tests.

## 6. Ablation Checklist

Must include:

- with vs without task description;
- with vs without sensor description;
- with vs without sampling rate;
- with vs without body location;
- with vs without unit;
- with vs without sensor ID/name;
- metadata-only tokens vs random tokens;
- compiled tokens vs per-sensor learned tokens;
- prefix injection vs post-backbone concatenation;
- number of tokens K = 1, 4, 8, 16;
- frozen backbone vs lightly adapted backbone.

## 7. Strongest Claim if Results Work

The paper can claim:

> We propose a sensor/task-to-token compiler as a structured adaptation interface for physiological foundation models. Across heterogeneous biomedical datasets, compiled adaptation tokens improve label efficiency, parameter efficiency, and robustness to missing or new sensor configurations compared with frozen probes, ordinary PEFT, and channel-description baselines.

For a strong ICLR paper, the claim should include one extra algorithmic sentence:

> We train the compiler with leave-configuration-out episodes and structured token regularization, yielding a compositional adaptation mechanism rather than a collection of sensor-specific prompts.

## 8. Go / No-Go Criteria

Continue toward ICLR only if:

- compiled tokens beat LoRA/adapters, Gen-P-Tuning-style, and channel-text baselines on at least 2 dataset families;
- MOMENT results are consistent with the custom/debug backbone results;
- gains are strongest in low-label or missing-sensor settings;
- removing sensor/task metadata fields meaningfully changes performance in interpretable ways;
- token space shows measurable structure;
- prefix tokens beat metadata side-feature concatenation;
- results are not only from WESAD or subject leakage.

If the method only beats frozen probes, or only works with sensor IDs, the idea is not strong enough for ICLR main.
