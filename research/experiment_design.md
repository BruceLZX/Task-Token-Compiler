# Task/Sensor Token Compiler — Experiment Design

## Goal

Test the original idea rigorously:

```text
sensor metadata + task description
    -> compiled sensor/task tokens
    -> frozen or lightly adapted foundation model
    -> downstream prediction
```

The method should be judged as an **adaptation interface**, not merely as a new model.

## Hypotheses

H1. Compiled sensor/task tokens outperform frozen probes and metadata side-feature baselines because they condition the backbone representation before prediction.

H2. Compiled tokens outperform ordinary PEFT under low-label and missing-sensor settings because they share structure across sensors/tasks rather than learning isolated parameter modules.

H3. Compiled tokens generalize better than per-sensor learned prompts because the compiler uses compositional metadata: sensor description, sampling rate, unit, body location, and task description.

H4. The learned token space has interpretable structure aligned with sensor/task semantics.

## Metadata Schema

Use the following metadata fields for every sensor/task instance:

| Field | Example values | Why included |
|---|---|---|
| sensor_description | ECG, EEG, EDA, PPG, BVP, ACC, Gyro, EMG, EOG | high-level modality semantics |
| measured_process | cardiac electrical, optical pulse, skin conductance, motion | helps unseen or related sensors |
| sampling_rate_hz | 4, 32, 64, 100, 500, 700 | temporal scale |
| body_location | chest, wrist, scalp, eye, chin, waist | deployment/context information |
| unit | mV, uV, uS, g, rad/s, arbitrary optical | physical scale/meaning |
| channel_layout | univariate, xyz, 12-lead, multi-channel | input structure |
| window_duration_sec | 5.12, 10, 30 | temporal context |
| task_description | stress classification, sleep staging, ECG diagnosis, HR regression, HAR | prediction intent |
| label_space | 3-class stress, 5-stage sleep, 5-class ECG superdiagnosis | task-head semantics |

Sensor ID/name should be ablated. The main paper must show the compiler is not only a lookup table.

## Dataset Roles

| Dataset | Sensors | Task | Main role |
|---|---|---|---|
| WESAD | ECG, EDA, BVP, ACC, TEMP, RESP | stress classification | primary heterogeneous sensor test |
| Sleep-EDF | EEG, EOG, EMG | sleep staging | physiological channel/task generalization |
| PTB-XL | 12-lead ECG | diagnosis classification | ECG source and same-sensor transfer |
| PPG-DaLiA | PPG, ACC, ECG | heart-rate regression | multimodal cardiac/motion regression |
| UCI HAR | ACC, Gyro | activity classification | IMU adaptation and SPT-style comparison |

Do not merge incompatible label spaces into a single classification head. Use task-specific heads when tasks differ.

## Model Variants

### Proposed

1. **TC-metadata:** compiler uses sensor metadata + task description.
2. **TC-metadata+summary:** compiler additionally uses lightweight signal summaries such as band power, amplitude stats, missingness, and autocorrelation scale.
3. **TC-light-adapt:** compiler + lightweight final-layer or LoRA adaptation to test complementarity.
4. **TC-episodic:** compiler trained with leave-sensor / leave-configuration episodes.

### Required Baselines

| Baseline | What it tests |
|---|---|
| Frozen TSFM + linear head | no adaptation |
| Frozen TSFM + metadata MLP after backbone | metadata as side info |
| Metadata MLP -> prefix tokens | whether transformer compiler is necessary |
| Random prefix tokens | parameter-count control |
| Per-sensor learned prompt tokens | prompt lookup table |
| Per-sensor-task learned prompt tokens | stronger prompt lookup baseline |
| Per-task learned prompt tokens | task prompt lookup table |
| LoRA rank 4/8/16 | standard PEFT |
| FourierFT / DoRA / VeRA where feasible | stronger PEFT challenge |
| Adapter tuning | standard PEFT |
| CHARM-style channel text embedding | channel-description competitor |
| Gen-P-Tuning-style prompt adaptation | healthcare TSFM prompt competitor |
| Sensor-Prompt Tuning-style filter/gating | IMU/HAR prompt competitor |
| MAML / ProtoNet-style meta-learning baseline | tests whether episodic compiler is just standard meta-learning |
| Full fine-tune | upper bound |
| InceptionTime/PatchTST/TimesNet | strong non-FM supervised baseline |

Backbone requirement:

- Main paper: true MOMENT prefix injection is mandatory.
- Debug only: custom PatchTransformer may be used to validate plumbing, but cannot support the main claim.
- Secondary paper-grade backbone: at least one of MOIRAI/Chronos/TimesFM-style models if engineering time permits.

## Experiment 1: Core Benchmark Across Datasets

Purpose:

- Establish that compiled sensor/task tokens are useful across heterogeneous biomedical time series.

Protocol:

1. Train each method on the standard train split of each dataset.
2. Use subject-level splits wherever subjects exist.
3. Keep preprocessing and heads matched across methods.
4. Report 5 seeds.

Metrics:

- WESAD: macro-F1, balanced accuracy.
- Sleep-EDF: macro-F1, Cohen's kappa.
- PTB-XL: macro-F1, AUROC.
- PPG-DaLiA: MAE, RMSE, Pearson r.
- UCI HAR: macro-F1, accuracy.
- All: trainable parameters, training time, peak memory.

Expected interpretation:

- This is not the main novelty proof, but it verifies the method is competitive.
- ICLR submission requires at least two dataset families, not WESAD-only.

## Experiment 2: Missing Sensor / New Sensor Combination

Purpose:

- Directly test the key generalization claim from the original idea.

### 2A. WESAD Leave-One-Sensor-Out

For each held-out sensor:

```text
Train: all other WESAD sensors
Test: held-out sensor
Task: stress classification
```

Held-out sensors:

- ECG, EDA, BVP, ACC, TEMP, RESP.

Important:

- Keep subject-level test split.
- Report per-held-out-sensor and average performance.

Why this is central:

- It tests whether metadata-compiled tokens can handle a sensor not used for adaptation training in the same task.

### 2B. Unseen Sensor Combinations

Protocol:

- Train on single sensors and common pairs.
- Test on unseen pairs/triples.
- Example: train ECG, EDA, ACC separately; test ECG+EDA, EDA+ACC, ECG+EDA+ACC.

Evaluation:

- performance on unseen combinations;
- degradation under missing sensors;
- calibration under missingness.

Expected result:

- LoRA/per-sensor prompts should be less compositional.
- Compiler should compose tokens from sensor/task metadata.

Reviewer stress tests:

- corrupt metadata at test time to measure sensitivity;
- remove sensor name while retaining process/rate/location/unit;
- compare transformer compiler to MLP token compiler;
- compare per-sensor-task prompt table to avoid a weak prompt baseline.

## Experiment 3: Few-Label Adaptation

Purpose:

- Show parameter and label efficiency.

Protocol:

For each target sensor/task:

```text
label budgets = 1%, 5%, 10%, 50%, 100%
```

Compare:

- compiled tokens;
- LoRA/adapters;
- per-sensor prompts;
- CHARM-style text embedding;
- frozen probe.

Metrics:

- task metric at each label budget;
- variance over seeds;
- performance per trainable parameter.

Expected result:

- compiled tokens should win most clearly at 1%-10%.

## Experiment 4: Cross-Dataset Same-Sensor Transfer

Purpose:

- Check whether tokens encode sensor/task structure rather than dataset shortcuts.

Protocols:

1. ECG transfer:
   - source: PTB-XL ECG;
   - target: WESAD ECG and PPG-DaLiA ECG;
   - target uses task-specific head with low-label budgets.

2. ACC transfer:
   - source: WESAD/PPG-DaLiA wrist ACC;
   - target: UCI HAR ACC/Gyro;
   - evaluate few-label adaptation.

3. Optical pulse transfer:
   - source: WESAD BVP;
   - target: PPG-DaLiA PPG;
   - evaluate HR-related representation transfer if preprocessing is valid.

Do not overclaim zero-shot task transfer when label spaces differ. The claim is adaptation-interface transfer.

## Experiment 5: Task Description Matters

Purpose:

- Prove this is task/sensor token compilation, not sensor-only conditioning.

Protocols:

- Same sensor, different tasks:
  - ECG for diagnosis vs stress vs HR context.
  - ACC for stress/activity/HR context.
- Same task, different sensors:
  - WESAD stress across ECG/EDA/BVP/ACC/TEMP/RESP.

Ablations:

- no task description;
- task ID only;
- natural language task phrase embedding;
- structured task fields: task type + target + label space.

Expected result:

- task description should help especially when the same sensor appears in multiple tasks.

## Experiment 5B: Multi-Task Joint Training

Purpose:

- Demonstrate why a compiler is more important than separate PEFT modules.

Protocol:

- Train one shared compiler across at least WESAD + one second dataset.
- Use task-specific heads, but share the compiler and backbone.
- Compare total trainable parameters and performance against:
  - separate LoRA per task;
  - separate adapter per task;
  - separate prompt table per task;
  - no task-token ablation.

Primary claim:

- The shared compiler provides better parameter scaling as the number of tasks/sensors grows.

## Experiment 6: Token Space Structure

Purpose:

- Show learned tokens have meaningful structure.

Analyses:

1. Token retrieval:
   - query token nearest neighbors should share sensor family, measured process, or task type.

2. Distance correlation:
   - correlate token distances with metadata distances.

3. Counterfactual metadata edits:
   - change sampling rate, unit, body location, or task description while keeping signal fixed.
   - observe changes in representation and prediction.

4. Prefix-token attention:
   - inspect attention from task/sensor tokens to signal patches if backbone access allows it.

5. t-SNE/UMAP:
   - use only as visualization, not the main evidence.

## Ablation Matrix

| Ablation | Purpose |
|---|---|
| remove sensor description | tests modality semantics |
| remove measured_process | tests physiological semantics |
| remove sampling rate | tests temporal scale |
| remove unit | tests physical meaning |
| remove body location | tests deployment context |
| remove task description | tests task-token contribution |
| remove sensor ID/name | tests non-lookup behavior |
| metadata only vs metadata+signal summary | tests need for waveform statistics |
| prefix injection vs post-backbone concat | tests mechanism |
| K = 1/4/8/16 tokens | tests token capacity |
| frozen vs lightly adapted backbone | tests complementarity |

## Pilot Priority

Do not start with all datasets. Start with the decisive pilot:

### Week 1

- Download/preprocess real WESAD; synthetic results are only smoke tests.
- Implement true prefix-token injection for MOMENT.
- Keep PatchTransformer only as a debugging scaffold.
- Build WESAD leave-one-sensor-out.
- Run frozen probe, metadata side MLP, MLP token compiler, per-sensor prompt, per-sensor-task prompt, and compiled tokens.

### Week 2

- Add LoRA rank sweep 4/8/16 and at least one stronger PEFT baseline if feasible.
- Add CHARM-style text embedding and Gen-P-Tuning-style baseline.
- Run 1%, 5%, 10%, 100% labels.
- Run no-sensor-ID and no-task-description ablations.

### Week 3

- Add Sleep-EDF or PPG-DaLiA; this is required before ICLR submission.
- Test whether WESAD result transfers to another physiological dataset.
- Start multi-task shared-compiler training.

### Week 4

- Add token-space analysis and counterfactual metadata edits.
- Add statistical tests and confidence intervals.

## Go / No-Go

Proceed toward ICLR only if:

- compiled tokens beat LoRA/per-sensor prompt/per-sensor-task prompt/channel-text/Gen-P baselines in low-label or missing-sensor settings;
- performance does not collapse when sensor ID/name is removed;
- task description contributes beyond sensor metadata;
- prefix injection beats post-backbone metadata concatenation;
- transformer compiler beats MLP token compiler;
- wins appear in at least two dataset families;
- MOMENT results support the claim.

If not, reframe as a narrower biomedical adaptation workshop paper.
