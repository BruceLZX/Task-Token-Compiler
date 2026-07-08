# Landscape Report: Task/Sensor Token Compiler

**Date:** 2026-07-06  
**Direction:** sensor/task metadata -> adaptation tokens -> frozen physiological foundation model  
**Target:** ICLR main only if experiments validate compositional generalization and strong baselines.

## 1. Short Verdict

The original idea is still the right one:

> Build a token compiler that maps sensor configuration and task intent into adaptation tokens for frozen or lightly adapted foundation models.

The idea should not be diluted into a pure benchmark or schema paper. However, to survive ICLR review, the experiments must prove that the compiler is not just:

- ordinary prompt tuning;
- metadata concatenation;
- a sensor-ID lookup table;
- LoRA with fewer parameters;
- a dataset shortcut.

The strongest framing is:

> Task/sensor tokens are a structured adaptation interface for heterogeneous physiological foundation models.

## 2. Why This Is a Real Gap

Wearable and biomedical time series are heterogeneous in ways that ordinary TSFM adaptation does not explicitly model:

- physiological process: electrical cardiac, electrical brain, optical pulse, skin conductance, motion;
- sampling rate: 4 Hz to 700 Hz in the proposed datasets;
- unit and amplitude scale: mV, uV, uS, g, rad/s, arbitrary optical;
- body location: chest, wrist, scalp, eye, chin, waist;
- task intent: stress, sleep stage, diagnosis, HR, activity.

The proposed compiler uses these fields as inputs to generate adaptation tokens. This is more structured than "learn a prompt for dataset X" or "train a LoRA module for sensor Y."

## 3. Related Work

### Task Tokens for Behavior Foundation Models

Task Tokens propose learned extra tokens for adapting frozen behavior foundation models to control tasks. This validates the general token-adaptation concept.

Relevance:

- We can borrow the foundation-model adaptation interface.
- We must differentiate by domain and compiler input: physiological sensor/task metadata, not behavior-task RL feedback.

### CHARM: Channel Descriptions for Time Series

CHARM uses channel-level textual descriptions to improve multivariate time-series representation and channel-order invariance.

Relevance:

- Strong evidence that channel semantics matter.
- Direct baseline: channel text embedding or description-aware conditioning.

Difference:

- CHARM is a channel-description representation model.
- Our method is a compiler that turns structured sensor/task metadata into adaptation tokens for a frozen TSFM/audio FM.

### NormWear

NormWear argues that wearable physiological foundation models must handle heterogeneous sensor configurations and applications.

Relevance:

- Strong motivation for the problem.
- Demonstrates that biomedical/wearable heterogeneity is central, not incidental.

Difference:

- NormWear trains a wearable foundation model.
- We adapt existing frozen foundation models through a lightweight token interface.

### Gen-P-Tuning

Gen-P-Tuning adapts frozen univariate TSFMs to multivariate healthcare time series.

Relevance:

- Healthcare prompt adaptation is already a live research area.
- It should be treated as a serious baseline or design reference.

Difference:

- Our focus is sensor/task metadata compilation and heterogeneous physiological sensor adaptation.

### Sensor-Prompt Tuning

Sensor-Prompt Tuning adapts MOMENT to motion sensor HAR using convolution-based filters and gating as soft prompts.

Relevance:

- This is a direct warning: "sensor prompt tuning" alone is not enough as novelty.

Difference:

- SPT focuses on motion sensors/HAR and filter-style sensor prompts.
- Our target is heterogeneous biomedical physiology and task/sensor metadata-to-token compilation.

### Sensor-Language Models

SensorLLM and SensorLM show growing interest in aligning sensor data with language and semantic descriptions.

Relevance:

- They support the broader trend that sensor semantics matter.

Difference:

- Our model does not need to train a sensor-language foundation model. It compiles compact task/sensor tokens for existing time-series/audio FMs.

## 4. Competitive Position

| Method family | What it does | Why compiler can be better |
|---|---|---|
| Frozen probe | Uses fixed representations | No sensor/task conditioning |
| LoRA / adapter | Adds trainable modules | Often tied to dataset/sensor/task; weak compositionality |
| Per-sensor prompt | Learns a prompt per sensor | Lookup table; poor unseen combination generalization |
| Channel text embedding | Uses text semantics | May depend on text encoder/free text; not necessarily adaptation tokens |
| Gen-P-Tuning / SPT | Prompt-style TSFM adaptation | Strong baselines, but less focused on structured physiological sensor/task metadata |
| NormWear | Trains wearable FM | Expensive; not an adaptation interface for existing FMs |
| Task/Sensor Token Compiler | Compiles sensor/task metadata into tokens | Structured, parameter-efficient, compositional interface |

## 5. Refined Novelty Claim

Avoid:

> We add learnable prompts for sensors.

Use:

> We introduce a sensor/task token compiler that maps structured physiological sensor metadata and task intent into adaptation tokens for frozen foundation models, enabling parameter-efficient and compositional adaptation across heterogeneous biomedical time series.

The word "compiler" must mean something concrete:

- shared function from metadata fields to tokens;
- supports unseen or missing sensor combinations;
- uses task description, not only sensor ID;
- produces tokens injected into the model sequence, not merely appended as features after encoding.

## 6. What the Paper Must Prove

### P1. Tokens are useful

Compiled tokens beat frozen probe and random-token controls.

### P2. Compiler is better than lookup prompts

Compiled tokens beat per-sensor/per-task learned prompts, especially on unseen combinations or low-label settings.

### P3. Compiler is competitive with PEFT

Compiled tokens beat or match LoRA/adapters with fewer trainable parameters, especially under 1%-10% labels.

### P4. Tokens are structured

Token space reflects sensor/task metadata and counterfactual metadata edits produce meaningful representation changes.

### P5. Not dataset shortcut

Results hold under subject-level splits and at least one cross-dataset same-sensor transfer.

## 7. Recommended Experimental Package

### Minimum viable ICLR pilot

1. WESAD leave-one-sensor-out.
2. WESAD unseen sensor combinations.
3. 1%, 5%, 10%, 100% label budgets.
4. Baselines: frozen probe, metadata side MLP, per-sensor prompt, CHARM-style text, LoRA.
5. Ablations: no task description, no sensor ID, prefix vs post-backbone concat.

### Strong paper expansion

1. Sleep-EDF sleep staging.
2. PPG-DaLiA HR regression.
3. PTB-XL -> WESAD/PPG-DaLiA ECG transfer.
4. UCI HAR IMU adaptation with SPT-style baseline.
5. Token-space structure analysis.

## 8. Risks

| Risk | Severity | Mitigation |
|---|---:|---|
| Reviewer says "just prompt tuning" | high | emphasize compiler, task+sensor metadata, unseen combinations, strong prompt baselines |
| Sensor ID dominates | high | remove-ID ablation, field dropout, measured_process metadata |
| Metadata side feature works equally well | high | compare post-backbone concat vs true prefix injection |
| WESAD too small | medium | add Sleep-EDF or PPG-DaLiA before paper commitment |
| LoRA dominates | high | focus low-label/missing-sensor settings or stop ICLR plan |
| Task description adds nothing | medium | revise method to better encode task semantics or reduce claim |

## 9. Final Recommendation

Keep the original idea. Strengthen it as:

```text
Task/Sensor Token Compiler
= a structured metadata-to-token adaptation interface
for frozen physiological foundation models.
```

Do not make the paper mainly about a new benchmark. Use missing-sensor, unseen-combination, low-label, and cross-dataset protocols as evidence that the compiler is a real interface rather than a prompt trick.

If the compiler beats LoRA/per-sensor prompts/channel text in these settings, this can be a serious ICLR submission. If it only improves over frozen probes, it is not enough.
