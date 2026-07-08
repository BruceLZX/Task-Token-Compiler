# ICLR Strong-Paper Criteria

This file defines the bar for continuing this project as an ICLR main-track target. The goal is not a borderline "works on some datasets" paper. The goal is a paper with a clear adaptation-interface contribution and evidence that the idea matters.

## Non-Negotiable Positioning

The paper must be framed as:

> A task/sensor token compiler: a structured metadata-to-token adaptation interface for frozen physiological foundation models.

It must not be framed as:

- "we add prompts to MOMENT";
- "we concatenate metadata";
- "we build a wearable foundation model";
- "we benchmark several biomedical datasets."

The key intellectual contribution is the interface:

```text
sensor metadata + task intent -> adaptation tokens -> frozen foundation model behavior
```

## What Makes It ICLR-Level

At least two of the following three must be strong.

### 1. Algorithmic Contribution

The method must be more than a learned prompt table.

Required components:

- a shared compiler from structured sensor/task metadata to tokens;
- true prefix or layer-wise token injection into the backbone;
- leave-sensor or leave-configuration episodic training;
- field dropout or sensor-name dropout to prevent lookup behavior;
- token-structure regularization or contrastive alignment with sensor/task semantics.
- a simple MLP-token-compiler baseline proving the transformer compiler is not over-engineering.

Optional but high-value:

- signal-summary-conditioned tokens;
- layer-wise tokens instead of single prefix;
- compiler + LoRA complementarity, showing the interface can compose with PEFT.

### 2. Empirical Contribution

The results must show a regime where existing methods are structurally weaker.

Required wins:

- better than frozen probe and metadata side-feature MLP;
- better than per-sensor learned prompts on unseen sensor combinations;
- better than per-sensor-task prompt tables;
- better than or competitive with LoRA/adapters under 1%, 5%, and 10% labels;
- better than CHARM-style channel text or Gen-P-Tuning/SPT-style baselines where applicable.
- competitive against stronger PEFT beyond plain LoRA where feasible, e.g. FourierFT/DoRA/VeRA.

Required coverage:

- at least 2 dataset families;
- at least 3 held-out sensor/configuration protocols;
- true MOMENT prefix-injection results;
- subject-level splits where subjects exist;
- 5 seeds for main results;
- trainable parameter and wall-clock reporting.

### 3. Mechanistic Evidence

The paper must prove the tokens encode structure.

Required analyses:

- remove sensor ID/name without collapse;
- remove task description and show measurable loss where tasks differ;
- prefix injection beats post-backbone concat;
- token distance correlates with metadata distance;
- counterfactual metadata edits change representations predictably;
- token retrieval recovers related sensors/tasks.

## Strong Acceptance Story

A strong version of the paper should be able to say:

1. Physiological data heterogeneity is a foundation-model adaptation problem, not merely a dataset preprocessing problem.
2. Ordinary PEFT learns separate target-specific parameters, while a compiler learns a reusable metadata-to-token map.
3. The compiler is trained explicitly for missing/unseen configuration generalization.
4. Across stress, sleep, ECG/PPG/IMU tasks, compiled tokens are more label-efficient and parameter-efficient than LoRA, prompts, and channel text baselines.
5. Learned tokens have measurable physiological/task structure.

## Kill Criteria

Stop targeting ICLR main if any of these happen:

- compiled tokens only beat frozen probes;
- metadata side-feature MLP matches compiled tokens;
- MLP token compiler matches transformer compiler everywhere;
- per-sensor learned prompts match compiled tokens on unseen combinations;
- per-sensor-task prompt tables match compiled tokens under fair parameter budgets;
- LoRA dominates in low-label settings;
- removing sensor ID/name collapses performance;
- token space has no measurable relation to metadata/task structure;
- results only work on WESAD;
- MOMENT prefix-injection results do not support the PatchTransformer findings;
- subject leakage or preprocessing shortcuts explain the gains.

## Minimum Result Table for ICLR Submission

Main table:

| Setting | Frozen | Metadata side | Prompt | CHARM-style | LoRA | Compiler |
|---|---:|---:|---:|---:|---:|---:|
| WESAD leave-one-sensor-out | | | | | | |
| WESAD unseen sensor combination | | | | | | |
| Sleep-EDF channel/sensor transfer | | | | | | |
| PPG-DaLiA sensor subset HR | | | | | | |
| ECG/ACC cross-dataset low-label transfer | | | | | | |

Efficiency table:

| Method | Trainable params | Adaptation time | Memory | 1% labels | 10% labels | 100% labels |
|---|---:|---:|---:|---:|---:|---:|

Ablation table:

| Variant | Main metric | Drop vs full | Interpretation |
|---|---:|---:|---|
| full compiler | | | |
| no task description | | | |
| no sensor name/ID | | | |
| no sampling rate | | | |
| no body location | | | |
| no unit | | | |
| post-backbone concat | | | |
| random tokens | | | |

## Recommended First Pilot

Do not spend months before knowing if this can be strong.

Pilot:

- WESAD leave-one-sensor-out;
- label budgets: 1%, 5%, 10%, 100%;
- baselines: frozen probe, metadata side MLP, MLP token compiler, per-sensor prompt, per-sensor-task prompt, LoRA rank sweep, CHARM-style text, Gen-P-style prompt;
- ablations: no sensor ID/name, no task description, prefix vs concat;
- backbone: true MOMENT prefix injection, not only custom PatchTransformer;
- 3 seeds first, then 5 seeds if promising.

Pass threshold:

- compiler beats LoRA or per-sensor prompt in most low-label held-out sensor settings;
- compiler beats metadata side-feature MLP clearly;
- no-ID ablation remains competitive;
- wins are not isolated to one held-out sensor.

If this pilot passes, expand. If it fails, do not force the ICLR story.
