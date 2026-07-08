# ICLR Revision Plan After External Reviewer Panel

## Verdict

The direction remains valid and should not be changed:

```text
sensor metadata + task intent
    -> task/sensor token compiler
    -> adaptation tokens
    -> frozen physiological foundation model
```

However, the project is **not ICLR-ready** until the P0 items below are complete. The current PatchTransformer experiment harness is useful for debugging, but it cannot support the paper's main claim by itself.

## What Changed After Review

The main story stays the same. The experimental bar is higher:

1. MOMENT prefix injection is mandatory.
2. WESAD-only is insufficient.
3. Gen-P-Tuning and CHARM-style baselines are mandatory.
4. Multi-task shared compiler is now a central differentiator.
5. The compiler must beat a simple MLP token generator and a per-sensor-task prompt table.

## P0: Required Before ICLR Target

### P0.1 Real MOMENT Prefix Injection

Requirement:

- Expose MOMENT's patch embedding and transformer encoder.
- Insert compiled tokens before the encoder.
- Do not use post-hoc MOMENT features as the main method.

Why:

- The paper claims foundation-model adaptation. A custom PatchTransformer is not enough.

### P0.2 Real WESAD Pilot

Requirement:

- Use real WESAD, not synthetic fallback.
- Subject-level splits.
- Leave-one-sensor-out across ECG, EDA, BVP, ACC, TEMP, RESP.
- Label budgets: 1%, 5%, 10%, 100%.
- 3 seeds first, 5 seeds if promising.

Pass condition:

- Compiler beats metadata side-feature, MLP-token compiler, per-sensor prompt, per-sensor-task prompt, and LoRA in most low-label held-out sensor settings.

### P0.3 Dangerous Baselines

Required:

- CHARM-style channel text embedding.
- Gen-P-Tuning-style prompt adaptation.
- LoRA rank 4/8/16.
- Adapter baseline.
- Per-sensor-task prompt table.
- Metadata MLP -> prefix-token generator.

Strongly recommended:

- FourierFT or another post-LoRA PEFT method if implementation cost is manageable.

### P0.4 Second Dataset

Required before ICLR:

- Sleep-EDF or PPG-DaLiA.

Recommended order:

1. Sleep-EDF if testing multi-physiology channel transfer and NAPS comparison.
2. PPG-DaLiA if testing cardiac/motion/HR regression and sensor subset robustness.

Pass condition:

- The WESAD result pattern replicates in at least one second dataset family.

## P1: Needed For Strong, Not Borderline, ICLR

### P1.1 Multi-Task Shared Compiler

Protocol:

- One shared compiler across WESAD plus Sleep-EDF or PPG-DaLiA.
- Task-specific heads.
- Compare against separate per-task LoRA/adapters and per-task prompts.

Why it matters:

- This is the clearest distinction from ordinary PEFT. PEFT scales linearly with tasks; the compiler should amortize adaptation through metadata.

### P1.2 Meta-Learning Baseline

Add at least one simple baseline:

- ProtoNet-style support/query classifier over frozen features;
- MAML/Reptile only if implementation is practical.

Why:

- Leave-configuration-out training can be dismissed as standard meta-learning unless compared.

### P1.3 Physiological Story

The paper must explain why this is not generic multimodal adaptation:

- physiological processes differ in signal generation mechanism;
- sampling rates span orders of magnitude;
- same latent process can be measured by different sensors, e.g. ECG vs PPG/BVP;
- clinical/wearable deployments see missing sensors and new device configurations.

## Revised Main Tables

### Table 1: WESAD Leave-One-Sensor-Out

Rows:

- Frozen probe
- Metadata side MLP
- MLP token compiler
- Per-sensor prompt
- Per-sensor-task prompt
- CHARM-style
- Gen-P-style
- LoRA r=4/8/16
- Adapter
- Task/Sensor Token Compiler

Columns:

- ECG held out
- EDA held out
- BVP held out
- ACC held out
- TEMP held out
- RESP held out
- average
- trainable parameters

### Table 2: Low-Label Adaptation

Rows: same methods.  
Columns: 1%, 5%, 10%, 100% labels plus trainable params and adaptation time.

### Table 3: Second Dataset Replication

Use Sleep-EDF or PPG-DaLiA. Show the same qualitative pattern as WESAD.

### Table 4: Multi-Task Shared Compiler

Compare:

- one shared compiler + task heads;
- separate LoRA per task;
- separate adapters per task;
- separate prompt tables per task.

Report:

- average task metric;
- total trainable parameters as number of tasks grows;
- low-label target adaptation.

## Ready-to-Run Decision

The direction is ready. The paper is not yet ready.

Proceed with experiments only after treating the current codebase as a scaffold:

1. keep PatchTransformer for smoke tests;
2. implement MOMENT prefix injection;
3. run real WESAD P0 pilot;
4. add Gen-P/CHARM and MLP-token baselines;
5. add second dataset before making any ICLR claim.

If P0 fails, downgrade early. If P0 passes, the project has credible ICLR main-track potential.
