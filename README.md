# Task/Sensor Token Compiler

Research code and planning documents for **Task Tokens for Physiological Foundation Models: Compiling Sensor Metadata into Adaptation Tokens**.

Core idea:

```text
sensor metadata + task intent -> compiled adaptation tokens -> frozen physiological/time-series foundation model
```

See:

- `research/iclr_revision_plan.md` for the ICLR readiness plan.
- `research/experiment_design.md` for the experimental protocol.
- `experiments/README.md` for runnable experiment commands.

Current code focuses on a reviewer-facing WESAD pilot scaffold with baselines for metadata side features, prompt lookup tables, LoRA/adapters, and compiled prefix tokens. PatchTransformer is included for smoke tests; paper-grade experiments must use true MOMENT prefix injection.
