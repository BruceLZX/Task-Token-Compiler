"""Smoke-test MOMENT availability for foundation-model P0 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from momentfm import MOMENTPipeline
    except Exception as exc:  # pragma: no cover - environment smoke test.
        raise RuntimeError(
            "Could not import momentfm. MOMENT official package currently "
            "requires a compatible Python/runtime stack."
        ) from exc

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = MOMENTPipeline.from_pretrained(
        args.model_id,
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(device)
    model.eval()

    x = torch.randn(2, 1, 512, device=device)
    with torch.no_grad():
        try:
            output = model(x_enc=x)
        except TypeError:
            output = model(x)

    if hasattr(output, "embeddings"):
        emb = output.embeddings
    elif hasattr(output, "last_hidden_state"):
        emb = output.last_hidden_state
    elif isinstance(output, dict):
        emb = output.get("embeddings") or output.get("last_hidden_state")
    else:
        emb = output[0] if isinstance(output, (tuple, list)) else output

    if not torch.is_tensor(emb):
        raise RuntimeError(f"Could not find tensor embedding in MOMENT output: {type(output)}")

    report = {
        "model_id": args.model_id,
        "device": str(device),
        "embedding_shape": list(emb.shape),
        "embedding_mean": float(emb.float().mean().cpu()),
        "embedding_std": float(emb.float().std().cpu()),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
