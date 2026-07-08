"""Download public Hugging Face WESAD files for real-data experiments."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


PARQUET_REPO = "LouisSimon/wesad-parquet"
PARQUET_PRESETS = {
    "small": ["wesad_4_3_120.parquet"],
    "medium": ["wesad_60_3_120.parquet"],
    "full": ["wesad_60_0.3_120.parquet", "wesad_60_3_120.parquet", "wesad_4_3_120.parquet"],
}
PARQUET_0000 = "https://huggingface.co/datasets/LouisSimon/wesad-parquet/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
CSV_FEATURES = "https://huggingface.co/datasets/GeorgiaCh96/WESAD_raw_data/resolve/main/WESAD_raw_data.csv"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"exists: {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    print(f"done: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def download_parquet_preset(preset: str, out_dir: Path) -> None:
    files = PARQUET_PRESETS[preset]
    available = set(HfApi().list_repo_files(PARQUET_REPO, repo_type="dataset"))
    for filename in files:
        if filename not in available:
            raise FileNotFoundError(f"{filename} not found in dataset {PARQUET_REPO}")
        dest = out_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        cached = hf_hub_download(repo_id=PARQUET_REPO, repo_type="dataset", filename=filename)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        print(f"done: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/wesad")
    parser.add_argument(
        "--source",
        choices=["parquet_small", "parquet_medium", "parquet_full", "parquet_0000", "csv_features"],
        default="parquet_medium",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.source == "parquet_small":
        download_parquet_preset("small", out_dir)
    elif args.source == "parquet_medium":
        download_parquet_preset("medium", out_dir)
    elif args.source == "parquet_full":
        download_parquet_preset("full", out_dir)
    elif args.source == "parquet_0000":
        download(PARQUET_0000, out_dir / "0000.parquet")
    else:
        download(CSV_FEATURES, out_dir / "WESAD_raw_data.csv")


if __name__ == "__main__":
    main()
