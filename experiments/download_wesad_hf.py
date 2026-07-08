"""Download public Hugging Face WESAD shards for real-data smoke tests."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/wesad")
    parser.add_argument("--source", choices=["parquet_0000", "csv_features"], default="parquet_0000")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.source == "parquet_0000":
        download(PARQUET_0000, out_dir / "0000.parquet")
    else:
        download(CSV_FEATURES, out_dir / "WESAD_raw_data.csv")


if __name__ == "__main__":
    main()
