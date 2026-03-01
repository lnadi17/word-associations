#!/usr/bin/env python3
"""Compare our edgelist generation to official LWOW edgelists.

Usage:
  PYTHONPATH=. python scripts/compare_edgelists.py [--datasets Haiku Llama3 Mistral]

Runs the pipeline on official processed CSVs and diffs against official edgelists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "lexicon/LWOW_datasets/processed_datasets"
EDGELISTS = PROJECT_ROOT / "lexicon/LWOW_datasets/graphs/edge_lists"

DATASETS = ["Haiku", "Llama3", "Mistral"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare our edgelist output to official LWOW.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help=f"Datasets to compare (default: {' '.join(DATASETS)})",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print summary.")
    args = parser.parse_args()

    results = []
    for name in args.datasets:
        proc = PROCESSED / f"FA_{name}.csv"
        official = EDGELISTS / f"FA_{name}_edgelist.csv"
        if not proc.exists():
            print(f"SKIP {name}: processed CSV not found: {proc}", file=sys.stderr)
            continue
        if not official.exists():
            print(f"SKIP {name}: official edgelist not found: {official}", file=sys.stderr)
            continue

        ours_path = str(PROJECT_ROOT / ".tmp_compare_ours" / f"FA_{name}_ours.csv")
        Path(ours_path).parent.mkdir(parents=True, exist_ok=True)

        r = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/process_associations.py"),
                "--from-processed",
                str(proc),
                "--edge-output",
                ours_path,
                "--match-lwow",
                *(["-v"] if not args.quiet else []),
            ],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(f"FAIL {name}: pipeline failed (code {r.returncode})\n{r.stderr}", file=sys.stderr)
            results.append((name, "FAIL", -1, -1))
            Path(ours_path).unlink(missing_ok=True)
            continue

        # Compare edge sets (skip headers)
        with open(ours_path) as f:
            next(f)  # skip header
            ours_set = set(line.strip() for line in f if line.strip())
        with open(official) as f:
            next(f)  # skip header
            off_set = set(line.strip() for line in f if line.strip())
        ours_lines = len(ours_set)
        off_lines = len(off_set)
        same = ours_set == off_set

        Path(ours_path).unlink(missing_ok=True)

        if same:
            results.append((name, "IDENTICAL", ours_lines, off_lines))
        else:
            results.append((name, "DIFF", ours_lines, off_lines))

    # Summary table
    print("\nDataset   | Ours    | Official | Status")
    print("-" * 45)
    for name, status, ours, off in results:
        ours_s = str(ours) if ours >= 0 else "-"
        off_s = str(off) if off >= 0 else "-"
        print(f"{name:8} | {ours_s:>7} | {off_s:>8} | {status}")
    print()


if __name__ == "__main__":
    main()
