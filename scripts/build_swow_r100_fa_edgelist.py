#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from lwow.fa_cleaning import build_missing_dict, cleaning_pipeline, get_wn_word_set
from lwow.fa_networks import build_filtered_graph, graph_to_csv


DEFAULT_INPUT = "lexicon/SWOW-EN18/SWOW-EN.R100.20180827.csv"
DEFAULT_PROCESSED_OUTPUT = "data/processed/SWOW_EN_R100_fa_processed.csv"
DEFAULT_EDGE_OUTPUT = "lexicon/LWOW_datasets/graphs/edge_lists/FA_SWOW_R100_edgelist.csv"


def _step(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"  {msg}...")


def _done(start_time: float, verbose: bool) -> None:
    if verbose:
        print(f"  done ({time.time() - start_time:.1f}s)")


def _build_raw_like_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        cue = str(getattr(row, "cue", "")).strip()
        r1 = str(getattr(row, "R1", "")).strip()
        r2 = str(getattr(row, "R2", "")).strip()
        r3 = str(getattr(row, "R3", "")).strip()
        parsed = "|".join([x for x in [r1, r2, r3] if x])
        rows.append({"cue": cue, "parsed": parsed})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FA-style processed CSV + edgelist from SWOW R100."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to SWOW R100 CSV.")
    parser.add_argument(
        "--processed-output",
        default=DEFAULT_PROCESSED_OUTPUT,
        help="Path for FA-style processed CSV (cue,R1,R2,R3).",
    )
    parser.add_argument(
        "--edge-output",
        default=DEFAULT_EDGE_OUTPUT,
        help="Path for FA-style edge list CSV (src,tgt,wt).",
    )
    parser.add_argument(
        "--match-lwow",
        action="store_true",
        help="Use direct wn.synsets() checks per node (matches original LWOW exactly, slower).",
    )
    parser.add_argument(
        "--repetitions-per-cue",
        type=int,
        default=100,
        help="Target trials per cue for cue100 alignment.",
    )
    parser.add_argument("--seed", type=int, default=30, help="Seed used by cue100 downsampling.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print step timings.")
    args = parser.parse_args()

    t0 = time.time()
    _step("Reading SWOW R100 CSV", args.verbose)
    source_df = pd.read_csv(args.input)
    required = {"cue", "R1", "R2", "R3"}
    missing = required - set(source_df.columns)
    if missing:
        raise SystemExit(
            f"Input is missing required columns {sorted(missing)}; found: {list(source_df.columns)}"
        )
    source_df = source_df[["cue", "R1", "R2", "R3"]].fillna("")
    _done(t0, args.verbose)

    t0 = time.time()
    _step("Adapting rows to FA cleaner input shape (cue + parsed)", args.verbose)
    rows = _build_raw_like_rows(source_df)
    _done(t0, args.verbose)

    t0 = time.time()
    _step("Loading WordNet cache (missing_dict + wn_word_set)", args.verbose)
    build_missing_dict(use_cache=True)
    wn_word_set = None if args.match_lwow else get_wn_word_set(use_cache=True)
    _done(t0, args.verbose)

    t0 = time.time()
    _step("Running FA cleaning pipeline", args.verbose)
    cleaned_df = cleaning_pipeline(
        rows,
        seed=args.seed,
        repetitions_per_cue=args.repetitions_per_cue,
    )
    _done(t0, args.verbose)

    t0 = time.time()
    _step("Writing processed CSV", args.verbose)
    processed_path = Path(args.processed_output)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    _done(t0, args.verbose)

    t0 = time.time()
    _step("Building FA filtered graph", args.verbose)
    g = build_filtered_graph(
        cleaned_df,
        wn_word_set=wn_word_set,
        use_direct_wn_lookup=args.match_lwow,
    )
    _done(t0, args.verbose)

    t0 = time.time()
    _step("Writing FA-style edge list CSV", args.verbose)
    graph_to_csv(g, args.edge_output)
    _done(t0, args.verbose)

    if args.verbose:
        print(f"Input:     {args.input}")
        print(f"Processed: {args.processed_output}")
        print(f"Edgelist:  {args.edge_output}")


if __name__ == "__main__":
    main()
