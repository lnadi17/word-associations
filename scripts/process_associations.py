from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from lwow.config import load_config
from lwow.fa_cleaning import build_missing_dict, cleaning_pipeline, get_wn_word_set
from lwow.fa_networks import build_filtered_graph, graph_to_csv
from lwow.io import read_csv


def _step(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"  {msg}...")


def _done(elapsed: float, verbose: bool) -> None:
    if verbose:
        print(f"  done ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw association outputs (LWOW-aligned).")
    parser.add_argument("--config", help="Path to YAML config (required unless --from-processed).")
    parser.add_argument("--raw-override", help="Override raw input path (e.g. for backup files).")
    parser.add_argument(
        "--from-processed",
        metavar="PATH",
        help="Build edgelist from existing processed CSV (cue,R1,R2,R3). Skips cleaning.",
    )
    parser.add_argument(
        "--match-lwow",
        action="store_true",
        help="Use direct wn.synsets() per node (matches original LWOW exactly, slower).",
    )
    parser.add_argument(
        "--edge-output",
        metavar="PATH",
        help="Output path for edgelist (default from config or required with --from-processed).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print step-by-step timing.")
    args = parser.parse_args()

    if args.from_processed:
        _run_from_processed(args)
    else:
        if not args.config:
            parser.error("--config is required unless --from-processed is used")
        _run_full_pipeline(args)


def _run_from_processed(args: argparse.Namespace) -> None:
    """Build edgelist from existing processed CSV."""
    if not args.edge_output:
        raise SystemExit("--edge-output is required when using --from-processed")
    edge_path = args.edge_output

    t0 = time.time()
    _step("Reading processed CSV", args.verbose)
    df = pd.read_csv(args.from_processed)
    _done(time.time() - t0, args.verbose)

    wn_word_set = None
    if not args.match_lwow:
        t0 = time.time()
        _step("Loading WordNet cache (wn_word_set)", args.verbose)
        wn_word_set = get_wn_word_set(use_cache=True)
        _done(time.time() - t0, args.verbose)
    else:
        _step("Using direct WordNet lookup (--match-lwow)", args.verbose)
        if args.verbose:
            print("  (skips cache, matches original LWOW script)")

    t0 = time.time()
    _step("Building filtered graph (edges, WN filter, idiosyn, LCC)", args.verbose)
    g = build_filtered_graph(df, wn_word_set=wn_word_set, use_direct_wn_lookup=args.match_lwow)
    _done(time.time() - t0, args.verbose)

    t0 = time.time()
    _step("Writing edge list CSV", args.verbose)
    graph_to_csv(g, edge_path)
    _done(time.time() - t0, args.verbose)

    if args.verbose:
        print(f"Edges: {edge_path}")


def _run_full_pipeline(args: argparse.Namespace) -> None:
    """Full pipeline: raw CSV -> cleaning -> processed CSV -> graph -> edgelist."""
    config = load_config(args.config)
    raw_path = args.raw_override or config.paths.raw_output_path

    t0 = time.time()
    _step("Reading raw CSV", args.verbose)
    rows = read_csv(raw_path)
    _done(time.time() - t0, args.verbose)

    spelling_path = config.paths.spelling_dict_path
    if spelling_path and not Path(spelling_path).exists():
        spelling_path = None

    t0 = time.time()
    _step("Loading WordNet cache (missing_dict + wn_word_set)", args.verbose)
    build_missing_dict(use_cache=True)
    wn_word_set = get_wn_word_set(use_cache=True)
    _done(time.time() - t0, args.verbose)

    t0 = time.time()
    _step("Cleaning pipeline (normalize, cue100, etc.)", args.verbose)
    df = cleaning_pipeline(
        rows,
        spelling_dict_path=spelling_path or None,
        seed=config.processing.seed,
        repetitions_per_cue=config.processing.repetitions_per_cue,
    )
    _done(time.time() - t0, args.verbose)

    t0 = time.time()
    _step("Writing processed CSV", args.verbose)
    Path(config.paths.processed_output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.paths.processed_output_path, index=False)
    _done(time.time() - t0, args.verbose)

    t0 = time.time()
    _step("Building filtered graph (edges, WN filter, idiosyn, LCC)", args.verbose)
    g = build_filtered_graph(df, wn_word_set=wn_word_set)
    _done(time.time() - t0, args.verbose)

    edge_path = args.edge_output or config.paths.edge_list_output_path
    t0 = time.time()
    _step("Writing edge list CSV", args.verbose)
    graph_to_csv(g, edge_path)
    _done(time.time() - t0, args.verbose)

    if args.verbose:
        print(f"Output: {config.paths.processed_output_path}")
        print(f"Edges:  {edge_path}")


if __name__ == "__main__":
    main()
