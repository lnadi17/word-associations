#!/usr/bin/env python3
"""Build FA graph summaries, comparisons, and exports from processed CSVs.

Expected processed input schema per file: cue,R1,R2,R3
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from lwow.fa_cleaning import get_wn_word_set
from lwow.fa_networks import (
    fa_edge_list,
    graph_from_edge_list,
    graph_summary,
    graph_to_csv,
    idiosyn_filter,
    largest_connected_component,
    make_undirected,
    net_comparison,
    wn_filter,
)


def _dataset_name_from_processed(path: Path) -> str:
    stem = path.stem
    if stem.startswith("FA_"):
        return stem[3:].lower()
    if stem.endswith("_processed"):
        return stem[: -len("_processed")]
    return stem


def _load_processed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cue", "R1", "R2", "R3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df[["cue", "R1", "R2", "R3"]]


def _to_fa_full_graph(df: pd.DataFrame):
    edges = fa_edge_list(df)
    g = graph_from_edge_list(edges, directed=True, weighted=True)
    return make_undirected(g)


def _to_fa_filtered_graph(df: pd.DataFrame, wn_word_set: set | None, match_lwow: bool):
    g = _to_fa_full_graph(df)
    g = wn_filter(g, wn_word_set=wn_word_set, use_direct_lookup=match_lwow)
    g = idiosyn_filter(g)
    g = largest_connected_component(g)
    return g


def _write_removed_nodes_sample(
    full_graphs: dict[str, object],
    filtered_graphs: dict[str, object],
    output_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    cols: dict[str, list[str]] = {}
    for model in full_graphs:
        removed = sorted(set(full_graphs[model].nodes()) - set(filtered_graphs[model].nodes()))
        if not removed:
            sample = []
        elif len(removed) <= sample_size:
            sample = removed
        else:
            sample = rng.sample(removed, sample_size)
        cols[model] = sample

    max_len = max((len(v) for v in cols.values()), default=0)
    for model, vals in cols.items():
        if len(vals) < max_len:
            cols[model] = vals + [""] * (max_len - len(vals))
    df = pd.DataFrame(cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _write_comparison_table(
    filtered_graphs: dict[str, object],
    human_model: str,
    output_path: Path,
) -> None:
    if human_model not in filtered_graphs:
        raise ValueError(
            f"Human reference model '{human_model}' not found in datasets: {list(filtered_graphs.keys())}"
        )

    metric_labels = [
        "Percentage of Human nodes not in LLM network",
        "Percentage of all nodes common to both networks",
        "Percentage of LLM nodes not in Human network",
        "Percentage of Human edges not in LLM network",
        "Percentage of all edges common to both networks",
        "Percentage of LLM edges not in Human network",
    ]
    metric_keys = [
        "(A-B)/A Nodes",
        "Jaccard Nodes",
        "(B-A)/B Nodes",
        "(A-B)/A Edges",
        "Jaccard Edges",
        "(B-A)/B Edges",
    ]

    comp_dict: dict[str, list[float] | list[str]] = {"Comparison with Humans": metric_labels}
    for model, graph in filtered_graphs.items():
        if model == human_model:
            continue
        stats = net_comparison(filtered_graphs[human_model], graph)
        comp_dict[model] = [stats[k] for k in metric_keys]

    df = pd.DataFrame(comp_dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _export_igraph_graphml(g, output_path: Path, weighted: bool = True) -> None:
    try:
        from igraph import Graph  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "python-igraph is required for --export-igraph. Install with: pip install igraph"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    G_ig = Graph(directed=False)
    nodes = list(g.nodes())
    G_ig.add_vertices(nodes)
    G_ig.vs["name"] = nodes
    G_ig.add_edges(list(g.edges()))
    if weighted:
        weights = [g[u][v].get("weight", 1) for u, v in g.edges()]
        G_ig.es["weight"] = weights
    G_ig.write_graphml(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FA graph summaries/comparisons and filtered exports from processed datasets."
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing *_processed.csv files (cue,R1,R2,R3).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset names to include (e.g., humans gpt5_mini gemma). Defaults to all *_processed.csv files.",
    )
    parser.add_argument(
        "--processed-paths",
        nargs="*",
        default=[],
        help="Optional extra processed CSV paths to include (e.g., lexicon/LWOW_datasets/processed_datasets/FA_Mistral.csv).",
    )
    parser.add_argument(
        "--human-model",
        default="humans",
        help="Dataset name to use as human reference for comparison table.",
    )
    parser.add_argument(
        "--summary-dir",
        default="data/summary_tables",
        help="Output directory for summary/comparison CSVs.",
    )
    parser.add_argument(
        "--edge-output-dir",
        default="data/graphs/edge_lists",
        help="Output directory for filtered edge lists.",
    )
    parser.add_argument(
        "--igraph-output-dir",
        default="data/graphs/igraphs",
        help="Output directory for optional igraph GraphML files.",
    )
    parser.add_argument(
        "--removed-sample-size",
        type=int,
        default=20,
        help="How many removed nodes to sample per model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=30,
        help="Seed for removed-node sampling.",
    )
    parser.add_argument(
        "--match-lwow",
        action="store_true",
        help="Use direct wn.synsets() checks per node (slower, matches original script).",
    )
    parser.add_argument(
        "--export-igraph",
        action="store_true",
        help="Also export filtered graphs as igraph GraphML files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise SystemExit(f"Processed directory not found: {processed_dir}")

    all_processed = sorted(processed_dir.glob("*_processed.csv"))
    extra_paths = [Path(p) for p in args.processed_paths]
    for p in extra_paths:
        if not p.exists():
            raise SystemExit(f"Extra processed CSV not found: {p}")

    all_candidates = []
    seen = set()
    for p in all_processed + extra_paths:
        key = str(p.resolve())
        if key not in seen:
            all_candidates.append(p)
            seen.add(key)

    if args.datasets:
        wanted = set(args.datasets)
        selected = [
            p for p in all_candidates if _dataset_name_from_processed(p) in wanted
        ]
    else:
        selected = all_candidates

    if not selected:
        raise SystemExit(f"No processed datasets found in {processed_dir}")

    if args.verbose:
        print("Datasets:")
        for p in selected:
            print(f"  - {_dataset_name_from_processed(p)}: {p}")

    wn_word_set = None if args.match_lwow else get_wn_word_set(use_cache=True)

    full_graphs = {}
    filtered_graphs = {}
    for path in selected:
        name = _dataset_name_from_processed(path)
        if args.verbose:
            print(f"Building graphs for {name}...")
        df = _load_processed_csv(path)
        g_full = _to_fa_full_graph(df)
        g_filt = _to_fa_filtered_graph(df, wn_word_set=wn_word_set, match_lwow=args.match_lwow)
        full_graphs[name] = g_full
        filtered_graphs[name] = g_filt

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    full_summary_path = summary_dir / "FA_graphs_full_summary.csv"
    filt_summary_path = summary_dir / "FA_graphs_filtered_summary.csv"
    removed_nodes_path = summary_dir / "FA_removed_nodes_sample.csv"
    comparisons_path = summary_dir / "FA_graph_comparisons.csv"

    graph_summary(full_graphs).to_csv(full_summary_path)
    graph_summary(filtered_graphs).to_csv(filt_summary_path)
    _write_removed_nodes_sample(
        full_graphs,
        filtered_graphs,
        output_path=removed_nodes_path,
        sample_size=args.removed_sample_size,
        seed=args.seed,
    )
    _write_comparison_table(
        filtered_graphs,
        human_model=args.human_model,
        output_path=comparisons_path,
    )

    edge_output_dir = Path(args.edge_output_dir)
    for model, g in filtered_graphs.items():
        edge_path = edge_output_dir / f"FA_{model}_edgelist.csv"
        graph_to_csv(g, edge_path)

    if args.export_igraph:
        igraph_dir = Path(args.igraph_output_dir)
        for model, g in filtered_graphs.items():
            graphml_path = igraph_dir / f"FA_{model}_ig.graphml"
            _export_igraph_graphml(g, graphml_path, weighted=True)

    if args.verbose:
        print(f"Full summary:      {full_summary_path}")
        print(f"Filtered summary:  {filt_summary_path}")
        print(f"Removed-node list: {removed_nodes_path}")
        print(f"Comparisons:       {comparisons_path}")
        print(f"Filtered edges:    {edge_output_dir}")
        if args.export_igraph:
            print(f"GraphML (igraph):  {args.igraph_output_dir}")


if __name__ == "__main__":
    main()
