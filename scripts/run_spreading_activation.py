"""Compute the LDT, gender and social-prime activations together on a filtered graph.

Uses the sparse, double-precision implementation of the weighted spreadr walk
in build_social_store. Models are named by their GraphML filename's dataset key.
"""
import argparse
import os
import tempfile
from pathlib import Path
from time import monotonic

import igraph as ig
import pandas as pd

from build_social_store import ROOT, STORE_DIR, DOMAIN_PRIMES, build_model, domain_frames, spread_graph
from wordviz.models import ARCHIVED_MATRICES


def write_matrix(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8",
                                     newline="") as stream:
        temporary = Path(stream.name)
        try:
            frame.to_csv(stream)
            stream.flush()
            temporary.chmod(0o644)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def build(model, graph_dir, analyses, social_output, raw_output):
    start = monotonic()
    archived = model.lower() in {key.lower() for key in ARCHIVED_MATRICES}
    if archived and (analyses.resolve() == ROOT / "data/LDT_analyses" or social_output.resolve() == STORE_DIR):
        raise ValueError(f"{model} has archived matrices. Use separate --analyses-dir and --social-output "
                         "directories for a fresh calculation on the current graph.")
    graph_path = graph_dir / f"FA_{model.lower()}_ig.graphml"
    graph = ig.Graph.Read_GraphML(str(graph_path))
    print(f"{model}: {graph.vcount():,} nodes, {graph.ecount():,} edges. Computing activation.", flush=True)
    tasks = {task: list(dict.fromkeys(pd.read_csv(analyses / filename)[column].str.strip().str.lower()))
             for task, filename, column in (("LDT", "LDT_primes.csv", "prime"),
                                            ("gender", "gender_primes.csv", "genderWord"))}
    primes = list(dict.fromkeys(p for group in (*tasks.values(), *DOMAIN_PRIMES.values()) for p in group))
    levels, steps = spread_graph(graph, primes)
    print(f"{model}: {len(levels.columns)}/{len(primes)} primes, {steps} steps, "
          f"{monotonic()-start:.1f}s including exact diameter.", flush=True)
    matrix_dir = analyses / "FA_matrices"
    for task, group in tasks.items():
        # R's archived matrix layout reserves an all-zero column for each absent
        # prime. The statistical analysis excludes trials using these primes.
        frame = levels.reindex(columns=group, fill_value=0)
        write_matrix(frame, matrix_dir / f"{model}_{task}.csv")
        missing = [p for p in group if p not in levels]
        if missing:
            print(f"{model}/{task} missing {len(missing)} primes: {', '.join(missing)}", flush=True)
    gender = levels[[p for p in tasks["gender"] if p in levels]]
    domains = domain_frames(matrix_dir / f"{model}_gender.csv", gender, graph_path, levels, steps)
    build_model(model, social_output, raw_output, domains=domains)
    print(f"{model}: matrices and stores finished in {monotonic()-start:.1f}s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Matrix file stems (graph keys are lowercase)")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/graphs/igraphs")
    parser.add_argument("--analyses-dir", type=Path, default=ROOT / "data/LDT_analyses")
    parser.add_argument("--social-output", type=Path, default=STORE_DIR)
    parser.add_argument("--raw-output", type=Path, help="Also save raw social matrices for norm analyses")
    args = parser.parse_args()
    for model in args.models:
        build(model, args.graph_dir, args.analyses_dir, args.social_output, args.raw_output)
