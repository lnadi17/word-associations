"""Build social-prime activation lookups from the local weighted, filtered graphs.

The sparse update is equivalent to run_spreading_activation.R on these
undirected graphs without self-loops. The archived gender matrix remains
the input for the existing gender results.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.models import ARCHIVED_MATRICES, MATRIX_KEYS, MATRIX_MODELS
from wordviz.social_data import STORE_DIR
from wordviz.social_domains import DOMAIN_PRIMES


def spread_graph(graph, primes):
    if graph.is_directed() or not graph.is_simple() or not graph.is_connected():
        raise ValueError("Expected the connected, undirected, simple filtered association graph")
    words = pd.Index(graph.vs["name"], name="node")
    if not words.is_unique:
        raise ValueError("Graph has duplicate word labels")
    steps = 2 * graph.diameter(directed=False, weights=None)
    adjacency = graph.get_adjacency_sparse(attribute="weight")
    if (adjacency.data <= 0).any():
        raise ValueError("Edge weights must be positive")
    strengths = np.asarray(adjacency.sum(axis=1)).reshape(-1, 1)
    if (strengths <= 0).any():
        raise ValueError("Graph contains isolated nodes")
    # spreadr retains half at each source and distributes the other half in
    # proportion to outgoing edge weight, simultaneously for every prime.
    transition = adjacency.multiply(0.5 / strengths).T.tocsr()
    available = [prime for prime in primes if prime in words]
    levels = np.zeros((len(words), len(available)))
    levels[words.get_indexer(available), np.arange(len(available))] = len(words)
    for _ in range(steps):
        levels = 0.5 * levels + transition @ levels
    if not np.allclose(levels.sum(axis=0), len(words), atol=1e-7, rtol=1e-11):
        raise ValueError("Unexpected loss of activation")
    return pd.DataFrame(levels, index=words, columns=available), steps


def raw_domains(model):
    archived_path = ROOT / "data/LDT_analyses/FA_matrices" / f"{model}_gender.csv"
    gender = pd.read_csv(archived_path, keep_default_na=False).set_index("node")
    graph_path = ROOT / "data/graphs/igraphs" / f"FA_{model.lower()}_ig.graphml"
    primes = list(dict.fromkeys(p for group in DOMAIN_PRIMES.values() for p in group))
    levels, steps = spread_graph(ig.Graph.Read_GraphML(str(graph_path)), primes)
    gender = gender.loc[:, gender.ne(0).any()]
    yield from domain_frames(archived_path, gender, graph_path, levels, steps,
                             compare_archive=model in ARCHIVED_MATRICES)


def domain_frames(gender_path, gender, graph_path, levels, steps, *, compare_archive=False):
    yield "gender", gender, {
        "source": gender_path.name, "sha256": hashlib.sha256(gender_path.read_bytes()).hexdigest(),
        "steps": None if compare_archive else steps, "nodes": len(gender),
    }
    source = {
        "source": graph_path.name, "sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "nodes": len(levels), "steps": steps,
    }
    if compare_archive:
        # Some archived networks have a different vertex set from the current graph.
        common = levels.index.intersection(gender.index)
        checked_primes = sorted(set(gender.columns) & set(levels.columns))
        current = levels.loc[common, checked_primes].to_numpy()
        archived = gender.loc[common, checked_primes].to_numpy()
        source["gender_crosscheck"] = {
            "archived_nodes": len(gender), "shared_nodes": len(common),
            "matches_archive": bool(len(common) == len(gender) == len(levels)
                                    and np.allclose(current, archived, rtol=1e-9, atol=1e-10)),
            "max_absolute_difference": float(np.abs(current - archived).max()),
        }
    for domain, group in DOMAIN_PRIMES.items():
        if domain != "gender":
            yield domain, levels[[p for p in group if p in levels.columns]], source.copy()


def build_model(model, output, raw_output=None, *, domains=None):
    metadata = {}
    matrices = []
    for domain, raw, info in (raw_domains(model) if domains is None else domains):
        if raw_output:
            model_dir = raw_output / model
            model_dir.mkdir(parents=True, exist_ok=True)
            raw.to_csv(model_dir / f"{domain}.csv.gz", compression={"method": "gzip", "mtime": 0})
        primes = list(raw.columns)
        info.update(primes=primes, missing_primes=[p for p in DOMAIN_PRIMES[domain] if p not in primes])
        metadata[domain] = info
        if domain == "gender":
            continue  # The experiment store already serves the archived gender values.
        values = raw.to_numpy(dtype=float)
        col_norms = np.linalg.norm(values, axis=0)
        if np.any(col_norms == 0):
            raise ValueError(f"Empty activation column in {model}/{domain}")
        scaled = values / col_norms
        row_norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        scaled = np.divide(scaled, row_norms, out=np.zeros_like(scaled), where=row_norms > 0)
        matrices.append(pd.DataFrame(scaled, index=raw.index, columns=primes))

    packed = pd.concat(matrices, axis=1)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output, delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with closing(sqlite3.connect(path)) as db:
            db.executescript("""
                CREATE TABLE activation(node TEXT PRIMARY KEY, levels BLOB NOT NULL) WITHOUT ROWID;
                CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            db.executemany("INSERT INTO metadata VALUES (?, ?)", [
                ("domains", json.dumps(metadata, allow_nan=False)),
                ("columns", json.dumps(list(packed.columns))),
            ])
            values = packed.to_numpy(dtype="<f8")
            db.executemany("INSERT INTO activation VALUES (?, ?)",
                           ((word, row.tobytes()) for word, row in zip(packed.index, values)))
            db.commit()
            db.execute("VACUUM")
        os.chmod(path, 0o644)
        os.replace(path, output / f"{model}.sqlite")
    finally:
        path.unlink(missing_ok=True)
    print(f"{model}: {len(packed):,} nodes, {info['steps']} steps", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MATRIX_KEYS, default=list(MATRIX_MODELS))
    parser.add_argument("--output", type=Path, default=STORE_DIR)
    parser.add_argument("--raw-output", type=Path, help="Also export full raw matrices for offline analyses.")
    args = parser.parse_args()
    for chosen in args.models:
        build_model(chosen, args.output, args.raw_output)
