"""Align weighted GraphML edges with the deployed path graph for arbitrary-prime activation."""

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from itertools import chain
from pathlib import Path

import igraph as ig
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.custom_activation import WEIGHTS_DIR
from wordviz.models import MATRIX_KEYS
from wordviz.path_data import GRAPH_DIR, load_graph
from wordviz.social_data import domain_info


def build(model, destination):
    # All four non-gender preset stores use this exact GraphML snapshot. Reuse
    # its verified step count so the user-defined domain runs the same experiment.
    info = domain_info("religion", model)
    graphml = ROOT / "data/graphs/igraphs" / info["source"]
    if hashlib.sha256(graphml.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError(f"Refresh the published social store before building weights for {model}")
    graph = ig.Graph.Read_GraphML(str(graphml))
    if graph.is_directed() or not graph.is_simple() or not graph.is_connected():
        raise ValueError("Expected a simple, connected and undirected association graph")
    key = model.lower()
    path = load_graph(key)
    index = {word: i for i, word in enumerate(graph.vs["name"])}
    if len(index) != graph.vcount() or index.keys() != path["index"].keys():
        raise ValueError(f"Word labels differ between the two snapshots for {model}")
    order = [index[word] for word in path["words"]]
    adjacency = graph.get_adjacency_sparse(attribute="weight")[order, :][:, order].tocsr()
    adjacency.sort_indices()
    counts = np.fromiter(map(len, path["neighbors"]), dtype=int)
    neighbors = np.fromiter(chain.from_iterable(path["neighbors"]), dtype=int)
    if not (np.array_equal(counts, np.diff(adjacency.indptr)) and np.array_equal(neighbors, adjacency.indices)):
        raise ValueError(f"Edges differ between the two snapshots for {model}")
    weights = adjacency.data
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Expected positive finite edge weights")
    # Small integer weights compress efficiently. Preserve arbitrary weighted
    # inputs losslessly when a future graph uses fractional edge weights.
    dtype = np.uint8 if np.all(weights == weights.astype(np.uint8)) else "<f8"
    weights = weights.astype(dtype)
    header = {k: info[k] for k in ("source", "sha256", "steps")}
    header.update(dtype=weights.dtype.str, topology_sha256=hashlib.sha256(
        (GRAPH_DIR / f"{key}.json.gz").read_bytes()).hexdigest())
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{key}.gz"
    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as temporary:
        name = Path(temporary.name)
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=temporary, mtime=0) as gz:
                gz.write(json.dumps(header, sort_keys=True).encode() + b"\n")
                gz.write(weights.tobytes())
            temporary.flush()
            os.chmod(name, 0o644)
            os.replace(name, output)
        finally:
            name.unlink(missing_ok=True)
    print(f"{model}: {graph.vcount():,} nodes, {graph.ecount():,} edges, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MATRIX_KEYS)
    parser.add_argument("--output", type=Path, default=WEIGHTS_DIR)
    args = parser.parse_args()
    for model in args.models or [key for key in MATRIX_KEYS if (WEIGHTS_DIR.parent / "social" / f"{key}.sqlite").is_file()]:
        build(model, args.output)
