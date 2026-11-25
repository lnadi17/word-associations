"""Spreading activation for user-selected graph nodes on the packaged networks."""

import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix

from .models import MATRIX_MODELS
from .path_data import GRAPH_DIR, load_graph

WEIGHTS_DIR = GRAPH_DIR.parent / "social_weights"
MAX_PRIMES = 12


@dataclass
class _Network:
    words: tuple
    index: dict
    incoming: csr_matrix
    info: dict


@lru_cache(maxsize=2)
def _network(model):
    if model not in MATRIX_MODELS:
        raise ValueError("Unknown activation network")
    key = model.lower()
    with gzip.open(WEIGHTS_DIR / f"{key}.gz", "rb") as stream:
        info = json.loads(stream.readline())
        weights = np.frombuffer(stream.read(), dtype=info["dtype"]).astype(np.float64)
    snapshot = GRAPH_DIR / f"{key}.json.gz"
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != info["topology_sha256"]:
        raise ValueError(f"Graph topology and activation weights have different versions for {model}")
    graph = load_graph(key)
    words = tuple(graph["words"])
    neighbors = graph["neighbors"]
    counts = np.fromiter(map(len, neighbors), dtype=np.int32, count=len(words))
    offsets = np.empty(len(words) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    if offsets[-1] != weights.size or np.any(counts == 0):
        raise ValueError(f"Inconsistent neighbor and weight counts for {model}")
    sources = np.fromiter(chain.from_iterable(neighbors), dtype=np.int32, count=weights.size)
    strengths = np.add.reduceat(weights, offsets[:-1])
    # Rows are destinations. Each source sends half its activation in proportion
    # to the weights of its outgoing edges on this undirected graph.
    weights *= 0.5 / strengths[sources]
    incoming = csr_matrix((weights, sources, offsets), shape=(len(words), len(words)))
    return _Network(words, graph["index"], incoming, info)


def _checked_primes(primes):
    if isinstance(primes, str) or not all(isinstance(p, str) for p in primes):
        raise ValueError("Primes must be a sequence of words")
    unique = tuple(dict.fromkeys(primes))
    if len(unique) > MAX_PRIMES:
        raise ValueError(f"Use up to {MAX_PRIMES} distinct primes")
    return unique


@lru_cache(maxsize=64)
def _raw_column(model, prime):
    network = _network(model)
    levels = np.zeros(len(network.words), dtype=np.float64)
    levels[network.index[prime]] = len(levels)
    for _ in range(network.info["steps"]):
        levels = 0.5 * levels + network.incoming @ levels
    return levels


def raw_columns(model, primes):
    """Return graph words, present primes in requested order and raw node × prime data."""
    requested = _checked_primes(primes)
    network = _network(model)
    present = tuple(p for p in requested if p in network.index)
    columns = (np.column_stack([_raw_column(model, prime) for prime in present])
               if present else np.empty((len(network.words), 0)))
    return network.words, present, columns


def custom_info(primes, model):
    requested = _checked_primes(primes)
    network = _network(model)
    return {
        "source": network.info["source"], "sha256": network.info["sha256"],
        "steps": network.info["steps"], "nodes": len(network.words),
        "primes": [p for p in requested if p in network.index],
        "missing_primes": [p for p in requested if p not in network.index],
    }


def custom_values(primes, words, models):
    """Apply column L2 over all graph nodes, then row L2 across the domain's primes."""
    result = {}
    for model in dict.fromkeys(models):
        _, available, raw = raw_columns(model, primes)
        index = _network(model).index
        targets = [word for word in dict.fromkeys(words) if word in index]
        values = raw[[index[word] for word in targets], :] / np.linalg.norm(raw, axis=0)
        lengths = np.linalg.norm(values, axis=1, keepdims=True)
        values = np.divide(values, lengths, out=np.zeros_like(values), where=lengths > 0)
        result[model] = {target: dict(zip(available, row.tolist())) for target, row in zip(targets, values)}
    return result
