"""FA-style network building for free association data.

Ported from LWOW reproducibility scripts (FA_Functions, FA_build_Networks)
to produce filtered edge lists (src,tgt,wt) with WordNet nodes, weight>1, and LCC.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import pandas as pd

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None  # type: ignore


def fa_edge_list(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Extract (cue, response) edges from cue,R1,R2,R3 dataframe. No blanks, no self-loops."""
    df = df.copy()
    for col in ["cue", "R1", "R2", "R3"]:
        if col == "cue":
            df[col] = [str(x) for x in df[col]]
        else:
            df[col] = ["" if pd.isna(x) else str(x) for x in df[col]]
    df = df[df["cue"] != ""]
    edges = (
        list(zip(df["cue"].values, df["R1"].values))
        + list(zip(df["cue"].values, df["R2"].values))
        + list(zip(df["cue"].values, df["R3"].values))
    )
    edges = [e for e in edges if "" not in e and e[0] != e[1]]
    return edges


def graph_from_edge_list(
    edges: List[Tuple[str, str]], directed: bool = True, weighted: bool = True
) -> nx.Graph:
    """Build NetworkX graph from edge list. Weights = occurrence counts."""
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    counter: defaultdict[Tuple[str, str], int] = defaultdict(int)
    for a, b in edges:
        if a is not None and b is not None and str(a) and str(b):
            counter[(a, b)] += 1
    if weighted:
        g.add_weighted_edges_from([(a, b, c) for (a, b), c in counter.items()])
    else:
        g.add_edges_from(set(edges))
    return g


def make_undirected(g: nx.Graph) -> nx.Graph:
    """Convert to undirected, using max weight for bidirectional edges."""
    ug = g.to_undirected()
    for u, v in ug.edges():
        w_uv = g.get_edge_data(u, v, default={}).get("weight", 0)
        w_vu = g.get_edge_data(v, u, default={}).get("weight", 0)
        ug.edges[u, v]["weight"] = max(w_uv, w_vu)
    return ug


def wn_filter(g: nx.Graph, wn_word_set: set | None = None, use_direct_lookup: bool = False) -> nx.Graph:
    """Keep only nodes that exist in WordNet.
    use_direct_lookup=True: call wn.synsets() per node (matches original FA script, slower).
    use_direct_lookup=False: use cached wn_word_set for O(1) lookup (faster, may differ slightly)."""
    if use_direct_lookup and wn is not None:
        keep = [n for n in g.nodes() if wn.synsets(str(n).replace(" ", "_"))]
    elif wn_word_set is not None:
        keep = [n for n in g.nodes() if str(n).replace(" ", "_") in wn_word_set]
    elif wn is not None:
        from lwow.fa_cleaning import get_wn_word_set
        wn_word_set = get_wn_word_set(use_cache=True)
        keep = [n for n in g.nodes() if str(n).replace(" ", "_") in wn_word_set]
    else:
        raise ImportError("nltk.corpus.wordnet required for wn_filter")
    return g.subgraph(keep).copy()


def idiosyn_filter(g: nx.Graph) -> nx.Graph:
    """Keep only edges with weight > 1 (remove idiosyncratic single-occurrence edges)."""
    keep_edges = [
        (u, v) for u, v in g.edges()
        if g.get_edge_data(u, v, default={}).get("weight", 0) > 1
    ]
    return g.edge_subgraph(keep_edges).copy()


def largest_connected_component(g: nx.Graph) -> nx.Graph:
    """Extract largest connected component."""
    if g.number_of_nodes() == 0:
        return g.copy()
    if nx.is_directed(g):
        components = list(nx.weakly_connected_components(g))
    else:
        components = list(nx.connected_components(g))
    if not components:
        return g.copy()
    largest = max(components, key=len)
    return g.subgraph(largest).copy()


def graph_to_csv(g: nx.Graph, path: str | Path) -> None:
    """Write graph edges to CSV with columns src,tgt,wt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for u, v in g.edges():
        w = g.get_edge_data(u, v, default={}).get("weight")
        if w is not None:
            rows.append({"src": u, "tgt": v, "wt": int(w)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def build_filtered_graph(
    df: pd.DataFrame,
    wn_word_set: set | None = None,
    use_direct_wn_lookup: bool = False,
) -> nx.Graph:
    """Run full FA network pipeline: edges -> directed weighted -> undirected -> WN -> idiosyn -> LCC.
    use_direct_wn_lookup=True: call wn.synsets() per node (matches original LWOW exactly, slower)."""
    edges = fa_edge_list(df)
    g = graph_from_edge_list(edges, directed=True, weighted=True)
    g = make_undirected(g)
    g = wn_filter(g, wn_word_set=wn_word_set, use_direct_lookup=use_direct_wn_lookup)
    g = idiosyn_filter(g)
    g = largest_connected_component(g)
    return g
