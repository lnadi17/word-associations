"""Read the bundled filtered graphs and find their shortest-path unions."""

from collections import deque
from functools import lru_cache
import gzip
import json
from pathlib import Path

from .models import GRAPH_MODELS as MODEL_NAMES


GRAPH_DIR = Path(__file__).parent / "resources/paths"


@lru_cache(maxsize=2)
def load_graph(model):
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown graph {model!r}")
    with gzip.open(GRAPH_DIR / f"{model}.json.gz", "rt", encoding="utf-8") as stream:
        graph = json.load(stream)
    graph["index"] = {word: i for i, word in enumerate(graph["words"])}
    return graph


def distances_to(graph, start, end):
    """Discover the target depth and every potential predecessor distance."""
    dist = [-1] * len(graph["words"])
    dist[start] = 0
    if start == end:
        return dist
    queue = deque([start])
    while queue:
        node = queue.popleft()
        next_depth = dist[node] + 1
        for neighbor in graph["neighbors"][node]:
            if dist[neighbor] != -1:
                continue
            dist[neighbor] = next_depth
            if neighbor == end:
                return dist
            queue.append(neighbor)
    return dist


def _display_edges(layers, successors, start, end, node_limit, edge_limit):
    """Keep complete paths on screen even when the full union is too dense."""
    parent = {}
    for layer in layers:
        for node in layer:
            for child in successors[node]:
                parent.setdefault(child, node)
    nodes = {start, end}
    edges = set()
    for layer in layers:
        for node in layer:
            for child in successors[node]:
                if (node, child) in edges:
                    continue
                path_edges = {(node, child)}
                current = node
                while current != start:
                    path_edges.add((parent[current], current))
                    current = parent[current]
                current = child
                while current != end:
                    following = successors[current][0]
                    path_edges.add((current, following))
                    current = following
                path_nodes = {n for edge in path_edges for n in edge}
                if edges and (len(nodes | path_nodes) > node_limit or len(edges | path_edges) > edge_limit):
                    continue
                nodes.update(path_nodes)
                edges.update(path_edges)
    return nodes, sorted(edges)


@lru_cache(maxsize=128)
def stream(model, prime, target):
    graph = load_graph(model)
    index, words = graph["index"], graph["words"]
    missing = list(dict.fromkeys(w for w in (prime, target) if w not in index))
    if missing:
        return {"missing": missing}
    start, end = index[prime], index[target]
    forward = distances_to(graph, start, end)
    hops = forward[end]
    if hops == -1:
        return {"unreachable": True}
    # Trace only distance-decreasing edges back from the target. This is the
    # union of every shortest path and never enumerates all possible routes.
    layers = [[] for _ in range(hops + 1)]
    layers[hops] = [end]
    successors = {end: []}
    routes = {end: 1}
    for level in range(hops, 0, -1):
        previous = set()
        for child in layers[level]:
            for node in graph["neighbors"][child]:
                if forward[node] == level - 1:
                    previous.add(node)
                    successors.setdefault(node, []).append(child)
                    routes[node] = routes.get(node, 0) + routes[child]
        layers[level - 1] = sorted(previous)
    total_nodes = len(successors)
    total_edges = sum(map(len, successors.values()))
    if total_nodes <= 55 and total_edges <= 110:
        visible = set(successors)
        edges = [(a, b) for a, neighbors in successors.items() for b in neighbors]
    else:
        visible, edges = _display_edges(layers, successors, start, end, 55, 110)
    return {
        "hops": hops,
        "routes": routes[start],
        "node_count": total_nodes,
        "edge_count": total_edges,
        "visible_edges": [(words[a], words[b]) for a, b in edges],
        "layers": [[words[n] for n in layer if n in visible] for layer in layers],
        "valence": {words[n]: graph["valence"][n] for n in visible},
        "source": graph["source"],
    }
