"""Build compact graph snapshots for the association-path view."""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "website/wordviz/resources/paths"


def build_graph(path, valence, destination):
    neighbors = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            src, tgt = row["src"], row["tgt"]
            neighbors.setdefault(src, set()).add(tgt)
            neighbors.setdefault(tgt, set()).add(src)
    words = sorted(neighbors)
    if not words:
        raise ValueError(f"Empty graph in {path}")
    index = {word: i for i, word in enumerate(words)}
    graph = {
        "words": words,
        "valence": [valence.get(word) for word in words],
        "neighbors": [sorted(index[s] for s in neighbors[word]) for word in words],
        "source": path.name,
    }
    temporary = destination.with_suffix(".part")
    with temporary.open("wb") as f:
        with gzip.GzipFile(fileobj=f, filename="", mode="wb", mtime=0, compresslevel=6) as gz:
            gz.write(json.dumps(graph, ensure_ascii=False, separators=(",", ":")).encode())
    os.replace(temporary, destination)
    print(f"{destination.name}  {len(words):,} nodes  {destination.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    with (ROOT / "lexicon/NRC-VAD-Lexicon-v2.1.txt").open(encoding="utf-8") as lexicon:
        valence = {
            row["term"]: float(row["valence"])
            for row in csv.DictReader(lexicon, delimiter="\t")
        }
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "data/graphs/edge_lists").glob("FA_*_edgelist.csv")):
        model = path.name[len("FA_") : -len("_edgelist.csv")]
        build_graph(path, valence, args.output / f"{model}.json.gz")


if __name__ == "__main__":
    main()
