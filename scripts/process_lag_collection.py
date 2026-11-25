"""Convert one completed LAG collection into cleaned trials and a filtered FA network.

Usage: python scripts/process_lag_collection.py MODEL RUN_DIRECTORY_OR_TRIAL_CSV
Each collected cue must contain each of its 100 distinct trial numbers exactly once.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lwow.fa_cleaning import clean_trial_frame, get_wn_word_set
from lwow.fa_networks import graph_to_csv, idiosyn_filter, largest_connected_component, make_undirected, wn_filter
from lwow.generation import parse_associations
from scripts.build_fa_graph_reports import _export_igraph_graphml

# Accept lexical items and short phrases. Keep Unicode letters, word-internal
# apostrophes, dots and hyphens. Exclude explanations, markup and incomplete lists.
ITEM = re.compile(r"[\w'’.‐‑–-]+(?: [\w'’.‐‑–-]+){0,3}\Z")
TRIALS = 100


def parse_list(text):
    text = text.strip()
    if not text:
        return (), "empty"
    # A terminated reasoning channel can be separated unambiguously from the answer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    line = text.partition("\n")[0].strip()
    if line.lower().startswith("output:"):
        line = line[7:].strip()
    items = parse_associations(line)
    if len(items) != 3:
        return (), "incomplete_list"
    if not all(ITEM.fullmatch(item) and any(c.isalnum() for c in item) for item in items):
        return (), "nonlexical_list"
    return tuple(items), "accepted"


def load_trials(source, expected_cues):
    batched = source.is_dir()
    paths = sorted((source / "outputs").glob("*.csv"), key=lambda p: int(p.stem)) if batched else [source]
    if batched and (not paths or [int(p.stem) for p in paths] != list(range(len(paths)))):
        raise ValueError(f"Missing or non-contiguous CSV shards in {source}")
    trial_bits = defaultdict(int)
    counts = Counter()

    def records():
        for path in paths:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                # Python 3.9's CSV reader rejects NUL even inside a quoted field.
                reader = csv.DictReader(line.replace("\0", "\ufffd") for line in handle)
                required = ({"cue", "variation_index", "finish_reason", "response"} if batched else
                            {"cue", "trial_id", "response_1", "response_2", "response_3"})
                if not required.issubset(reader.fieldnames or ()):
                    raise ValueError(f"Missing required columns in {path}")
                for row in reader:
                    cue = row["cue"].strip()
                    trial = int(row["variation_index"]) if batched else int(row["trial_id"]) - 1
                    if not cue or not 0 <= trial < TRIALS or trial_bits[cue] & (1 << trial):
                        raise ValueError(f"Missing cue, repeated or invalid trial in {path}: {cue!r}, {trial}")
                    trial_bits[cue] |= 1 << trial
                    if batched and row["finish_reason"] != "stop":
                        items, status = (), "finish_" + row["finish_reason"]
                    elif batched:
                        items, status = parse_list(row["response"])
                    else:
                        slots = [row["response_" + str(i)].strip() for i in (1, 2, 3)]
                        items, status = parse_list(",".join(s for s in slots if s and s.lower() != "nan"))
                    counts[status] += 1
                    yield (sys.intern(cue), *(sys.intern(item) for item in (items or ("", "", ""))))

    df = pd.DataFrame.from_records(records(), columns=["cue", "R1", "R2", "R3"])
    if any(bits != (1 << TRIALS) - 1 for bits in trial_bits.values()):
        raise ValueError(f"At least one cue in {source} has incomplete trial coverage")
    if expected_cues is not None and len(trial_bits) != expected_cues:
        raise ValueError(f"Expected {expected_cues} distinct cues in {source}, read {len(trial_bits)}")
    return df, {"raw_cues": len(trial_bits), "raw_rows": len(df), "input_files": len(paths),
                "parse_outcomes": dict(sorted(counts.items()))}


def filtered_graph(df):
    # Count within slots before union to avoid materializing millions of identical tuples.
    counts = Counter()
    for slot in ["R1", "R2", "R3"]:
        present = df[slot].ne("") & df["cue"].ne("") & df[slot].ne(df["cue"])
        counts.update(df.loc[present].groupby(["cue", slot], sort=False).size().to_dict())
    graph = nx.DiGraph()
    graph.add_weighted_edges_from((cue, word, count) for (cue, word), count in counts.items())
    undirected = make_undirected(graph)
    wordnet = wn_filter(undirected, wn_word_set=get_wn_word_set())
    return largest_connected_component(idiosyn_filter(wordnet))


def process(model, source, output, expected_cues):
    started = time.monotonic()
    df, report = load_trials(source, expected_cues)
    print(f"{model}: read {len(df):,} trials in {time.monotonic()-started:.1f}s", flush=True)
    df = clean_trial_frame(df, spelling_dict_path=ROOT / "lexicon/mapping_tables/EnglishCustomDict.txt", seed=42)
    print(f"{model}: cleaned {len(df):,} trials after {time.monotonic()-started:.1f}s", flush=True)
    graph = filtered_graph(df)
    report.update(clean_cues=int(df["cue"].nunique()), clean_rows=len(df),
                  nonblank_responses=int(df[["R1", "R2", "R3"]].ne("").sum().sum()),
                  filtered_nodes=graph.number_of_nodes(), filtered_edges=graph.number_of_edges())
    if not report["filtered_edges"]:
        raise ValueError(f"No graph edges survived filtering in {source}")
    paths = (Path(f"processed/{model}_processed.csv"), Path(f"graphs/edge_lists/FA_{model}_edgelist.csv"),
             Path(f"graphs/igraphs/FA_{model}_ig.graphml"), Path(f"summary_tables/collection_audits/{model}.json"))
    output.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        prepared = [Path(tmp) / path.name for path in paths]
        df.to_csv(prepared[0], index=False)
        graph_to_csv(graph, prepared[1])
        _export_igraph_graphml(graph, prepared[2])
        prepared[3].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for source_file, path in zip(prepared, paths):
            dest = output / path
            dest.parent.mkdir(exist_ok=True, parents=True)
            source_file.replace(dest)
    print(f"{model}: wrote {report['filtered_nodes']:,} nodes, {report['filtered_edges']:,} edges "
          f"in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--expected-cues", type=int, help="Fail if a collection is missing complete cue blocks")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.model):
        parser.error("model must be a lowercase dataset key")
    process(args.model, args.source, args.output, args.expected_cues)
