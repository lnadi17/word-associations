"""Package the normalized activation matrices for the research website."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import struct
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))
from wordviz.experiment_data import MODELS, STORE
from run_fa_statistical_tests import normalize_df

ANALYSES = ROOT / "data" / "LDT_analyses"


def read_csv(name):
    with (ANALYSES / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build(destination=STORE):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=destination.parent, suffix=".sqlite")
    os.close(fd)
    try:
        with closing(sqlite3.connect(path)) as db:
            db.executescript("""
                CREATE TABLE gender(node TEXT, model TEXT, vector BLOB NOT NULL,
                    PRIMARY KEY (node, model)) WITHOUT ROWID;
                CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            triplets = [
                [r["Target"], r["Related Prime"], r["Unrelated Prime"],
                 float(r["Target-Related RT"]), float(r["Target-Unrelated RT"])]
                for r in read_csv("LDT_50_triplets.csv")
            ]
            targets = read_csv("gender_targets.csv")
            primes = [r["genderWord"] for r in read_csv("gender_primes.csv")]
            provenance = {}
            activation = {}
            for model in MODELS:
                provenance[model] = {}
                for task in ("LDT", "gender"):
                    source = ANALYSES / "FA_matrices" / f"{model}_{task}.csv"
                    frame = pd.read_csv(source, keep_default_na=False)
                    norm = normalize_df(frame, normalize_rows=True)
                    if not norm.index.is_unique:
                        raise ValueError(f"Repeated node labels in {source}")
                    provenance[model][task] = {
                        "file": source.name, "nodes": len(norm),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                    if task == "LDT":
                        activation[model] = [
                            [norm.at[target, related], norm.at[target, unrelated]]
                            for target, related, unrelated, *_ in triplets
                        ]
                    else:
                        if set(norm.columns) != set(primes):
                            raise ValueError(f"Unexpected prime columns in {source}")
                        vector = struct.Struct(f"<{len(primes)}d")
                        db.executemany(
                            "INSERT INTO gender VALUES (?, ?, ?)",
                            ((node, model, vector.pack(*row))
                             for node, *row in norm[primes].itertuples(name=None)),
                        )
            info = {
                "ldt": {"triplets": triplets, "activation": activation},
                "primes": primes,
                "targets": [[r[group] for r in targets if r[group]] for group in ("Female", "Male")],
                "provenance": provenance,
            }
            db.executemany("INSERT INTO metadata VALUES (?, ?)",
                           ((key, json.dumps(value, allow_nan=False)) for key, value in info.items()))
            db.commit()
            db.execute("VACUUM")
        os.chmod(path, 0o644)
        os.replace(path, destination)
    finally:
        Path(path).unlink(missing_ok=True)
    print(f"{destination}: {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    build()
