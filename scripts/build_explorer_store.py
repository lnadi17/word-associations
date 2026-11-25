"""Build compact read-only word indexes from the cleaned trial tables."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))
from wordviz.explore_data import BUCKETS, DATA_DIR, bucket_name
from wordviz.models import MODEL_LABELS


def sources() -> dict[str, Path]:
    local = {path.stem.removesuffix("_processed"): path
             for path in (ROOT / "data/processed").glob("*_processed.csv")}
    external = {path.stem.removeprefix("FA_").lower(): path
                for path in (ROOT / "lexicon/LWOW_datasets/processed_datasets").glob("FA_*.csv")}
    return {**external, **local}


def write_json(archive: ZipFile, filename: str, value) -> None:
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    archive.writestr(ZipInfo(filename), gzip.compress(data, compresslevel=6, mtime=0))


def build(source: Path, destination: Path) -> None:
    forward = defaultdict(Counter)
    backward = defaultdict(Counter)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"cue", "R1", "R2", "R3"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{source} requires a cleaned cue,R1,R2,R3 table")
        for row in reader:
            cue = sys.intern(row["cue"].strip().casefold())
            if not cue:
                continue
            outgoing = forward[cue]  # Preserve observed cues with zero usable responses.
            for slot in ("R1", "R2", "R3"):
                associate = sys.intern(row[slot].strip().casefold())
                if associate and associate != cue:
                    outgoing[associate] += 1
                    backward[associate][cue] += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    words = sorted(forward.keys() | backward.keys())
    buckets = {f"{index:02d}.json.gz": {} for index in range(BUCKETS)}
    for word in words:
        buckets[bucket_name(word)][word] = (forward.get(word, {}), backward.get(word, {}))
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        try:
            with ZipFile(temp, "w", compression=ZIP_STORED) as archive:
                for name, entries in buckets.items():
                    write_json(archive, name, entries)
                write_json(archive, "words.json.gz", words)
            os.chmod(temp.name, 0o644)
            os.replace(temp.name, destination)
        finally:
            Path(temp.name).unlink(missing_ok=True)
    print(f"{destination.stem}: {len(words):,} words, {sum(map(len, forward.values())):,} directed pairs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIR,
                        help="Directory for one independently readable archive per model.")
    parser.add_argument("--models", nargs="+", choices=MODEL_LABELS)
    args = parser.parse_args()
    available = sources()
    for model in args.models or [key for key in MODEL_LABELS if key in available]:
        build(available[model], args.output / f"{model}.zip")


if __name__ == "__main__":
    main()
