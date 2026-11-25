"""Build aggregate valence regressions and eight NRC emotion-set contrasts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))
from wordviz.bias_norms import EMOTIONS, STORE
from wordviz.norm_statistics import emotion_statistics, valence_statistics
from wordviz.models import MATRIX_KEYS, RESOURCES
from wordviz.social_domains import DOMAIN_PRIMES


VAD_MEMBER = "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
EMOTION_MEMBER = "NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"


def english_source(path, member):
    if is_zipfile(path):
        with ZipFile(path) as archive:
            return archive.read(member)
    return path.read_bytes()


def valence_summary(frame, ratings):
    aligned = ratings.reindex(frame.index).to_numpy(dtype=float)
    return valence_statistics(frame.to_numpy(dtype=float), frame.columns, aligned)


def emotion_summary(frame, memberships):
    pairs = {}
    indices = {emotion: np.flatnonzero(frame.index.isin(words)) for emotion, words in memberships.items()}
    for left, right in combinations(sorted(frame.columns), 2):
        pair = frame[[left, right]].to_numpy(dtype=float)
        pairs.setdefault(left, {})[right] = emotion_statistics(pair, indices)
    return pairs


def read_matrices(model, directory):
    if directory is None:
        from build_social_store import raw_domains
        for domain, frame, _ in raw_domains(model):
            yield domain, frame
    else:
        for domain in DOMAIN_PRIMES:
            path = directory / model / f"{domain}.csv.gz"
            yield domain, pd.read_csv(path, index_col=0, keep_default_na=False, float_precision="round_trip")


def build(vad_path, emotion_path, matrix_dir, destination, models=None):
    vad_bytes = english_source(vad_path, VAD_MEMBER)
    emotion_bytes = english_source(emotion_path, EMOTION_MEMBER)
    vad = pd.read_csv(io.BytesIO(vad_bytes), sep="\t", header=None, index_col=0,
                      keep_default_na=False, names=["word", "valence", "arousal", "dominance"]).valence
    if not vad.index.is_unique or not vad.between(0, 1).all():
        raise ValueError("Expected the NRC-VAD v1 English word list (0 to 1)")
    memberships = {emotion: set() for emotion in EMOTIONS}
    for line in csv.reader(io.StringIO(emotion_bytes.decode("utf-8")), delimiter="\t"):
        if len(line) == 3 and line[1] in memberships and line[2] == "1":
            memberships[line[1]].add(line[0])
    if not all(memberships.values()):
        raise ValueError("Expected the eight word-level NRC Emotion Lexicon categories")
    results = {}
    sources = {"NRC-VAD v1": hashlib.sha256(vad_bytes).hexdigest(),
               "NRC EmoLex v0.92": hashlib.sha256(emotion_bytes).hexdigest()}
    if models and destination.is_file():
        with gzip.open(destination, "rt") as previous:
            payload = json.load(previous)
        if payload["sources"] != sources:
            raise ValueError("Rebuild all models when the norm lexicons change")
        results = payload["results"]
    for model in models or [key for key in MATRIX_KEYS if (RESOURCES / "social" / f"{key}.sqlite").is_file()]:
        for domain, frame in read_matrices(model, matrix_dir):
            if not frame.index.is_unique or not frame.columns.is_unique:
                raise ValueError(f"Duplicate node or prime in {model}/{domain}")
            results.setdefault(domain, {})[model] = {
                "primes": list(frame.columns),
                "valence": valence_summary(frame, vad),
                "emotion": emotion_summary(frame, memberships),
            }
        print(f"Computed valence and emotion analyses for {model}", flush=True)
    payload = {
        "sources": sources,
        "results": results,
    }
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
        temporary = Path(out.name)
    try:
        temporary.write_bytes(gzip.compress(raw, mtime=0))
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad", type=Path, required=True, help="Official NRC-VAD v1 ZIP or English text file")
    parser.add_argument("--emolex", type=Path, required=True, help="Official NRC EmoLex v0.92 ZIP or English word-level text")
    parser.add_argument("--matrix-dir", type=Path, help="Use raw matrices exported by build_social_store.py")
    parser.add_argument("--output", type=Path, default=STORE)
    parser.add_argument("--models", nargs="+", choices=MATRIX_KEYS, help="Refresh only these models in the output store.")
    args = parser.parse_args()
    build(args.vad, args.emolex, args.matrix_dir, args.output, args.models)
