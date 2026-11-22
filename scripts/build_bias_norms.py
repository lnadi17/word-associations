"""Build aggregate valence regressions and eight NRC emotion-set contrasts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))
from wordviz.bias_norms import EMOTIONS, STORE
from wordviz.experiment_data import signed_rank
from wordviz.models import MATRIX_MODELS
from wordviz.social_domains import DOMAIN_PRIMES


VAD_MEMBER = "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
EMOTION_MEMBER = "NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"


def english_source(path, member):
    if is_zipfile(path):
        with ZipFile(path) as archive:
            return archive.read(member)
    return path.read_bytes()


def normalize(matrix, order):
    """Normalize full columns, then full rows, before selecting rated targets."""
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Activation matrix contains negative or non-finite values")
    values = np.array(matrix, dtype=float, copy=True)
    for axis in (0, 1):
        norms = np.linalg.norm(values, ord=order, axis=axis, keepdims=True)
        values /= np.where(norms == 0, 1, norms)
    return values


def equal_slopes(x, y):
    """Gaussian identity-link GLM, intercept included. H0: all prime slopes equal."""
    n, k = x.shape
    if n <= k + 1 or k < 2:
        return None
    full = np.column_stack((np.ones(n), x))
    coefficients, _, rank, _ = np.linalg.lstsq(full, y, rcond=None)
    if rank != k + 1:
        return None
    reduced = np.column_stack((np.ones(n), x.sum(axis=1)))
    common, _, reduced_rank, _ = np.linalg.lstsq(reduced, y, rcond=None)
    if reduced_rank != 2:
        return None
    full_sse = np.square(y - full @ coefficients).sum()
    reduced_sse = np.square(y - reduced @ common).sum()
    if full_sse <= 0 or reduced_sse <= 0:
        return None
    improvement = max(0.0, reduced_sse - full_sse)
    # For linear restrictions in a Gaussian model, Wald chi-square = q * partial F.
    statistic = improvement / (full_sse / (n - rank))
    return {"n": n, "df": k - 1, "statistic": float(statistic),
            "p": float(stats.chi2.sf(statistic, k - 1)), "partial_r2": float(improvement / reduced_sse)}


def valence_summary(frame, ratings):
    matrix = normalize(frame.to_numpy(dtype=float), 2)
    aligned = ratings.reindex(frame.index).to_numpy(dtype=float)
    rated = np.isfinite(aligned)
    x, y = matrix[rated], aligned[rated]
    n = len(y)
    rows = []
    for j, prime in enumerate(frame.columns):
        row = dict(prime=prime, n=n, slope=None, p=None, ci_low=None, ci_high=None)
        if n >= 3 and np.ptp(x[:, j]) > 0 and np.ptp(y) > 0:
            fit = stats.linregress(x[:, j], y)
            margin = float(stats.t.ppf(.975, n - 2) * fit.stderr)
            row.update(slope=float(fit.slope), p=float(fit.pvalue),
                       ci_low=float(fit.slope - margin), ci_high=float(fit.slope + margin))
        rows.append(row)
    return {"rows": rows, "joint": equal_slopes(x, y)}


def emotion_summary(frame, memberships):
    pairs = {}
    indices = {emotion: np.flatnonzero(frame.index.isin(words)) for emotion, words in memberships.items()}
    for left, right in combinations(sorted(frame.columns), 2):
        x = normalize(frame[[left, right]].to_numpy(dtype=float), 1)
        difference = x[:, 0] - x[:, 1]
        rows = []
        for emotion in EMOTIONS:
            d = difference[indices[emotion]]
            effect, _ = signed_rank(d.tolist())
            # Recover the signed z from the shared tie-corrected implementation.
            # Computing the two-sided p directly avoids cancellation in the upper tail.
            p = math.erfc(abs(effect) * math.sqrt(len(d) / 2)) if effect is not None else None
            rows.append({"emotion": emotion, "n": len(d), "nonzero": int(np.count_nonzero(d)),
                         "effect": effect, "p": p})
        pairs.setdefault(left, {})[right] = rows
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


def build(vad_path, emotion_path, matrix_dir, destination):
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
    for model in MATRIX_MODELS:
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
        "sources": {"NRC-VAD v1": hashlib.sha256(vad_bytes).hexdigest(),
                    "NRC EmoLex v0.92": hashlib.sha256(emotion_bytes).hexdigest()},
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
    args = parser.parse_args()
    build(args.vad, args.emolex, args.matrix_dir, args.output)
