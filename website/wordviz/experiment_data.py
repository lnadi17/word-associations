"""Queries and statistics over immutable, fully normalized experiment matrices."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from collections import Counter
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from .models import MATRIX_MODELS as MODELS

STORE = Path(__file__).resolve().parent / "resources" / "experiments.sqlite"
DEFAULT_PAIRS = (
    ("woman", "man"), ("female", "male"), ("mother", "father"),
    ("girl", "boy"), ("feminine", "masculine"),
)
EXACT_SIGN_RANK_LIMIT = 20


def connect():
    return sqlite3.connect(f"{STORE.as_uri()}?mode=ro&immutable=1", uri=True)


@lru_cache(maxsize=4)
def metadata(key):
    with closing(connect()) as conn:
        row = conn.execute("SELECT value FROM metadata WHERE name = ?", (key,)).fetchone()
    return json.loads(row[0])


def genders(words):
    """Look up only requested target rows, across the available networks."""
    result = {model: {} for model in MODELS}
    if not words:
        return result
    provenance = metadata("provenance")
    default_primes = metadata("primes")
    placeholders = ",".join("?" for _ in words)
    with closing(connect()) as conn:
        for node, model, blob in conn.execute(
            f"SELECT node, model, vector FROM gender WHERE node IN ({placeholders})", words
        ):
            if model not in result:
                continue
            primes = provenance[model]["gender"].get("primes", default_primes)
            result[model][node] = dict(zip(primes, struct.unpack(f"<{len(primes)}d", blob)))
    return result


def suggest_targets(prefix, models, limit=35):
    prefix = (prefix or "").strip().lower()
    if not prefix or not models:
        return []
    placeholders = ",".join("?" for _ in models)
    with closing(connect()) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT node FROM gender WHERE node >= ? AND node < ? "
            f"AND model IN ({placeholders}) ORDER BY node LIMIT ?",
            [prefix, prefix + chr(0x10ffff), *models, limit],
        )
        return [row[0] for row in rows]


def ranks(values: Sequence[float]):
    """Average tied ranks using the convention of scipy.stats.rankdata."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranked = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[start]] == values[order[end]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranked[index] = rank
        start = end
    return ranked


def signed_rank(differences):
    """One-sided, asymptotic Wilcoxon; drop zeros and correct for ties."""
    nz = [d for d in differences if d != 0]
    n = len(nz)
    if not n:
        return None, None
    absolute = [abs(d) for d in nz]
    ties = sum(size ** 3 - size for size in Counter(absolute).values())
    variance = (n * (n + 1) * (2 * n + 1) - ties / 2) / 24
    positive = sum(rank for d, rank in zip(nz, ranks(absolute)) if d > 0)
    z = (positive - n * (n + 1) / 4) / math.sqrt(variance)
    return z / math.sqrt(len(differences)), math.erfc(z / math.sqrt(2)) / 2


def contrast_signed_rank(differences, *, directional=False):
    """Signed effect and exact or asymptotic p for a positive or two-sided alternative."""
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return None, None, 0, "No nonzero differences"
    effect, upper_p = signed_rank(differences)
    if n > EXACT_SIGN_RANK_LIMIT:
        p = upper_p if directional else math.erfc(abs(effect) * math.sqrt(len(differences) / 2))
        return effect, p, n, "Normal approximation"

    # Double average ranks so tied half-ranks have integer indices. Each
    # sign assignment contributes once to the distribution of positive sums.
    weights = [int(2 * rank) for rank in ranks([abs(d) for d in nonzero])]
    observed = sum(w for w, d in zip(weights, nonzero) if d > 0)
    total = sum(weights)
    counts = [0] * (total + 1)
    counts[0] = 1
    reached = 0
    for weight in weights:
        for subtotal in range(reached, -1, -1):
            counts[subtotal + weight] += counts[subtotal]
        reached += weight
    if directional:
        extreme = sum(counts[observed:])
    else:
        tail = abs(2 * observed - total)
        extreme = sum(count for subtotal, count in enumerate(counts) if abs(2 * subtotal - total) >= tail)
    return effect, extreme / (2 ** n), n, "Exact sign permutation"


def two_sided_signed_rank(differences):
    return contrast_signed_rank(differences)


def ldt_stats(model, experiment):
    pairs = [(trip, values) for trip, values in zip(experiment["triplets"], experiment["activation"][model])
             if values is not None]
    observations = [values for _, values in pairs]
    effect, wp = signed_rank([related - unrelated for related, unrelated in observations])
    activation = [v for pair in observations for v in pair]
    rt = [v for trip, _ in pairs for v in trip[3:]]
    n = len(activation)
    ar, rr = ranks(activation), ranks(rt)
    center = (n + 1) / 2
    ar = [v - center for v in ar]
    rr = [v - center for v in rr]
    norm = math.sqrt(sum(v*v for v in ar) * sum(v*v for v in rr))
    if not norm or n < 4:
        return effect, wp, None, None
    rho = max(-1.0, min(1.0, sum(a*b for a, b in zip(ar, rr)) / norm))
    # scipy.spearmanr uses a two-sided Student-t approximation. Each triplet
    # contributes two observations; for even n its beta tail is a binomial sum.
    a = (n - 2) // 2
    q = (1 - abs(rho)) / 2
    m = 2*a - 1
    p = 2 * sum(math.comb(m, k) * q**k * (1-q)**(m-k) for k in range(a, m+1))
    return effect, wp, rho, min(1.0, p)
