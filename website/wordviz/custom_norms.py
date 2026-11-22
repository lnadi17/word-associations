"""Compute aggregate NRC analyses for primes selected from a graph."""

from functools import lru_cache

import numpy as np

from .custom_activation import raw_columns
from .norm_lexicons import load_lexicons
from .norm_statistics import emotion_statistics, valence_statistics


@lru_cache(maxsize=6)
def _aligned_norms(words):
    ratings, memberships = load_lexicons()
    aligned = np.array([ratings.get(word, np.nan) for word in words], dtype=float)
    indices = {emotion: np.fromiter((i for i, word in enumerate(words) if word in targets), dtype=int)
               for emotion, targets in memberships.items()}
    return aligned, indices


@lru_cache(maxsize=128)
def custom_valence(model, primes):
    words, available, raw = raw_columns(model, primes)
    missing = [word for word in primes if word not in available]
    if len(available) < 2:
        return {"rows": [], "joint": None, "available_primes": available,
                "reason": "The model needs two available primes to estimate a valence slope. "
                          "Missing primes: " + ", ".join(missing) + "."}
    ratings, _ = _aligned_norms(words)
    result = valence_statistics(raw, available, ratings)
    result["available_primes"] = available
    if missing:
        result["reason"] = ("Normalization uses the " + str(len(available)) + " available primes. "
                            "The graph lacks " + ", ".join(missing) + ".")
    return result


@lru_cache(maxsize=128)
def custom_emotion(model, left, right):
    words, available, raw = raw_columns(model, (left, right))
    if len(available) != 2:
        missing = [word for word in (left, right) if word not in available]
        return {"rows": [], "available_primes": available,
                "reason": "The graph lacks the selected prime" + ("s " if len(missing) > 1 else " ")
                          + ", ".join(missing) + "."}
    _, indices = _aligned_norms(words)
    return {"rows": emotion_statistics(raw, indices), "available_primes": available}
