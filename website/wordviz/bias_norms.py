"""Read the precomputed NRC analyses without shipping the underlying lexicons."""

import gzip
import json
from functools import lru_cache
from pathlib import Path


STORE = Path(__file__).parent / "resources" / "social_norms.json.gz"
EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")


@lru_cache(maxsize=1)
def _store():
    with gzip.open(STORE, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def valence_result(domain, model):
    analysis = _store()["results"].get(domain, {}).get(model)
    if analysis is None:
        return {"rows": [], "joint": None, "available_primes": [], "reason": "No matrix for this network and domain."}
    return {**analysis["valence"], "available_primes": analysis["primes"]}


def emotion_result(domain, model, left, right):
    analysis = _store()["results"].get(domain, {}).get(model, {})
    primes = analysis.get("primes", [])
    if left not in primes or right not in primes or left == right:
        return {"rows": [], "available_primes": primes, "reason": "Choose two distinct available primes."}
    low, high = sorted((left, right))
    rows = analysis["emotion"][low][high]
    if left != low:
        rows = [{**row, "effect": -row["effect"] if row["effect"] is not None else None} for row in rows]
    return {"rows": rows, "available_primes": primes}
