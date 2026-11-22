"""Read precomputed social-prime activations; the web server never simulates a graph."""
import json
import sqlite3
import struct
from contextlib import closing
from functools import lru_cache
from pathlib import Path

from .experiment_data import genders, suggest_targets
from .models import MATRIX_MODELS
from .social_domains import DOMAIN_PRIMES

STORE_DIR = Path(__file__).resolve().parent / "resources" / "social"


def _connect(model):
    if model not in MATRIX_MODELS:
        raise ValueError("Unknown activation model")
    return sqlite3.connect(f"{(STORE_DIR / (model + '.sqlite')).as_uri()}?mode=ro&immutable=1", uri=True)


@lru_cache(maxsize=len(MATRIX_MODELS))
def _metadata(model):
    with closing(_connect(model)) as db:
        return {k: json.loads(v) for k, v in db.execute("SELECT name, value FROM metadata")}


def domain_info(domain, model):
    return _metadata(model)["domains"][domain]


def domain_values(domain, words, models):
    """L2-normalized over all nodes then the available primes of the complete domain."""
    models = list(dict.fromkeys(models))
    if domain == "gender":
        values = genders(words)
        return {model: values[model] for model in models}
    result = {model: {} for model in models}
    if domain not in DOMAIN_PRIMES or not words:
        return result
    words = list(dict.fromkeys(words))
    marks = ",".join("?" for _ in words)
    for model in models:
        meta = _metadata(model)
        columns = meta["columns"]
        vector = struct.Struct(f"<{len(columns)}d")
        selected = [(p, columns.index(p)) for p in meta["domains"][domain]["primes"]]
        with closing(_connect(model)) as db:
            for node, blob in db.execute(f"SELECT node, levels FROM activation WHERE node IN ({marks})", words):
                row = vector.unpack(blob)
                result[model][node] = {p: row[i] for p, i in selected}
    return result


def domain_suggest(domain, prefix, models, limit=35):
    """Autocomplete over the union of the selected models' target vocabularies."""
    if domain == "gender":
        return suggest_targets(prefix, models, limit)
    prefix = (prefix or "").strip().casefold()
    if domain not in DOMAIN_PRIMES or not prefix or not models:
        return []
    found = set()
    for model in set(models):
        with closing(_connect(model)) as db:
            found.update(row[0] for row in db.execute(
                "SELECT node FROM activation WHERE node >= ? AND node < ? ORDER BY node LIMIT ?",
                (prefix, prefix + chr(0x10ffff), limit)))
    return sorted(found)[:limit]
