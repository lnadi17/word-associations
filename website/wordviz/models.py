"""Catalog labels and selectable datasets backed by the packaged analysis files."""

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

from .explore_archives import local_archive, remote_archives
from .model_provenance import MODEL_PROVENANCE

RESOURCES = Path(__file__).resolve().parent / "resources"


def model_sort_key(label):
    """Order the displayed model name, then its build, with numeric sizes and versions."""
    def tokens(text):
        parts = re.findall(r"\d+(?:\.\d+)*|[^\W\d_]+", text)
        return tuple((0, tuple(map(int, part.split(".")))) if part[0].isdecimal() else (1, part)
                     for part in parts)

    name, _, build = label.casefold().partition(" · ")
    return tokens(name), tokens(build), label.casefold()


def ordered_model_keys(keys):
    return sorted(keys, key=lambda key: model_sort_key(MODEL_PROVENANCE[key.lower()]["label"]))


MODEL_LABELS = {key: MODEL_PROVENANCE[key]["label"] for key in ordered_model_keys(MODEL_PROVENANCE)}

# Keep the case-sensitive identifiers used by the six original matrix files.
ARCHIVED_MATRICES = ("Humans", "Gpt5_mini", "Gemma", "Haiku", "Llama3", "Mistral")
_archive_keys = {key.lower(): key for key in ARCHIVED_MATRICES}
MATRIX_KEYS = tuple(_archive_keys.get(key, key) for key in MODEL_LABELS)

GRAPH_MODELS = {
    key: label for key, label in MODEL_LABELS.items()
    if (local_archive(key) or key in remote_archives())
    and (RESOURCES / "paths" / f"{key}.json.gz").is_file()
}


def _experiment_keys():
    path = RESOURCES / "experiments.sqlite"
    if not path.is_file():
        return set()
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)) as db:
        row = db.execute("SELECT value FROM metadata WHERE name = 'provenance'").fetchone()
    return set(json.loads(row[0]))


_published_experiments = _experiment_keys()
MATRIX_MODELS = {
    key: GRAPH_MODELS[key.lower()] for key in MATRIX_KEYS
    if key in _published_experiments and key.lower() in GRAPH_MODELS
    and (RESOURCES / "social" / f"{key}.sqlite").is_file()
    and (RESOURCES / "social_weights" / f"{key.lower()}.gz").is_file()
}
