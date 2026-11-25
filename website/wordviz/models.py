"""Catalog labels and selectable datasets backed by the packaged analysis files."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from .model_provenance import MODEL_PROVENANCE

RESOURCES = Path(__file__).resolve().parent / "resources"
MODEL_LABELS = {key: record["label"] for key, record in MODEL_PROVENANCE.items()}

# The published matrices use these case-sensitive filenames and display order.
ARCHIVED_MATRICES = ("Humans", "Gpt5_mini", "Gemma", "Haiku", "Llama3", "Mistral")
_archive_keys = {key.lower() for key in ARCHIVED_MATRICES}
MATRIX_KEYS = (*ARCHIVED_MATRICES, *(key for key in MODEL_LABELS if key not in _archive_keys))

GRAPH_MODELS = {
    key: label for key, label in MODEL_LABELS.items()
    if (RESOURCES / "explore" / f"{key}.zip").is_file()
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
