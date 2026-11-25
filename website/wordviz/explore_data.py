"""Indexed lookup of directional associations from cleaned response trials."""
from __future__ import annotations

import gzip
import json
from http.client import HTTPException
from dataclasses import dataclass
from functools import lru_cache
from zipfile import BadZipFile, ZipFile
from zlib import crc32

from .explore_archives import DATA_DIR, archive_path
from .models import GRAPH_MODELS as MODELS

BUCKETS = 64
DIRECTIONS = ("forward", "backward", "merged")


def bucket_name(word: str) -> str:
    return f"{crc32(word.encode('utf-8')) % BUCKETS:02d}.json.gz"


class StoreUnavailable(Exception):
    """Prepared data is missing or unreadable."""


@lru_cache(maxsize=24)
def _read(model: str, filename: str):
    try:
        with ZipFile(archive_path(model)) as archive:
            contents = archive.read(filename)
        return json.loads(gzip.decompress(contents))
    except (OSError, ValueError, EOFError, KeyError, BadZipFile, HTTPException) as error:
        raise StoreUnavailable(f"Association data for {MODELS[model]} could not be loaded.") from error


def vocabulary(model: str) -> list[str]:
    return _read(model, "words.json.gz")


@dataclass
class Neighbors:
    model: str
    word: str
    counts: dict[str, int]
    exists: bool

    @property
    def label(self) -> str:
        return f"{self.word} · {MODELS[self.model]}"

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def neighbors(model: str, word: str, direction: str) -> Neighbors:
    if model not in MODELS or direction not in DIRECTIONS:
        raise ValueError("Unknown model or connection direction")
    word = word.strip().casefold()
    entry = _read(model, bucket_name(word)).get(word)
    if entry is None:
        return Neighbors(model, word, {}, False)
    forward, backward = entry
    if direction == "forward":
        counts = forward
    elif direction == "backward":
        counts = backward
    else:
        counts = forward.copy()
        for source, weight in backward.items():
            counts[source] = max(weight, counts.get(source, 0))
    return Neighbors(model, word, counts, True)
