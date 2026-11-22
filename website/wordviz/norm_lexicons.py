"""Verified, server-only NRC inputs for analyses of user-defined primes."""

import csv
import hashlib
import io
import os
import tempfile
import threading
from functools import lru_cache
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from .bias_norms import EMOTIONS


SOURCES = {
    "vad": ("https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip",
            "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt",
            "fd49023f760155c8377424d96ca18d57c6685891d78ba381e47af6f4a1b148a7"),
    "emotion": ("https://saifmohammad.com/Lexicons/NRC-Emotion-Lexicon-v0.92.zip",
                "NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt",
                "bd90eb18a3a4f4cfbb41665909553c47dad43f8edec7da602f3b77317d2f7604"),
}
SERVER_PACK = Path(__file__).parent / "resources" / "private" / "nrc-word-norms.zip"
_CACHE_LOCK = threading.Lock()


class NormSourceError(Exception):
    """The verified source pack could not be loaded or assembled."""


def _verified_member(archive, key):
    _, member, digest = SOURCES[key]
    data = archive.read(member)
    if hashlib.sha256(data).hexdigest() != digest:
        raise NormSourceError(f"NRC {key} source differs from the published analysis input")
    return data


def prepare_pack(destination, local_archives=None):
    """Bundle only the pinned English source members on the server."""
    inputs = {}
    for key, (url, _, _) in SOURCES.items():
        try:
            if local_archives:
                archive_input = local_archives[key]
            else:
                request = Request(url, headers={"User-Agent": "Word-Associations/1.0 (+https://word-associations.vercel.app/)"})
                with urlopen(request, timeout=10) as response:
                    archive_input = io.BytesIO(response.read())
            with ZipFile(archive_input) as archive:
                inputs[key] = _verified_member(archive, key)
        except (OSError, URLError, BadZipFile, KeyError, EOFError, HTTPException) as exc:
            raise NormSourceError(f"Unable to load the official NRC {key} archive") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
        pending = Path(out.name)
    try:
        with ZipFile(pending, "w", ZIP_DEFLATED) as archive:
            archive.comment = b"Copyright National Research Council Canada. Non-commercial research use."
            for key, data in inputs.items():
                archive.writestr(SOURCES[key][1], data)
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def _read_pack(path):
    with ZipFile(path) as archive:
        vad_data = _verified_member(archive, "vad")
        emotion_data = _verified_member(archive, "emotion")
    vad = {word: float(valence) for word, valence, _, _ in
           csv.reader(io.StringIO(vad_data.decode("utf-8")), delimiter="\t")}
    memberships = {emotion: set() for emotion in EMOTIONS}
    for row in csv.reader(io.StringIO(emotion_data.decode("utf-8")), delimiter="\t"):
        if len(row) == 3 and row[1] in memberships and row[2] == "1":
            memberships[row[1]].add(row[0])
    return vad, memberships


@lru_cache(maxsize=1)
def load_lexicons():
    """Prefer the build's pack. Cache verified downloads across warm invocations."""
    if SERVER_PACK.is_file():
        try:
            return _read_pack(SERVER_PACK)
        except (OSError, BadZipFile, KeyError, NormSourceError):
            pass
    version = "-".join(source[2][:12] for source in SOURCES.values())
    cached = Path(tempfile.gettempdir()) / f"wordviz-nrc-{version}.zip"
    with _CACHE_LOCK:
        if cached.is_file():
            try:
                return _read_pack(cached)
            except (OSError, BadZipFile, KeyError, NormSourceError):
                pass
        try:
            prepare_pack(cached)
            return _read_pack(cached)
        except (OSError, BadZipFile, KeyError) as exc:
            raise NormSourceError("The local NRC norm cache is unavailable") from exc
