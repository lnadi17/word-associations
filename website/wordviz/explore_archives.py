"""Verified, commit-pinned Explore archives for models outside the function bundle."""

import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock
from urllib.request import urlopen

DATA_DIR = Path(__file__).resolve().parent / "resources" / "explore"
BUNDLE_DIR = DATA_DIR.parent / "bundled_explore"
MANIFEST = DATA_DIR / "remote.json"
CACHE = Path(tempfile.gettempdir()) / "word-associations-explore"
# Keep the original network dropdown defaults available without a network request.
BUNDLED = {"humans", "gpt5_mini", "gemma", "haiku", "llama3", "mistral"}


@lru_cache(maxsize=1)
def remote_archives():
    if not MANIFEST.is_file():
        return {}
    manifest = json.loads(MANIFEST.read_text())
    commit = manifest["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid Explore source commit")
    records = {}
    for model, entry in manifest["archives"].items():
        digest, size = entry["sha256"], entry["bytes"]
        if not (re.fullmatch(r"[a-z0-9_]+", model) and re.fullmatch(r"[0-9a-f]{64}", digest)
                and isinstance(size, int) and size > 0):
            raise ValueError(f"Invalid Explore source for {model}")
        records[model] = (commit, digest, size)
    return records


@lru_cache(maxsize=None)
def _lock(digest):
    return Lock()


@lru_cache(maxsize=None)
def _cached_archive(model, commit, digest, size):
    target = CACHE / f"{digest}.zip"
    with _lock(digest):
        if target.is_file() and target.stat().st_size == size:
            with target.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() == digest:
                    return target
        CACHE.mkdir(parents=True, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/lnadi17/word-associations/{commit}"
               f"/website/wordviz/resources/explore/{model}.zip")
        # Publish only a complete, verified ZIP. Concurrent processes can safely
        # replace the same content-addressed path with identical checked bytes.
        temporary = None
        try:
            with urlopen(url, timeout=15) as response, tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as out:
                temporary = Path(out.name)
                hasher = hashlib.sha256()
                received = 0
                while chunk := response.read(256 * 1024):
                    received += len(chunk)
                    if received > size:
                        raise ValueError(f"Explore archive size mismatch for {model}")
                    hasher.update(chunk)
                    out.write(chunk)
            if received != size or hasher.hexdigest() != digest:
                raise ValueError(f"Explore archive checksum mismatch for {model}")
            os.replace(temporary, target)
            return target
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


def local_archive(model):
    for directory in (DATA_DIR, BUNDLE_DIR):
        path = directory / f"{model}.zip"
        if path.is_file():
            return path
    return None


def archive_path(model):
    local = local_archive(model)
    if local:
        return local
    return _cached_archive(model, *remote_archives()[model])
