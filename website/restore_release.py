"""Materialize large research files for a source-file deployment from a pinned Git commit."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent


def restore(commit, entry):
    name, digest = entry
    path = (ROOT / name).resolve()
    path.relative_to(ROOT / "wordviz" / "resources")
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError(f"Invalid SHA-256 for {name}")
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return
    url = f"https://raw.githubusercontent.com/lnadi17/word-associations/{commit}/website/{quote(name)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with urlopen(url, timeout=120) as response, temp.open("wb") as output:
            shutil.copyfileobj(response, output)
        with temp.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Release checksum mismatch for {name}")
        temp.chmod(0o644)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    manifest = ROOT / "release-files.json"
    if manifest.is_file():
        release = json.loads(manifest.read_text())
        commit = release["commit"]
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("Expected a full Git commit hash")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda entry: restore(commit, entry), release["files"].items()))
        print(f"Verified {len(release['files'])} research files from {commit}", flush=True)
