from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Dict, Any


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_cue_stats(path: str | Path) -> List[str]:
    cues: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cue = (row.get("cue") or "").strip()
            if cue:
                cues.append(cue)
    return cues


def write_csv(path: str | Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)
