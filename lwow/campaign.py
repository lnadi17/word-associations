from __future__ import annotations

import json
import os
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List, Set

from lwow.io import ensure_parent_dir


def _atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_parent_dir(target)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def _read_json(path: str | Path, default: Dict[str, Any]) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return default
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return default
    return data


def _read_json_list(path: str | Path) -> List[str]:
    target = Path(path)
    if not target.exists():
        return []
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    return [str(x) for x in data]


def campaign_paths(campaign_root_dir: str, campaign_name: str) -> Dict[str, str]:
    root = Path(campaign_root_dir) / campaign_name
    return {
        "root_dir": str(root),
        "meta_path": str(root / "meta.json"),
        "master_cues_path": str(root / "master_cues.json"),
        "ledger_path": str(root / "ledger.jsonl"),
        "index_path": str(root / "index.json"),
    }


def init_or_load_campaign(
    campaign_root_dir: str,
    campaign_name: str,
    all_cues: List[str],
    master_seed: int,
) -> Dict[str, str]:
    paths = campaign_paths(campaign_root_dir, campaign_name)
    root_dir = Path(paths["root_dir"])
    root_dir.mkdir(parents=True, exist_ok=True)

    if not Path(paths["master_cues_path"]).exists():
        rng = Random(master_seed)
        master_cues = list(all_cues)
        rng.shuffle(master_cues)
        with open(paths["master_cues_path"], "w", encoding="utf-8") as handle:
            json.dump(master_cues, handle, ensure_ascii=False, indent=2)
        _atomic_write_json(
            paths["meta_path"],
            {
                "campaign_name": campaign_name,
                "master_seed": master_seed,
                "source_total_cues": len(all_cues),
            },
        )
    if not Path(paths["index_path"]).exists():
        _atomic_write_json(
            paths["index_path"],
            {
                "completed_ids": [],
                "completed_count_by_cue": {},
                "total_completed": 0,
            },
        )
    if not Path(paths["ledger_path"]).exists():
        ensure_parent_dir(paths["ledger_path"])
        Path(paths["ledger_path"]).touch()
    return paths


def select_cue_window(master_cues_path: str, cue_start_index: int, cue_count: int) -> List[str]:
    cues = _read_json_list(master_cues_path)
    if cue_start_index < 0:
        cue_start_index = 0
    if cue_count <= 0:
        return []
    end = cue_start_index + cue_count
    return cues[cue_start_index:end]


def load_completed_ids(index_path: str) -> Set[str]:
    data = _read_json(index_path, {"completed_ids": []})
    ids = data.get("completed_ids", [])
    if not isinstance(ids, list):
        return set()
    return {str(x) for x in ids}


def append_completed_entries(
    ledger_path: str,
    index_path: str,
    rows: Iterable[Dict[str, Any]],
) -> int:
    index = _read_json(
        index_path,
        {"completed_ids": [], "completed_count_by_cue": {}, "total_completed": 0},
    )
    completed_ids = {str(x) for x in index.get("completed_ids", []) if x}
    completed_by_cue = index.get("completed_count_by_cue", {})
    if not isinstance(completed_by_cue, dict):
        completed_by_cue = {}

    added = 0
    ensure_parent_dir(ledger_path)
    with open(ledger_path, "a", encoding="utf-8") as ledger:
        for row in rows:
            custom_id = str(row.get("custom_id") or "")
            cue = str(row.get("cue") or "")
            if not custom_id or custom_id in completed_ids:
                continue
            completed_ids.add(custom_id)
            completed_by_cue[cue] = int(completed_by_cue.get(cue, 0)) + 1
            ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1

    index["completed_ids"] = sorted(completed_ids)
    index["completed_count_by_cue"] = completed_by_cue
    index["total_completed"] = len(completed_ids)
    _atomic_write_json(index_path, index)
    return added

