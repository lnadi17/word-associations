#!/usr/bin/env python3
"""Sync run output to CSV and campaign (exactly 100 rows per cue).

Reads rows from run's rows.jsonl, groups by cue, caps at 100 per cue,
writes CSV + campaign ledger + index.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    run_dir = root / "data/runs/20260228-142013"
    campaign_root = root / "data/campaigns/mini_campaign"
    csv_path = root / "data/raw/gpt5_mini_responses.csv"
    rows_path = run_dir / "rows.jsonl"
    ledger_path = campaign_root / "ledger.jsonl"
    index_path = campaign_root / "index.json"

    if not rows_path.exists():
        print(f"Rows not found: {rows_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with rows_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    by_cue = defaultdict(list)
    for r in rows:
        cue = (r.get("cue") or "").strip()
        if not cue:
            continue
        trial = int(r.get("trial", 0))
        by_cue[cue].append((trial, r))

    clean_rows = []
    for cue in sorted(by_cue.keys()):
        group = by_cue[cue]
        group.sort(key=lambda x: x[0])
        for trial, r in group[:100]:
            raw_text = (r.get("raw_text") or "").replace("\n", " ").replace("\r", " ")
            parsed = (r.get("parsed") or "").replace("\n", " ").replace("\r", " ")
            ok_val = r.get("ok")
            ok = ok_val is True or str(ok_val).strip().lower() in ("true", "1", "yes")
            clean_rows.append({"cue": cue, "trial": trial, "raw_text": raw_text, "parsed": parsed, "ok": ok})

    completed_ids = []
    for r in clean_rows:
        cid = hashlib.sha1(f"{r['cue']}|{r['trial']}".encode("utf-8")).hexdigest()[:16]
        completed_ids.append(cid)
    completed_set = set(completed_ids)
    now_iso = _utc_now_iso()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cue", "trial", "raw_text", "parsed", "ok"])
        w.writeheader()
        w.writerows(clean_rows)

    campaign_root.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8") as f:
        for r in clean_rows:
            cid = hashlib.sha1(f"{r['cue']}|{r['trial']}".encode("utf-8")).hexdigest()[:16]
            f.write(
                json.dumps(
                    {
                        "custom_id": cid,
                        "cue": r["cue"],
                        "trial": r["trial"],
                        "raw_text": r["raw_text"],
                        "parsed": r["parsed"],
                        "ok": bool(r["ok"]),
                        "updated_at": now_iso,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    completed_by_cue = defaultdict(int)
    for r in clean_rows:
        completed_by_cue[r["cue"]] += 1

    index = {
        "completed_ids": sorted(completed_set),
        "completed_count_by_cue": dict(completed_by_cue),
        "total_completed": len(completed_set),
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    cues_full = len([c for c, n in completed_by_cue.items() if n >= 100])
    print(
        json.dumps(
            {
                "total_rows": len(clean_rows),
                "cues_with_100": cues_full,
                "total_completed_ids": len(completed_set),
                "csv_path": str(csv_path),
                "ledger_path": str(ledger_path),
                "index_path": str(index_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
