#!/usr/bin/env python3
"""Sync raw CSV to campaign ledger and index (exactly 100 rows per cue).

Usage:
  python scripts/sync_csv_to_campaign.py --csv data/raw/gpt5_nano_responses.csv --campaign gpt5_nano
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Sync raw CSV to campaign")
    parser.add_argument("--csv", required=True, help="Path to raw responses CSV")
    parser.add_argument("--campaign", required=True, help="Campaign name (e.g. gpt5_nano)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    csv_path = root / args.csv
    campaign_root = root / "data/campaigns" / args.campaign
    ledger_path = campaign_root / "ledger.jsonl"
    index_path = campaign_root / "index.json"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cue = (row.get("cue") or "").strip()
            if not cue:
                continue
            trial = int(row.get("trial", 0))
            raw_text = (row.get("raw_text") or "").replace("\n", " ").replace("\r", " ")
            parsed = (row.get("parsed") or "").replace("\n", " ").replace("\r", " ")
            ok_val = row.get("ok")
            ok = ok_val is True or str(ok_val).strip().lower() in ("true", "1", "yes")
            rows.append((cue, trial, {"raw_text": raw_text, "parsed": parsed, "ok": ok}))

    by_cue = defaultdict(list)
    for cue, trial, data in rows:
        by_cue[cue].append((trial, data))

    clean_rows = []
    for cue in sorted(by_cue.keys()):
        group = by_cue[cue]
        group.sort(key=lambda x: x[0])
        for trial, data in group[:100]:
            clean_rows.append({
                "cue": cue,
                "trial": trial,
                "raw_text": data["raw_text"],
                "parsed": data["parsed"],
                "ok": data["ok"],
            })

    completed_ids = []
    for r in clean_rows:
        cid = hashlib.sha1(f"{r['cue']}|{r['trial']}".encode("utf-8")).hexdigest()[:16]
        completed_ids.append(cid)
    completed_set = set(completed_ids)
    now_iso = _utc_now_iso()
    completed_by_cue = defaultdict(int)
    for r in clean_rows:
        completed_by_cue[r["cue"]] += 1

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
                "ledger_path": str(ledger_path),
                "index_path": str(index_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
