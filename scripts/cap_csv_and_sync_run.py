#!/usr/bin/env python3
"""Cap CSV to 100 rows per cue and sync to a run directory.

1. Reads CSV, caps each cue to at most 100 rows (keeps first 100 by trial).
2. Writes capped CSV back to raw output path.
3. Syncs to run: writes rows.jsonl, updates checkpoint completed_ids.

Usage:
  python scripts/cap_csv_and_sync_run.py --csv data/raw/gpt5_nano_responses.csv --run-dir data/runs/20260301-194228
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
    parser = argparse.ArgumentParser(
        description="Cap CSV to 100 rows per cue and sync to run directory"
    )
    parser.add_argument("--csv", required=True, help="Path to raw responses CSV")
    parser.add_argument("--run-dir", required=True, help="Run directory (e.g. data/runs/20260301-194228)")
    parser.add_argument("--max-per-cue", type=int, default=100, help="Max rows per cue (default: 100)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    csv_path = root / args.csv
    run_dir = root / args.run_dir
    max_per_cue = args.max_per_cue

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_output_path = root / manifest.get("raw_output_path", "")
    checkpoint_path = run_dir / "checkpoint.json"
    rows_jsonl_path = run_dir / "rows.jsonl"

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
        for trial, data in group[:max_per_cue]:
            clean_rows.append({
                "cue": cue,
                "trial": trial,
                "raw_text": data["raw_text"],
                "parsed": data["parsed"],
                "ok": data["ok"],
            })

    fieldnames = ["cue", "trial", "raw_text", "parsed", "ok"]
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)
    print(f"Capped CSV written: {raw_output_path} ({len(clean_rows)} rows)")

    now_iso = _utc_now_iso()
    completed_ids = []
    rows_for_jsonl = []
    for r in clean_rows:
        cid = hashlib.sha1(f"{r['cue']}|{r['trial']}".encode("utf-8")).hexdigest()[:16]
        completed_ids.append(cid)
        rows_for_jsonl.append({
            "custom_id": cid,
            "cue": r["cue"],
            "trial": r["trial"],
            "raw_text": r["raw_text"],
            "parsed": r["parsed"],
            "ok": bool(r["ok"]),
            "updated_at": now_iso,
        })

    with rows_jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows_for_jsonl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"rows.jsonl written: {rows_jsonl_path} ({len(rows_for_jsonl)} rows)")

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["completed_ids"] = sorted(set(completed_ids))
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"checkpoint.json updated: {len(checkpoint['completed_ids'])} completed_ids")

    cues_full = len([c for c, g in by_cue.items() if len(g) >= max_per_cue])
    print(
        json.dumps(
            {
                "total_rows": len(clean_rows),
                "cues_with_full_count": cues_full,
                "total_cues": len(by_cue),
                "max_per_cue": max_per_cue,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
