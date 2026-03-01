#!/usr/bin/env python3
"""Fetch in-flight batch results from OpenAI and merge into reconciled run.

Only adds results for custom_ids not already in the reconciled dataset.
Batches still running are added back to open_batches for resume.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lwow.config import load_config
from lwow.generation import _extract_result_data, parse_associations, _utc_now_iso


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    run_dir = root / "data/runs/20260228-142013"
    campaign_root = root / "data/campaigns/mini_campaign"
    backup_path = run_dir / "checkpoint.json.bak_reconcile_20260228T171405Z"

    if not backup_path.exists():
        print("Backup checkpoint not found")
        sys.exit(1)

    backup = json.loads(backup_path.read_text(encoding="utf-8"))
    open_batches_old = backup.get("open_batches") or {}
    if not open_batches_old:
        print("No open batches in backup")
        sys.exit(0)

    config = load_config(root / "configs/lwow_gpt5_mini.yaml")
    from lwow.clients.openai_client import OpenAITextClient

    client = OpenAITextClient(
        model=config.generation.model,
        max_tokens=config.generation.max_tokens,
        temperature=config.generation.temperature,
        request_timeout_sec=config.generation.request_timeout_sec,
        reasoning_effort=config.generation.reasoning_effort or None,
    )

    request_index = json.loads((run_dir / "request_index.json").read_text(encoding="utf-8"))
    by_id = {r["custom_id"]: (r["cue"], r["trial"]) for r in request_index.get("requests", [])}

    current_checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    completed_set = set(current_checkpoint.get("completed_ids", []))

    csv_path = root / "data/raw/gpt5_mini_responses.csv"
    rows_path = run_dir / "rows.jsonl"
    usage_path = run_dir / "usage.jsonl"
    ledger_path = campaign_root / "ledger.jsonl"
    index_path = campaign_root / "index.json"
    manifest_path = run_dir / "manifest.json"
    progress_path = run_dir / "progress.json"
    cost_path = run_dir / "cost_summary.json"
    checkpoint_path = run_dir / "checkpoint.json"

    # Load current CSV as base
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        existing_rows = list(csv.DictReader(f))
    existing_by_key = {(r["cue"], int(r.get("trial", 0))): r for r in existing_rows}

    new_open_batches = {}
    new_rows = []
    new_usage = []
    added_count = 0
    still_running_count = 0

    for batch_id, meta in open_batches_old.items():
        request_ids = meta.get("request_ids", [])
        status = client.get_batch_status(batch_id, allow_not_found=True)
        if status.get("_not_found"):
            continue
        proc = status.get("processing_status", "") or status.get("status", "")
        terminal = {"ended", "completed", "failed", "expired", "cancelled"}
        if proc not in terminal:
            new_open_batches[batch_id] = meta
            still_running_count += len(request_ids)
            continue

        results = client.get_batch_results(batch_id, allow_not_found=True)
        if results.get("_not_found"):
            continue
        data = results.get("data") or []
        for item in data:
            custom_id, usage, raw_text = _extract_result_data(item)
            if custom_id in completed_set:
                continue
            info = by_id.get(custom_id)
            if not info:
                continue
            cue, trial = info
            key = (cue, trial)
            if key in existing_by_key:
                continue
            parsed_list = parse_associations(raw_text)
            parsed = "|".join(parsed_list)
            ok = len(parsed_list) == 3
            row = {"cue": cue, "trial": trial, "raw_text": raw_text, "parsed": parsed, "ok": ok}
            new_rows.append(row)
            new_usage.append({"custom_id": custom_id, "usage": usage})
            existing_by_key[key] = row
            completed_set.add(custom_id)
            added_count += 1

    if not new_rows and not new_open_batches:
        print(json.dumps({"added": 0, "still_running": 0, "message": "Nothing to merge"}))
        return

    # Merge: existing + new, then cap at 100 per cue
    all_rows = existing_rows + new_rows
    by_cue = defaultdict(list)
    for r in all_rows:
        cue = (r.get("cue") or "").strip()
        if not cue:
            continue
        by_cue[cue].append(r)

    clean_rows = []
    for cue in sorted(by_cue.keys()):
        group = by_cue[cue]
        if len(group) < 100:
            continue
        for trial, r in enumerate(group[:100]):
            raw_text = r.get("raw_text") or ""
            parsed = r.get("parsed") or ""
            ok_val = r.get("ok")
            ok = ok_val is True or str(ok_val).strip().lower() in ("true", "1", "yes")
            clean_rows.append({"cue": cue, "trial": trial, "raw_text": raw_text, "parsed": parsed, "ok": ok})

    completed_ids = []
    for r in clean_rows:
        cid = hashlib.sha1(f"{r['cue']}|{r['trial']}".encode("utf-8")).hexdigest()[:16]
        completed_ids.append(cid)
    completed_set = set(completed_ids)
    now_iso = _utc_now_iso()

    # Append new usage to usage.jsonl (only for IDs in final completed set)
    for u in new_usage:
        if u["custom_id"] in completed_set:
            with usage_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**u, "updated_at": now_iso}, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cue", "trial", "raw_text", "parsed", "ok"])
        w.writeheader()
        w.writerows(clean_rows)

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

    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"completed_ids": sorted(completed_set)}, f, ensure_ascii=False, indent=2)

    with rows_path.open("w", encoding="utf-8") as f:
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

    # Recompute usage totals from usage.jsonl for completed_ids
    input_tot = 0.0
    output_tot = 0.0
    with usage_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("custom_id") not in completed_set:
                continue
            u = rec.get("usage") or {}
            input_tot += float(u.get("input_tokens") or 0)
            output_tot += float(u.get("output_tokens") or 0)

    total_requests = 600000
    completed = len(completed_set)
    remaining = total_requests - completed
    pct = (completed / total_requests * 100.0) if total_requests else 0.0
    pricing = {"input_per_mtok": 0.25, "output_per_mtok": 2.0}
    cost_usd = (input_tot / 1e6) * pricing["input_per_mtok"] + (output_tot / 1e6) * pricing["output_per_mtok"]

    checkpoint = {
        "open_batches": new_open_batches,
        "submitted_ids": sorted(completed_set | {rid for meta in new_open_batches.values() for rid in meta.get("request_ids", [])}),
        "completed_ids": sorted(completed_set),
        "failed_ids": [],
    }
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["updated_at"] = now_iso
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    progress = {
        "run_id": manifest["run_id"],
        "state": manifest.get("state", "running"),
        "updated_at": now_iso,
        "total_requests": total_requests,
        "submitted_requests": len(checkpoint["submitted_ids"]),
        "completed_requests": completed,
        "failed_requests": 0,
        "remaining_requests": remaining,
        "percent_complete": round(pct, 4),
        "usage_totals": {"input_tokens": input_tot, "output_tokens": output_tot},
        "estimated_cost_usd": round(cost_usd, 6),
        "open_batches": len(new_open_batches),
        "skipped_existing_requests": 0,
        "campaign_name": manifest.get("campaign_name", ""),
    }
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    cost_summary = {
        "run_id": manifest["run_id"],
        "updated_at": now_iso,
        "estimated_cost_usd": round(cost_usd, 6),
        "pricing": pricing,
        "usage_totals": {"input_tokens": input_tot, "output_tokens": output_tot},
    }
    with cost_path.open("w", encoding="utf-8") as f:
        json.dump(cost_summary, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "added_from_batches": added_count,
                "still_running_batches": len(new_open_batches),
                "still_running_requests": still_running_count,
                "total_rows_after": len(clean_rows),
                "completed_ids": completed,
                "remaining_requests": remaining,
                "estimated_cost_usd": round(cost_usd, 6),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
