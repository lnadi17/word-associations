from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lwow.clients.anthropic_client import AnthropicTextClient
from lwow.io import ensure_parent_dir


_PUNCT_RE = re.compile(r"[\"'“”‘’`]")


def parse_associations(text: str, expected: int = 3) -> List[str]:
    if not text:
        return []
    cleaned = text.replace("\n", " ").strip()
    cleaned = _PUNCT_RE.sub("", cleaned)
    parts = [p.strip() for p in cleaned.split(",")]
    parts = [p for p in parts if p]
    return parts[:expected]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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


def _append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(target, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _chunk(items: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    if chunk_size <= 0:
        chunk_size = 1
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def _extract_text_from_message(message: Dict[str, Any]) -> str:
    content = message.get("content") or []
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def _extract_result_data(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    custom_id = str(item.get("custom_id") or "")
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    result_type = result.get("type")

    if result_type == "succeeded":
        message = result.get("message") if isinstance(result.get("message"), dict) else {}
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        raw_text = _extract_text_from_message(message)
        return custom_id, usage, raw_text

    # Handle known non-success variants while keeping run progress moving.
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    error_text = str(error.get("message") or result_type or "unknown_batch_result")
    return custom_id, {}, error_text


def _estimate_cost_usd(totals: Dict[str, float], pricing: Dict[str, float]) -> float:
    input_tokens = totals.get("input_tokens", 0.0)
    output_tokens = totals.get("output_tokens", 0.0)
    cache_create_tokens = totals.get("cache_creation_input_tokens", 0.0)
    cache_read_tokens = totals.get("cache_read_input_tokens", 0.0)
    return (
        (input_tokens / 1_000_000.0) * pricing["input_per_mtok"]
        + (output_tokens / 1_000_000.0) * pricing["output_per_mtok"]
        + (cache_create_tokens / 1_000_000.0) * pricing["cache_creation_input_per_mtok"]
        + (cache_read_tokens / 1_000_000.0) * pricing["cache_read_input_per_mtok"]
    )


@dataclass
class GenerationRunner:
    client: AnthropicTextClient
    system_prompt: str
    repetitions_per_cue: int
    max_retries: int
    retry_backoff_sec: float
    batch_request_limit: int
    batch_poll_interval_sec: float
    batch_timeout_sec: float
    checkpoint_every_n_results: int
    enable_prompt_caching: bool
    cache_control_type: str
    pricing: Dict[str, float]

    def _make_request_index(self, cues: List[str]) -> List[Dict[str, Any]]:
        requests_index: List[Dict[str, Any]] = []
        for cue_idx, cue in enumerate(cues):
            for trial in range(self.repetitions_per_cue):
                raw_id = f"{cue_idx}:{trial}:{cue}"
                custom_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
                requests_index.append(
                    {
                        "custom_id": custom_id,
                        "cue": cue,
                        "trial": trial,
                    }
                )
        return requests_index

    def _load_or_create_manifest(
        self,
        run_dir: str | Path,
        run_id: str,
        config_path: str,
        raw_output_path: str,
        cues: List[str],
    ) -> Dict[str, Any]:
        run_path = Path(run_dir)
        manifest_path = run_path / "manifest.json"
        request_index_path = run_path / "request_index.json"
        checkpoint_path = run_path / "checkpoint.json"
        rows_jsonl_path = run_path / "rows.jsonl"
        usage_jsonl_path = run_path / "usage.jsonl"
        progress_path = run_path / "progress.json"
        cost_path = run_path / "cost_summary.json"

        existing = _read_json(manifest_path, {})
        if existing:
            return existing

        requests_index = self._make_request_index(cues)
        _atomic_write_json(request_index_path, {"requests": requests_index})
        _atomic_write_json(
            checkpoint_path,
            {
                "open_batches": {},
                "submitted_ids": [],
                "completed_ids": [],
                "failed_ids": [],
            },
        )
        _atomic_write_json(
            manifest_path,
            {
                "run_id": run_id,
                "state": "running",
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "config_path": config_path,
                "run_dir": str(run_path),
                "request_index_path": str(request_index_path),
                "checkpoint_path": str(checkpoint_path),
                "rows_jsonl_path": str(rows_jsonl_path),
                "usage_jsonl_path": str(usage_jsonl_path),
                "progress_path": str(progress_path),
                "cost_path": str(cost_path),
                "raw_output_path": raw_output_path,
                "total_requests": len(requests_index),
            },
        )
        return _read_json(manifest_path, {})

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        manifest["updated_at"] = _utc_now_iso()
        _atomic_write_json(manifest["run_dir"] + "/manifest.json", manifest)

    def _write_progress(
        self,
        manifest: Dict[str, Any],
        checkpoint: Dict[str, Any],
        totals: Dict[str, float],
    ) -> Dict[str, Any]:
        total = int(manifest.get("total_requests", 0))
        completed = len(checkpoint.get("completed_ids", []))
        failed = len(checkpoint.get("failed_ids", []))
        submitted = len(checkpoint.get("submitted_ids", []))
        remaining = max(total - completed - failed, 0)
        pct = (completed / total * 100.0) if total else 0.0
        cost_usd = _estimate_cost_usd(totals=totals, pricing=self.pricing)
        progress = {
            "run_id": manifest.get("run_id"),
            "state": manifest.get("state"),
            "updated_at": _utc_now_iso(),
            "total_requests": total,
            "submitted_requests": submitted,
            "completed_requests": completed,
            "failed_requests": failed,
            "remaining_requests": remaining,
            "percent_complete": round(pct, 4),
            "usage_totals": totals,
            "estimated_cost_usd": round(cost_usd, 6),
            "open_batches": len(checkpoint.get("open_batches", {})),
        }
        _atomic_write_json(manifest["progress_path"], progress)
        _atomic_write_json(
            manifest["cost_path"],
            {
                "run_id": manifest.get("run_id"),
                "updated_at": progress["updated_at"],
                "estimated_cost_usd": progress["estimated_cost_usd"],
                "pricing": self.pricing,
                "usage_totals": totals,
            },
        )
        return progress

    def _export_rows_csv(self, manifest: Dict[str, Any]) -> None:
        rows = _read_jsonl(manifest["rows_jsonl_path"])
        rows_sorted = sorted(rows, key=lambda x: (str(x.get("cue", "")), int(x.get("trial", 0))))
        output_path = manifest["raw_output_path"]
        ensure_parent_dir(output_path)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["cue", "trial", "raw_text", "parsed", "ok"],
            )
            writer.writeheader()
            for row in rows_sorted:
                writer.writerow(
                    {
                        "cue": row.get("cue", ""),
                        "trial": row.get("trial", 0),
                        "raw_text": row.get("raw_text", ""),
                        "parsed": row.get("parsed", ""),
                        "ok": bool(row.get("ok", False)),
                    }
                )

    def _load_usage_totals(self, usage_jsonl_path: str) -> Dict[str, float]:
        totals = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "cache_creation_input_tokens": 0.0,
            "cache_read_input_tokens": 0.0,
        }
        for row in _read_jsonl(usage_jsonl_path):
            usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
            totals["input_tokens"] += float(usage.get("input_tokens", 0) or 0)
            totals["output_tokens"] += float(usage.get("output_tokens", 0) or 0)
            totals["cache_creation_input_tokens"] += float(
                usage.get("cache_creation_input_tokens", 0) or 0
            )
            totals["cache_read_input_tokens"] += float(usage.get("cache_read_input_tokens", 0) or 0)
        return totals

    def _submit_pending_batches(
        self,
        manifest: Dict[str, Any],
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_index = _read_json(manifest["request_index_path"], {"requests": []}).get("requests", [])
        submitted_ids = set(checkpoint.get("submitted_ids", []))
        completed_ids = set(checkpoint.get("completed_ids", []))
        failed_ids = set(checkpoint.get("failed_ids", []))
        pending = [
            row
            for row in request_index
            if row.get("custom_id")
            and row["custom_id"] not in submitted_ids
            and row["custom_id"] not in completed_ids
            and row["custom_id"] not in failed_ids
        ]
        if not pending:
            return checkpoint

        for batch_items in _chunk(pending, self.batch_request_limit):
            latest_manifest = _read_json(manifest["run_dir"] + "/manifest.json", manifest)
            if latest_manifest.get("state") == "paused":
                manifest["state"] = "paused"
                return checkpoint

            batch_payload: List[Dict[str, Any]] = []
            for item in batch_items:
                params = self.client.build_message_params(
                    system_prompt=self.system_prompt,
                    cue=str(item["cue"]),
                    enable_prompt_caching=self.enable_prompt_caching,
                    cache_control_type=self.cache_control_type,
                )
                batch_payload.append(
                    {
                        "custom_id": item["custom_id"],
                        "params": params,
                    }
                )
            created = self.client.submit_batch(batch_payload)
            batch_id = str(created.get("id") or "")
            if not batch_id:
                raise RuntimeError(f"Unexpected batch response, missing id: {created}")
            checkpoint.setdefault("open_batches", {})
            checkpoint["open_batches"][batch_id] = {
                "submitted_at": _utc_now_iso(),
                "request_ids": [item["custom_id"] for item in batch_items],
            }
            checkpoint.setdefault("submitted_ids", [])
            checkpoint["submitted_ids"].extend([item["custom_id"] for item in batch_items])
            _atomic_write_json(manifest["checkpoint_path"], checkpoint)
        return checkpoint

    def _collect_finished_batches(
        self,
        manifest: Dict[str, Any],
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        open_batches = dict(checkpoint.get("open_batches", {}))
        if not open_batches:
            return checkpoint

        request_index = _read_json(manifest["request_index_path"], {"requests": []}).get("requests", [])
        by_id = {row["custom_id"]: row for row in request_index if row.get("custom_id")}
        completed_ids = set(checkpoint.get("completed_ids", []))
        failed_ids = set(checkpoint.get("failed_ids", []))
        processed_since_flush = 0

        for batch_id in list(open_batches.keys()):
            status = self.client.get_batch_status(batch_id)
            processing_status = str(status.get("processing_status") or "")
            if processing_status and processing_status not in ("ended", "completed", "succeeded"):
                continue

            results_payload = self.client.get_batch_results(batch_id)
            results = results_payload.get("data") if isinstance(results_payload.get("data"), list) else []
            for result_item in results:
                if not isinstance(result_item, dict):
                    continue
                custom_id, usage, raw_text = _extract_result_data(result_item)
                if not custom_id or custom_id in completed_ids or custom_id in failed_ids:
                    continue
                request_row = by_id.get(custom_id)
                if not request_row:
                    failed_ids.add(custom_id)
                    continue

                parsed = parse_associations(raw_text)
                ok = len(parsed) == 3
                _append_jsonl(
                    manifest["rows_jsonl_path"],
                    {
                        "custom_id": custom_id,
                        "cue": request_row["cue"],
                        "trial": request_row["trial"],
                        "raw_text": raw_text,
                        "parsed": "|".join(parsed),
                        "ok": ok,
                        "updated_at": _utc_now_iso(),
                    },
                )
                _append_jsonl(
                    manifest["usage_jsonl_path"],
                    {
                        "custom_id": custom_id,
                        "usage": usage,
                        "updated_at": _utc_now_iso(),
                    },
                )
                completed_ids.add(custom_id)
                processed_since_flush += 1
                if processed_since_flush >= self.checkpoint_every_n_results:
                    checkpoint["completed_ids"] = sorted(completed_ids)
                    checkpoint["failed_ids"] = sorted(failed_ids)
                    _atomic_write_json(manifest["checkpoint_path"], checkpoint)
                    processed_since_flush = 0

            checkpoint["open_batches"].pop(batch_id, None)
            checkpoint["completed_ids"] = sorted(completed_ids)
            checkpoint["failed_ids"] = sorted(failed_ids)
            _atomic_write_json(manifest["checkpoint_path"], checkpoint)
        return checkpoint

    def run_to_completion(
        self,
        run_dir: str | Path,
        run_id: str,
        config_path: str,
        raw_output_path: str,
        cues: List[str],
    ) -> Dict[str, Any]:
        manifest = self._load_or_create_manifest(
            run_dir=run_dir,
            run_id=run_id,
            config_path=config_path,
            raw_output_path=raw_output_path,
            cues=cues,
        )
        manifest["state"] = "running"
        self._write_manifest(manifest)

        started = time.time()
        while True:
            manifest = _read_json(manifest["run_dir"] + "/manifest.json", manifest)
            checkpoint = _read_json(
                manifest["checkpoint_path"],
                {"open_batches": {}, "submitted_ids": [], "completed_ids": [], "failed_ids": []},
            )
            if manifest.get("state") == "paused":
                totals = self._load_usage_totals(manifest["usage_jsonl_path"])
                progress = self._write_progress(manifest, checkpoint, totals)
                return progress

            checkpoint = self._submit_pending_batches(manifest, checkpoint)
            checkpoint = self._collect_finished_batches(manifest, checkpoint)
            totals = self._load_usage_totals(manifest["usage_jsonl_path"])
            progress = self._write_progress(manifest, checkpoint, totals)

            if progress["remaining_requests"] <= 0 and progress["open_batches"] == 0:
                manifest["state"] = "completed"
                self._write_manifest(manifest)
                self._export_rows_csv(manifest)
                progress = self._write_progress(manifest, checkpoint, totals)
                return progress

            if (time.time() - started) >= self.batch_timeout_sec:
                manifest["state"] = "timed_out"
                self._write_manifest(manifest)
                progress = self._write_progress(manifest, checkpoint, totals)
                return progress

            time.sleep(self.batch_poll_interval_sec)

    @staticmethod
    def pause_run(run_dir: str | Path, cancel_remote: bool = False, client: Optional[AnthropicTextClient] = None) -> None:
        run_path = Path(run_dir)
        manifest_path = run_path / "manifest.json"
        manifest = _read_json(manifest_path, {})
        if not manifest:
            raise ValueError(f"Run manifest not found at {manifest_path}")
        manifest["state"] = "paused"
        _atomic_write_json(manifest_path, manifest)

        if cancel_remote and client:
            checkpoint = _read_json(manifest.get("checkpoint_path", ""), {"open_batches": {}})
            for batch_id in list(checkpoint.get("open_batches", {}).keys()):
                try:
                    client.cancel_batch(batch_id)
                except Exception:
                    continue

    @staticmethod
    def load_progress(run_dir: str | Path) -> Dict[str, Any]:
        run_path = Path(run_dir)
        manifest = _read_json(run_path / "manifest.json", {})
        if not manifest:
            raise ValueError(f"Run manifest not found at {run_path / 'manifest.json'}")
        progress = _read_json(manifest.get("progress_path", ""), {})
        if progress:
            return progress
        checkpoint = _read_json(
            manifest["checkpoint_path"],
            {"open_batches": {}, "submitted_ids": [], "completed_ids": [], "failed_ids": []},
        )
        total = int(manifest.get("total_requests", 0))
        completed = len(checkpoint.get("completed_ids", []))
        failed = len(checkpoint.get("failed_ids", []))
        submitted = len(checkpoint.get("submitted_ids", []))
        remaining = max(total - completed - failed, 0)
        return {
            "run_id": manifest.get("run_id"),
            "state": manifest.get("state"),
            "total_requests": total,
            "submitted_requests": submitted,
            "completed_requests": completed,
            "failed_requests": failed,
            "remaining_requests": remaining,
            "percent_complete": round((completed / total * 100.0) if total else 0.0, 4),
            "estimated_cost_usd": 0.0,
            "open_batches": len(checkpoint.get("open_batches", {})),
        }
