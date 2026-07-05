#!/usr/bin/env python3
"""Check OpenAI batch statuses for open batches in a run. Diagnostic script."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lwow.clients.openai_client import OpenAITextClient
from lwow.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="data/runs/20260301-194228")
    parser.add_argument("--config", default="configs/lwow_gpt5_nano.yaml")
    parser.add_argument("--sample", type=int, default=20, help="Number of batches to sample (0 = all)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    run_dir = root / args.run_dir
    cp_path = run_dir / "checkpoint.json"
    if not cp_path.exists():
        print(f"Checkpoint not found: {cp_path}")
        sys.exit(1)

    cp = json.loads(cp_path.read_text())
    open_batches = cp.get("open_batches", {})
    batch_ids = list(open_batches.keys())
    if not batch_ids:
        print("No open batches.")
        return

    sample_n = args.sample if args.sample > 0 else len(batch_ids)
    to_check = batch_ids[:sample_n]

    cfg = load_config(str(root / args.config))
    g = cfg.generation
    client = OpenAITextClient(
        model=g.model,
        max_tokens=g.max_tokens,
        temperature=g.temperature,
        request_timeout_sec=g.request_timeout_sec,
        api_base_url=g.api_base_url or "https://api.openai.com",
        reasoning_effort=g.reasoning_effort or None,
    )

    def _fetch(bid: str) -> tuple[str, str, str | None]:
        try:
            st = client.get_batch_status(bid, allow_not_found=True)
            if st.get("_not_found"):
                return (bid, "_not_found", None)
            oai_status = st.get("status", "?")
            output_file = st.get("output_file_id") or "(none)"
            return (bid, oai_status, output_file)
        except Exception as e:
            return (bid, "_error", str(e))

    status_counts: Counter = Counter()
    results: list[tuple[str, str, str | None]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as ex:
        results = list(ex.map(_fetch, to_check))
    for _bid, status, extra in results:
        status_counts[status] += 1

    print(f"Checked {len(to_check)} of {len(batch_ids)} open batches\n")
    print("Status distribution:")
    for k, v in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print("\nSample results (first 10):")
    for bid, status, extra in results[:10]:
        short = bid[:36] + "..." if len(bid) > 36 else bid
        print(f"  {short} -> {status}  {extra or ''}")


if __name__ == "__main__":
    main()
