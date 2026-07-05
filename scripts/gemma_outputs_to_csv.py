#!/usr/bin/env python3
"""
Transform data/raw/gemma_outputs (cue folders, one file per response) into
a standard raw CSV (cue,trial,raw_text,parsed,ok) compatible with the LWOW
pipeline.

Strategy for problematic files:
- Empty or whitespace-only: parsed="", ok=False
- Multi-line: parse first line only (often valid response on line 1, code on rest)
- Parse fails or yields != 3 tokens: parsed="", ok=False
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lwow.generation import parse_associations
from lwow.io import write_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "gemma_outputs"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "gemma_responses.csv"
FIELDNAMES = ["cue", "trial", "raw_text", "parsed", "ok"]


def _parse_file_content(content: str) -> tuple[str, str, bool]:
    """
    Parse file content into (raw_text, parsed, ok).

    Uses first line only for parsing (multi-line files often have valid
    response on line 1, code/explanation on rest). Empty/invalid → parsed="", ok=False.
    """
    raw = content.strip()
    if not raw:
        return "", "", False

    first_line = raw.split("\n")[0].strip()
    parsed_tokens = parse_associations(first_line)
    ok = len(parsed_tokens) == 3
    parsed_str = "|".join(parsed_tokens) if ok else ""

    return raw, parsed_str, ok


def _collect_rows(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    cue_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    for cue_dir in cue_dirs:
        cue = cue_dir.name
        txt_files = sorted(cue_dir.glob("*.txt"), key=_file_trial_key)
        for f in txt_files:
            try:
                trial = int(f.stem)
            except ValueError:
                trial = -1
            content = f.read_text(encoding="utf-8", errors="replace")
            raw_text, parsed, ok = _parse_file_content(content)
            rows.append({
                "cue": cue,
                "trial": trial,
                "raw_text": raw_text,
                "parsed": parsed,
                "ok": str(ok),
            })
    return rows


def _file_trial_key(path: Path) -> int:
    """Sort trial filenames numerically (0, 1, 2, ..., 10, 11)."""
    try:
        return int(path.stem)
    except ValueError:
        return 999_999


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform gemma_outputs folder structure to raw CSV.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print summary stats",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input not a directory: {args.input}")

    rows = _collect_rows(args.input)
    write_csv(args.output, FIELDNAMES, rows)

    if args.verbose:
        ok_count = sum(1 for r in rows if r["ok"] == "True")
        empty_count = sum(1 for r in rows if not r["raw_text"])
        print(f"Cues: {len({r['cue'] for r in rows})}")
        print(f"Total rows: {len(rows)}")
        print(f"Valid (ok=True): {ok_count}")
        print(f"Empty files: {empty_count}")
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
