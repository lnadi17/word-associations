#!/usr/bin/env python3
"""
One-off script: normalize responses CSV.

1. Replace multiline/empty rows: raw_text and parsed set to empty, ok to False when:
   - parsed is empty (after strip), OR
   - raw_text contains newlines (multiline/code)

2. Strip "Output:" prefix from raw_text and parsed when present.

Writes to a new file; does not overwrite the original unless --in-place is used.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace multiline/empty-parsed rows with empty values in responses CSV."
    )
    parser.add_argument("csv_path", help="Path to responses CSV (cue,trial,raw_text,parsed,ok)")
    parser.add_argument(
        "-o", "--output",
        help="Output path (default: same dir with _normalized suffix)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file (use with caution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count changes only, do not write",
    )
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    if args.in_place and args.output:
        print("Error: cannot use both --in-place and -o", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else path.parent / f"{path.stem}_normalized{path.suffix}"
    if args.in_place:
        out_path = path

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["cue", "trial", "raw_text", "parsed", "ok"]
        rows = list(reader)

    def strip_output_prefix(s: str) -> str:
        """Remove leading 'Output:' or 'Output: ' from string."""
        if not s:
            return s
        stripped = s.strip()
        if stripped.lower().startswith("output:"):
            return stripped[7:].strip()
        return s

    replaced = 0
    output_stripped = 0
    new_rows = []
    for row in rows:
        row = dict(row)
        raw = row.get("raw_text") or ""
        parsed = row.get("parsed") or ""
        parsed_stripped = parsed.strip()
        raw_has_newlines = "\n" in raw

        if not parsed_stripped or raw_has_newlines:
            row["raw_text"] = ""
            row["parsed"] = ""
            row["ok"] = "False"
            replaced += 1
        else:
            # Strip "Output:" prefix from raw_text and parsed
            new_raw = strip_output_prefix(raw)
            new_parsed = strip_output_prefix(parsed)
            if new_raw != raw or new_parsed != parsed:
                row["raw_text"] = new_raw
                row["parsed"] = new_parsed
                output_stripped += 1
        new_rows.append(row)

    print(f"Rows processed: {len(rows):,}")
    print(f"Rows replaced with empty: {replaced:,}")
    print(f"'Output:' prefix stripped: {output_stripped:,}")

    if args.dry_run:
        print("Dry run: no file written.")
        return

    if args.in_place:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        print(f"Written in-place: {out_path}")
    else:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
