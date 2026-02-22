from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


def _normalize_token(token: str) -> str:
    token = token.replace("_", " ").strip().lower()
    return token


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


@dataclass
class ProcessingRunner:
    repetitions_per_cue: int
    seed: int

    def process_rows(self, rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            cue = _normalize_token(row["cue"])
            parsed = row.get("parsed", "")
            tokens = [_normalize_token(t) for t in parsed.split("|") if t]
            tokens = [t for t in tokens if t and t != cue]
            tokens = _dedupe_preserve_order(tokens)
            grouped[cue].append(
                {
                    "cue": cue,
                    "trial": int(row.get("trial", 0)),
                    "responses": tokens,
                    "raw_text": row.get("raw_text", ""),
                }
            )

        rng = random.Random(self.seed)
        processed: List[Dict[str, Any]] = []
        for cue, cue_rows in grouped.items():
            if len(cue_rows) > self.repetitions_per_cue:
                cue_rows = rng.sample(cue_rows, self.repetitions_per_cue)
            while len(cue_rows) < self.repetitions_per_cue:
                cue_rows.append(
                    {
                        "cue": cue,
                        "trial": len(cue_rows),
                        "responses": [],
                        "raw_text": "",
                    }
                )
            processed.extend(cue_rows)

        return processed

    @staticmethod
    def build_edge_counts(processed_rows: List[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
        counter: Counter[Tuple[str, str]] = Counter()
        for row in processed_rows:
            cue = row["cue"]
            for response in row.get("responses", []):
                if response:
                    counter[(cue, response)] += 1
        return [(src, tgt, wt) for (src, tgt), wt in counter.items()]
