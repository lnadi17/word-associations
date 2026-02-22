from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any

from tqdm import tqdm

from lwow.clients.anthropic_client import AnthropicTextClient


_PUNCT_RE = re.compile(r"[\"'“”‘’`]")


def parse_associations(text: str, expected: int = 3) -> List[str]:
    if not text:
        return []
    cleaned = text.replace("\n", " ").strip()
    cleaned = _PUNCT_RE.sub("", cleaned)
    parts = [p.strip() for p in cleaned.split(",")]
    parts = [p for p in parts if p]
    return parts[:expected]


@dataclass
class GenerationRunner:
    client: AnthropicTextClient
    system_prompt: str
    repetitions_per_cue: int
    max_retries: int
    retry_backoff_sec: float

    def generate_for_cues(self, cues: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for cue in tqdm(cues, desc="Generating associations"):
            for trial in range(self.repetitions_per_cue):
                raw_text = ""
                parsed: List[str] = []
                ok = False
                for attempt in range(1, self.max_retries + 1):
                    raw_text = self.client.generate(self.system_prompt, cue)
                    parsed = parse_associations(raw_text)
                    if len(parsed) == 3:
                        ok = True
                        break
                    time.sleep(self.retry_backoff_sec * attempt)
                rows.append(
                    {
                        "cue": cue,
                        "trial": trial,
                        "raw_text": raw_text,
                        "parsed": "|".join(parsed),
                        "ok": ok,
                    }
                )
        return rows
