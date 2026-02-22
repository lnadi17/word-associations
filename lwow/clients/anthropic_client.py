from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import anthropic


@dataclass
class AnthropicTextClient:
    model: str
    max_tokens: int
    temperature: float
    request_timeout_sec: float
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude generation.")
        self._client = anthropic.Anthropic(api_key=key)

    def generate(self, system_prompt: str, cue: str) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Input: {cue}\nOutput:"},
            ],
            timeout=self.request_timeout_sec,
        )
        return message.content[0].text if message.content else ""
