from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class AnthropicTextClient:
    model: str
    max_tokens: int
    temperature: float
    request_timeout_sec: float
    max_rate_limit_retries: int = 8
    max_backoff_sec: float = 30.0
    jitter_sec: float = 0.5
    api_base_url: str = "https://api.anthropic.com"
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude generation.")
        self._api_key = key
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _request_with_retries(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        allow_404: bool = False,
        expect_jsonl_results: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.api_base_url.rstrip('/')}{path}"
        attempt = 0
        while True:
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    data=json.dumps(payload) if payload is not None else None,
                    timeout=self.request_timeout_sec,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_rate_limit_retries:
                    raise RuntimeError(f"Anthropic request failed after retries: {exc}") from exc
                self._sleep_backoff(attempt)
                attempt += 1
                continue

            if response.status_code == 404 and allow_404:
                return {"_not_found": True}

            if response.status_code in (408, 409, 429, 500, 502, 503, 504):
                if attempt >= self.max_rate_limit_retries:
                    raise RuntimeError(
                        f"Anthropic request failed after retries ({response.status_code}): {response.text}"
                    )
                self._sleep_backoff(attempt)
                attempt += 1
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Anthropic request failed ({response.status_code}): {response.text}"
                )
            if expect_jsonl_results:
                return self._parse_batch_results_response(response.text)
            return response.json()

    def _parse_batch_results_response(self, payload_text: str) -> Dict[str, Any]:
        text = (payload_text or "").strip()
        if not text:
            return {"data": []}

        # Some environments return wrapped JSON, others return JSONL/NDJSON.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = parsed.get("data")
                if isinstance(data, list):
                    return {"data": data}
                # A single result object as JSON.
                return {"data": [parsed]}
            if isinstance(parsed, list):
                return {"data": parsed}
        except json.JSONDecodeError:
            pass

        data: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                data.append(row)
        return {"data": data}

    def _sleep_backoff(self, attempt: int) -> None:
        base = min(self.max_backoff_sec, (2**attempt))
        jitter = random.uniform(0.0, self.jitter_sec) if self.jitter_sec > 0 else 0.0
        time.sleep(base + jitter)

    def build_message_params(
        self,
        system_prompt: str,
        cue: str,
    ) -> Dict[str, Any]:
        system_blocks: List[Dict[str, Any]] = [{"type": "text", "text": system_prompt}]
        user_blocks: List[Dict[str, Any]] = [{"type": "text", "text": f"Input: {cue}\nOutput:"}]

        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_blocks}],
        }

    def generate(self, system_prompt: str, cue: str) -> str:
        payload = self.build_message_params(system_prompt=system_prompt, cue=cue)
        message = self._request_with_retries("POST", "/v1/messages", payload)
        content = message.get("content") or []
        if not content:
            return ""
        return content[0].get("text", "")

    def submit_batch(self, requests_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {"requests": requests_payload}
        return self._request_with_retries("POST", "/v1/messages/batches", payload)

    def get_batch_status(self, batch_id: str, allow_not_found: bool = False) -> Dict[str, Any]:
        return self._request_with_retries(
            "GET",
            f"/v1/messages/batches/{batch_id}",
            allow_404=allow_not_found,
        )

    def get_batch_results(self, batch_id: str, allow_not_found: bool = False) -> Dict[str, Any]:
        return self._request_with_retries(
            "GET",
            f"/v1/messages/batches/{batch_id}/results",
            allow_404=allow_not_found,
            expect_jsonl_results=True,
        )

    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        return self._request_with_retries("POST", f"/v1/messages/batches/{batch_id}/cancel", {})
