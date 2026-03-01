from __future__ import annotations

import io
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class OpenAITextClient:
    """HTTP client for OpenAI Chat Completions & Batch APIs.

    Exposes the same public interface as ``AnthropicTextClient`` so it can be
    used as a drop-in replacement inside ``GenerationRunner``.
    """

    model: str
    max_tokens: int
    temperature: float
    request_timeout_sec: float
    max_rate_limit_retries: int = 8
    max_backoff_sec: float = 30.0
    jitter_sec: float = 0.5
    api_base_url: str = "https://api.openai.com"
    api_key: Optional[str] = None
    reasoning_effort: Optional[str] = None

    def __post_init__(self) -> None:
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI generation.")
        self._api_key = key
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _request_with_retries(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        allow_404: bool = False,
        files: Optional[Dict[str, Any]] = None,
        raw_response: bool = False,
    ) -> Any:
        url = f"{self.api_base_url.rstrip('/')}{path}"
        attempt = 0
        while True:
            if files is not None:
                for _key, val in files.items():
                    fobj = val[1] if isinstance(val, tuple) else val
                    if hasattr(fobj, "seek"):
                        fobj.seek(0)

            try:
                kwargs: Dict[str, Any] = {
                    "method": method,
                    "url": url,
                    "timeout": self.request_timeout_sec,
                }
                if files is not None:
                    kwargs["headers"] = {"Authorization": f"Bearer {self._api_key}"}
                    kwargs["files"] = files
                    if payload:
                        kwargs["data"] = payload
                else:
                    kwargs["headers"] = self._headers()
                    if payload is not None:
                        kwargs["data"] = json.dumps(payload)

                response = self._session.request(**kwargs)
            except requests.RequestException as exc:
                if attempt >= self.max_rate_limit_retries:
                    raise RuntimeError(f"OpenAI request failed after retries: {exc}") from exc
                self._sleep_backoff(attempt)
                attempt += 1
                continue

            if response.status_code == 404 and allow_404:
                return {"_not_found": True}

            if response.status_code in (408, 409, 429, 500, 502, 503, 504):
                if attempt >= self.max_rate_limit_retries:
                    raise RuntimeError(
                        f"OpenAI request failed after retries ({response.status_code}): {response.text}"
                    )
                self._sleep_backoff(attempt)
                attempt += 1
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI request failed ({response.status_code}): {response.text}"
                )

            if raw_response:
                return response.text
            return response.json()

    def _sleep_backoff(self, attempt: int) -> None:
        base = min(self.max_backoff_sec, (2**attempt))
        jitter = random.uniform(0.0, self.jitter_sec) if self.jitter_sec > 0 else 0.0
        time.sleep(base + jitter)

    def build_message_params(
        self,
        system_prompt: str,
        cue: str,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input: {cue}\nOutput:"},
            ],
        }
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    def generate(self, system_prompt: str, cue: str) -> str:
        payload = self.build_message_params(system_prompt=system_prompt, cue=cue)
        result = self._request_with_retries("POST", "/v1/chat/completions", payload)
        choices = result.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def _upload_batch_file(self, jsonl_content: str) -> str:
        """Upload a JSONL string as a batch input file, return the file ID."""
        file_obj = io.BytesIO(jsonl_content.encode("utf-8"))
        result = self._request_with_retries(
            "POST",
            "/v1/files",
            payload={"purpose": "batch"},
            files={"file": ("batch_input.jsonl", file_obj, "application/jsonl")},
        )
        file_id = result.get("id")
        if not file_id:
            raise RuntimeError(f"OpenAI file upload returned no id: {result}")
        return file_id

    def submit_batch(self, requests_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        lines: List[str] = []
        for item in requests_payload:
            line = {
                "custom_id": item["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": item["params"],
            }
            lines.append(json.dumps(line, ensure_ascii=False))

        file_id = self._upload_batch_file("\n".join(lines))
        return self._request_with_retries(
            "POST",
            "/v1/batches",
            {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )

    def get_batch_status(self, batch_id: str, allow_not_found: bool = False) -> Dict[str, Any]:
        result = self._request_with_retries(
            "GET",
            f"/v1/batches/{batch_id}",
            allow_404=allow_not_found,
        )
        if result.get("_not_found"):
            return result

        # Map OpenAI status to the ``processing_status`` field that
        # ``GenerationRunner._collect_finished_batches`` checks.
        oai_status = result.get("status", "")
        terminal = {"completed", "failed", "expired", "cancelled"}
        result["processing_status"] = "ended" if oai_status in terminal else oai_status
        return result

    def _download_file_lines(self, file_id: str, allow_not_found: bool = False) -> List[Dict[str, Any]]:
        raw = self._request_with_retries(
            "GET",
            f"/v1/files/{file_id}/content",
            allow_404=allow_not_found,
            raw_response=True,
        )
        if isinstance(raw, dict) and raw.get("_not_found"):
            return []
        rows: List[Dict[str, Any]] = []
        for line in (raw or "").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    def get_batch_results(self, batch_id: str, allow_not_found: bool = False) -> Dict[str, Any]:
        status = self.get_batch_status(batch_id, allow_not_found=allow_not_found)
        if status.get("_not_found"):
            return status

        output_file_id = status.get("output_file_id")
        error_file_id = status.get("error_file_id")

        if not output_file_id and not error_file_id:
            if allow_not_found:
                return {"_not_found": True}
            return {"data": []}

        data: List[Dict[str, Any]] = []
        if output_file_id:
            for row in self._download_file_lines(output_file_id, allow_not_found):
                data.append(self._translate_result_item(row))
        if error_file_id:
            for row in self._download_file_lines(error_file_id, allow_not_found):
                data.append(self._translate_result_item(row))
        return {"data": data}

    @staticmethod
    def _translate_result_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """Translate an OpenAI batch result row into the format expected by
        ``_extract_result_data`` in generation.py."""
        custom_id = item.get("custom_id", "")
        response = item.get("response") if isinstance(item.get("response"), dict) else {}
        error = item.get("error")
        status_code = response.get("status_code", 0)
        body = response.get("body") if isinstance(response.get("body"), dict) else {}

        if error or status_code != 200:
            if isinstance(error, dict):
                error_msg = error.get("message", str(error))
            elif error:
                error_msg = str(error)
            else:
                error_msg = body.get("error", {}).get("message", f"HTTP {status_code}")
            return {
                "custom_id": custom_id,
                "result": {
                    "type": "errored",
                    "error": {"message": error_msg},
                },
            }

        choices = body.get("choices") or []
        text = choices[0].get("message", {}).get("content") or "" if choices else ""

        oai_usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        usage = {
            "input_tokens": oai_usage.get("prompt_tokens", 0),
            "output_tokens": oai_usage.get("completion_tokens", 0),
        }

        return {
            "custom_id": custom_id,
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": text}],
                    "usage": usage,
                },
            },
        }

    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        return self._request_with_retries("POST", f"/v1/batches/{batch_id}/cancel")
