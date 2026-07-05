from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lwow.io import ensure_parent_dir, read_csv


# Approximate pricing per 1M tokens (USD)
OPENAI_EMBEDDING_PRICES: Dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def _dataset_name_from_csv_path(csv_path: str | Path) -> str:
    """Derive dataset name from edgelist CSV path, e.g. gpt5_mini_edges.csv -> gpt5_mini."""
    stem = Path(csv_path).stem
    for suffix in ("_edges", "_edgelist"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _model_to_suffix(model: str) -> str:
    """Return filename suffix for embedding model: 'small' or 'large'."""
    if "large" in model.lower():
        return "large"
    return "small"


class EmbeddingStore:
    """Manages a folder with words.txt, config.json, and embeddings.csv (or _small/_large variants)."""

    def __init__(
        self,
        path: str | Path,
        *,
        embeddings_filename: Optional[str] = None,
        config_filename: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.embeddings_filename = embeddings_filename or "embeddings.csv"
        self.config_filename = config_filename or "config.json"
        self.words_path = self.path / "words.txt"
        self.config_path = self.path / self.config_filename
        self.embeddings_path = self.path / self.embeddings_filename

    def ensure_from_edgelist(
        self,
        csv_path: str | Path,
        *,
        model: str = "text-embedding-3-small",
        overwrite_words: bool = False,
    ) -> int:
        """
        On first run: read edgelist CSV, collect unique src+tgt, write words.txt,
        create empty embeddings.csv with header, init config. Returns total unique words.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        csv_path = Path(csv_path)

        rows = read_csv(csv_path)
        words_set: set[str] = set()
        for row in rows:
            src = (row.get("src") or "").strip()
            tgt = (row.get("tgt") or "").strip()
            if src:
                words_set.add(src)
            if tgt:
                words_set.add(tgt)

        words_list = sorted(words_set)
        total = len(words_list)

        if overwrite_words or not self.words_path.exists():
            ensure_parent_dir(self.words_path)
            with open(self.words_path, "w", encoding="utf-8") as f:
                for w in words_list:
                    f.write(w + "\n")

        if not self.embeddings_path.exists():
            ensure_parent_dir(self.embeddings_path)
            with open(self.embeddings_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["word", "vector", "tokens"])

        if not self.config_path.exists():
            self._write_config(
                model=model,
                prompt_tokens=0,
                total_spend_usd=0.0,
                source_csv=str(csv_path),
            )

        return total

    def _write_config(
        self,
        *,
        model: str,
        prompt_tokens: int,
        total_spend_usd: float,
        source_csv: Optional[str] = None,
        preserve_existing: bool = False,
    ) -> None:
        if preserve_existing and self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config.update(
                model=model,
                prompt_tokens=prompt_tokens,
                total_spend_usd=total_spend_usd,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        else:
            config = {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "total_spend_usd": total_spend_usd,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if source_csv:
                config["source_csv"] = source_csv
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def get_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_config(
        self,
        prompt_tokens_delta: int = 0,
        total_spend_delta: float = 0.0,
        model: Optional[str] = None,
    ) -> None:
        cfg = self.get_config()
        prompt_tokens = cfg.get("prompt_tokens", 0) + prompt_tokens_delta
        total_spend = cfg.get("total_spend_usd", 0.0) + total_spend_delta
        model_val = model or cfg.get("model", "text-embedding-3-small")
        self._write_config(
            model=model_val,
            prompt_tokens=prompt_tokens,
            total_spend_usd=total_spend,
            source_csv=cfg.get("source_csv"),
            preserve_existing=True,
        )

    def _get_embedded_words(self) -> set[str]:
        """Return set of words already in embeddings.csv."""
        if not self.embeddings_path.exists():
            return set()
        embedded: set[str] = set()
        with open(self.embeddings_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                w = (row.get("word") or "").strip()
                if w:
                    embedded.add(w)
        return embedded

    def _has_tokens_column(self) -> bool:
        """Return True if embeddings.csv has a tokens column."""
        if not self.embeddings_path.exists():
            return False
        with open(self.embeddings_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return "tokens" in (reader.fieldnames or [])

    def get_pending_words(self, limit: Optional[int] = None) -> List[str]:
        """
        Return words not yet in embeddings.csv, up to limit.
        Preserves order from words.txt.
        """
        embedded = self._get_embedded_words()
        pending: List[str] = []
        if not self.words_path.exists():
            return pending
        with open(self.words_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and w not in embedded:
                    pending.append(w)
                    if limit is not None and len(pending) >= limit:
                        break
        return pending

    def count_embedded(self) -> int:
        return len(self._get_embedded_words())

    def migrate_add_tokens_column(
        self, token_counter: Optional[Callable[[str], int]] = None
    ) -> bool:
        """
        If embeddings.csv exists with only word,vector, rewrite with tokens column.
        token_counter(word) -> int. If None, uses 0 for existing rows.
        Returns True if migration was performed.
        """
        if not self.embeddings_path.exists() or self._has_tokens_column():
            return False
        token_fn = token_counter or (lambda _: 0)
        rows: List[Dict[str, Any]] = []
        with open(self.embeddings_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row.get("word", "")
                rows.append(
                    {
                        "word": word,
                        "vector": row.get("vector", ""),
                        "tokens": token_fn(word),
                    }
                )
        with open(self.embeddings_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["word", "vector", "tokens"])
            writer.writeheader()
            writer.writerows(rows)
        return True

    def append_embeddings(
        self, items: List[Tuple[str, List[float], int]]
    ) -> None:
        """Append (word, vector, tokens) rows to embeddings.csv. Vector serialized as JSON."""
        if not items:
            return
        ensure_parent_dir(self.embeddings_path)
        with open(self.embeddings_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for item in items:
                if len(item) == 3:
                    word, vector, tokens = item
                else:
                    word, vector = item[:2]
                    tokens = 0
                vector_json = json.dumps(vector)
                writer.writerow([word, vector_json, tokens])

    @staticmethod
    def store_path_for_csv(
        csv_path: str | Path, base_dir: str | Path = "data/embeddings"
    ) -> Path:
        """Derive store folder path from edgelist CSV path."""
        name = _dataset_name_from_csv_path(csv_path)
        return Path(base_dir) / name
