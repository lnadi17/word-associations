from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from gensim.models import KeyedVectors
import gensim.downloader as api


@dataclass
class EmbeddingClient:
    provider: str
    model: str
    input_type: str
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    oov_strategy: str = "skip"
    batch_size: int = 64
    _kv: Optional[KeyedVectors] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        provider = self.provider.lower()
        if provider == "word2vec":
            if not self.model_path:
                raise ValueError("embeddings.model_path is required for word2vec.")
            binary = self.model_path.endswith(".bin") or self.model_path.endswith(".bin.gz")
            self._kv = KeyedVectors.load_word2vec_format(self.model_path, binary=binary)
        elif provider == "gensim":
            if not self.model_name:
                raise ValueError("embeddings.model_name is required for gensim.")
            self._kv = api.load(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        provider = self.provider.lower()
        if provider == "voyage":
            return self._embed_voyage(texts)
        if provider == "word2vec":
            return self._embed_word2vec(texts)
        if provider == "gensim":
            return self._embed_word2vec(texts)
        if provider == "anthropic":
            raise NotImplementedError(
                "Anthropic does not provide embeddings. "
                "Set embeddings.provider to 'voyage' and provide VOYAGE_API_KEY."
            )
        raise ValueError(f"Unknown embeddings provider: {self.provider}")

    def _embed_voyage(self, texts: List[str]) -> List[List[float]]:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required for Voyage embeddings.")

        url = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model,
            "input_type": self.input_type,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data.get("data", [])]

    def _embed_word2vec(self, texts: List[str]) -> List[List[float]]:
        if not self._kv:
            raise ValueError("Word2Vec model not loaded.")
        vectors: List[List[float]] = []
        for text in texts:
            vector = self.get_vector(text)
            if vector is None:
                if self.oov_strategy == "skip":
                    raise ValueError(f"OOV token in word2vec: {text}")
                vector = [0.0] * self._kv.vector_size
            vectors.append(vector)
        return vectors

    def get_vector(self, text: str) -> Optional[List[float]]:
        if self.provider.lower() not in ("word2vec", "gensim"):
            return None
        if not self._kv:
            raise ValueError("Word2Vec model not loaded.")
        candidates = [text, text.replace(" ", "_")]
        for candidate in candidates:
            if candidate in self._kv:
                return self._kv[candidate].tolist()
        if self.oov_strategy == "zero":
            return [0.0] * self._kv.vector_size
        return None
