from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from lwow.clients.embeddings import EmbeddingClient


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(vec_a, vec_b) / denom)


@dataclass
class EmbeddingAnalyzer:
    client: EmbeddingClient
    missing_words: List[str] = field(default_factory=list)
    missing_edges: int = 0

    def embed_vocab(self, vocab: List[str]) -> Dict[str, np.ndarray]:
        embeddings: Dict[str, np.ndarray] = {}
        if self.client.provider.lower() in ("word2vec", "gensim"):
            for word in vocab:
                vector = self.client.get_vector(word)
                if vector is None:
                    self.missing_words.append(word)
                    continue
                embeddings[word] = np.array(vector, dtype=np.float32)
        else:
            batch_size = self.client.batch_size
            for start in range(0, len(vocab), batch_size):
                batch = vocab[start : start + batch_size]
                vectors = self.client.embed_texts(batch)
                for word, vector in zip(batch, vectors):
                    embeddings[word] = np.array(vector, dtype=np.float32)
        return embeddings

    def compute_distances(
        self, edges: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, int, float]]:
        vocab = sorted({src for src, _, _ in edges} | {tgt for _, tgt, _ in edges})
        embeddings = self.embed_vocab(vocab)
        results: List[Tuple[str, str, int, float]] = []
        for src, tgt, count in edges:
            if src not in embeddings or tgt not in embeddings:
                self.missing_edges += 1
                continue
            dist = cosine_distance(embeddings[src], embeddings[tgt])
            results.append((src, tgt, count, dist))
        return results

    @staticmethod
    def correlate(distances: List[Tuple[str, str, int, float]]) -> Dict[str, float]:
        if len(distances) < 2:
            return {
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_r": float("nan"),
                "spearman_p": float("nan"),
            }
        counts = np.array([item[2] for item in distances], dtype=np.float32)
        dists = np.array([item[3] for item in distances], dtype=np.float32)
        pearson = pearsonr(counts, dists)
        spearman = spearmanr(counts, dists)
        return {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }

    @staticmethod
    def find_outliers(
        distances: List[Tuple[str, str, int, float]],
        count_percentile: float,
        distance_percentile: float,
    ) -> List[Tuple[str, str, int, float]]:
        if not distances:
            return []
        counts = np.array([item[2] for item in distances], dtype=np.float32)
        dists = np.array([item[3] for item in distances], dtype=np.float32)
        count_cutoff = float(np.quantile(counts, count_percentile))
        dist_cutoff = float(np.quantile(dists, distance_percentile))
        return [
            item
            for item in distances
            if item[2] >= count_cutoff and item[3] >= dist_cutoff
        ]
