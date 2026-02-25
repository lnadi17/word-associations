from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PathsConfig:
    cue_stats_path: str = "lexicon/SWOW-EN18/cueStats.SWOW-EN.R123.20180827.csv"
    raw_output_path: str = "data/raw/claude_opus45_responses.csv"
    processed_output_path: str = "data/processed/claude_opus45_processed.csv"
    edge_list_output_path: str = "data/processed/claude_opus45_edges.csv"
    embedding_analysis_path: str = "data/analysis/embedding_correlation.csv"
    outliers_path: str = "data/analysis/outliers.csv"


@dataclass
class SamplingConfig:
    num_cues: int = 100
    seed: int = 42


@dataclass
class GenerationConfig:
    model: str = "claude-opus-4-5"
    max_tokens: int = 32
    temperature: float = 0.7
    repetitions_per_cue: int = 10
    max_retries: int = 3  # retained for compatibility
    retry_backoff_sec: float = 1.5  # retained for compatibility
    request_timeout_sec: float = 60.0
    api_base_url: str = "https://api.anthropic.com"
    batch_request_limit: int = 200
    batch_poll_interval_sec: float = 5.0
    batch_timeout_sec: float = 7200.0
    max_rate_limit_retries: int = 8
    max_backoff_sec: float = 30.0
    jitter_sec: float = 0.5
    run_root_dir: str = "data/runs"
    checkpoint_every_n_results: int = 50
    enable_prompt_caching: bool = True
    cache_control_type: str = "ephemeral"
    input_cost_per_mtok: float = 5.0
    output_cost_per_mtok: float = 25.0
    cache_creation_input_cost_per_mtok: float = 6.25
    cache_read_input_cost_per_mtok: float = 0.5
    prompt_template: str = (
        "Task: "
        "• You will be provided with an input word: write the first 3 words you associate to it separated by a comma"
        "• No additional output text is allowed"
        "Constraints:"
        "• No carriage return characters are allowed in the answers"
        "• Answers should be as short as possible"
        "Example: Input: sea Output: water, beach, sun"
    )


@dataclass
class ProcessingConfig:
    repetitions_per_cue: int = 100
    seed: int = 42


@dataclass
class EmbeddingConfig:
    provider: str = "gensim"
    model: str = "voyage-3.5"
    model_path: str = "path/to/word2vec.bin"
    model_name: str = "glove-wiki-gigaword-100"
    oov_strategy: str = "skip"
    batch_size: int = 64
    input_type: str = "document"


@dataclass
class AnalysisConfig:
    count_percentile: float = 0.9
    distance_percentile: float = 0.9


@dataclass
class RunConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RunConfig":
        return RunConfig(
            paths=PathsConfig(**data.get("paths", {})),
            sampling=SamplingConfig(**data.get("sampling", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            processing=ProcessingConfig(**data.get("processing", {})),
            embeddings=EmbeddingConfig(**data.get("embeddings", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
        )


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return RunConfig.from_dict(data)
