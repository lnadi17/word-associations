from __future__ import annotations

import argparse

from lwow.config import load_config
from lwow.io import read_cue_stats, write_csv
from lwow.sampling import sample_cues
from lwow.clients.anthropic_client import AnthropicTextClient
from lwow.generation import GenerationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate word associations with Claude.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    cues = read_cue_stats(config.paths.cue_stats_path)
    sampled = sample_cues(cues, config.sampling.num_cues, config.sampling.seed)

    client = AnthropicTextClient(
        model=config.generation.model,
        max_tokens=config.generation.max_tokens,
        temperature=config.generation.temperature,
        request_timeout_sec=config.generation.request_timeout_sec,
    )
    runner = GenerationRunner(
        client=client,
        system_prompt=config.generation.prompt_template,
        repetitions_per_cue=config.generation.repetitions_per_cue,
        max_retries=config.generation.max_retries,
        retry_backoff_sec=config.generation.retry_backoff_sec,
    )
    rows = runner.generate_for_cues(sampled)
    write_csv(
        config.paths.raw_output_path,
        ["cue", "trial", "raw_text", "parsed", "ok"],
        rows,
    )


if __name__ == "__main__":
    main()
