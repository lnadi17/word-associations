from __future__ import annotations

import argparse
from datetime import datetime, timezone

from lwow.config import load_config
from lwow.io import read_cue_stats
from lwow.sampling import sample_cues
from lwow.clients.anthropic_client import AnthropicTextClient
from lwow.clients.openai_client import OpenAITextClient
from lwow.generation import GenerationRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate word associations with Claude (batch-first compatibility entrypoint)."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory. Defaults to <generation.run_root_dir>/<timestamp>.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cues = read_cue_stats(config.paths.cue_stats_path)
    sampled = sample_cues(cues, config.sampling.num_cues, config.sampling.seed)

    g = config.generation
    provider = g.api_provider.lower()
    common = dict(
        model=g.model,
        max_tokens=g.max_tokens,
        temperature=g.temperature,
        request_timeout_sec=g.request_timeout_sec,
        max_rate_limit_retries=g.max_rate_limit_retries,
        max_backoff_sec=g.max_backoff_sec,
        jitter_sec=g.jitter_sec,
    )
    if provider == "openai":
        base_url = g.api_base_url if g.api_base_url != "https://api.anthropic.com" else "https://api.openai.com"
        client: AnthropicTextClient | OpenAITextClient = OpenAITextClient(
            **common,
            api_base_url=base_url,
            reasoning_effort=g.reasoning_effort or None,
        )
    else:
        client = AnthropicTextClient(**common, api_base_url=g.api_base_url)
    runner = GenerationRunner(
        client=client,
        repetitions_per_cue=config.generation.repetitions_per_cue,
        max_retries=config.generation.max_retries,
        retry_backoff_sec=config.generation.retry_backoff_sec,
        batch_request_limit=config.generation.batch_request_limit,
        batch_poll_interval_sec=config.generation.batch_poll_interval_sec,
        batch_timeout_sec=config.generation.batch_timeout_sec,
        checkpoint_every_n_results=config.generation.checkpoint_every_n_results,
        pricing={
            "input_per_mtok": config.generation.input_cost_per_mtok,
            "output_per_mtok": config.generation.output_cost_per_mtok,
        },
    )

    run_dir = args.run_dir
    if run_dir is None:
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = f"{config.generation.run_root_dir}/{run_id}"
    progress = runner.run_to_completion(
        run_dir=run_dir,
        run_id=run_dir.rstrip("/").split("/")[-1],
        config_path=args.config,
        raw_output_path=config.paths.raw_output_path,
        cues=sampled,
    )
    print(f"run_dir={run_dir}")
    print(f"state={progress.get('state')}")
    print(f"completed={progress.get('completed_requests')}/{progress.get('total_requests')}")
    print(f"estimated_cost_usd={progress.get('estimated_cost_usd')}")


if __name__ == "__main__":
    main()
