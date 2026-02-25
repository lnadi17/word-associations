from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from lwow.campaign import init_or_load_campaign, load_completed_ids, select_cue_window
from lwow.clients.anthropic_client import AnthropicTextClient
from lwow.config import RunConfig, load_config
from lwow.generation import GenerationRunner
from lwow.io import read_cue_stats
from lwow.sampling import sample_cues


TERMINAL_STATES = {"completed", "paused", "timed_out", "failed"}


def _now_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def _build_client(config: RunConfig) -> AnthropicTextClient:
    g = config.generation
    return AnthropicTextClient(
        model=g.model,
        max_tokens=g.max_tokens,
        temperature=g.temperature,
        request_timeout_sec=g.request_timeout_sec,
        max_rate_limit_retries=g.max_rate_limit_retries,
        max_backoff_sec=g.max_backoff_sec,
        jitter_sec=g.jitter_sec,
        api_base_url=g.api_base_url,
    )


def _build_runner(config: RunConfig, client: AnthropicTextClient) -> GenerationRunner:
    g = config.generation
    return GenerationRunner(
        client=client,
        system_prompt=g.prompt_template,
        repetitions_per_cue=g.repetitions_per_cue,
        max_retries=g.max_retries,
        retry_backoff_sec=g.retry_backoff_sec,
        batch_request_limit=g.batch_request_limit,
        batch_poll_interval_sec=g.batch_poll_interval_sec,
        batch_timeout_sec=g.batch_timeout_sec,
        checkpoint_every_n_results=g.checkpoint_every_n_results,
        enable_prompt_caching=g.enable_prompt_caching,
        cache_control_type=g.cache_control_type,
        pricing={
            "input_per_mtok": g.input_cost_per_mtok,
            "output_per_mtok": g.output_cost_per_mtok,
            "cache_creation_input_per_mtok": g.cache_creation_input_cost_per_mtok,
            "cache_read_input_per_mtok": g.cache_read_input_cost_per_mtok,
        },
        campaign_name=g.campaign_name,
        max_open_batches=g.max_open_batches,
        submission_mode=g.submission_mode,
    )


def _resolve_run_dir(config: RunConfig, run_id: str | None, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    resolved_run_id = run_id or _now_run_id()
    return Path(config.generation.run_root_dir) / resolved_run_id


def _apply_generation_overrides(config: RunConfig, args: argparse.Namespace) -> None:
    g = config.generation
    if getattr(args, "campaign_name", None):
        g.campaign_name = args.campaign_name
    if getattr(args, "cue_start_index", None) is not None:
        g.cue_start_index = args.cue_start_index
    if getattr(args, "cue_count", None) is not None:
        g.cue_count = args.cue_count
    if getattr(args, "master_seed", None) is not None:
        g.master_seed = args.master_seed
    if getattr(args, "max_open_batches", None) is not None:
        g.max_open_batches = args.max_open_batches


def _resolve_sampled_cues(config: RunConfig) -> tuple[list[str], Dict[str, str] | None]:
    cues = read_cue_stats(config.paths.cue_stats_path)
    g = config.generation
    if g.campaign_name:
        campaign_paths = init_or_load_campaign(
            campaign_root_dir=g.campaign_root_dir,
            campaign_name=g.campaign_name,
            all_cues=cues,
            master_seed=g.master_seed,
        )
        cue_count = g.cue_count if g.cue_count > 0 else config.sampling.num_cues
        sampled = select_cue_window(
            master_cues_path=campaign_paths["master_cues_path"],
            cue_start_index=g.cue_start_index,
            cue_count=cue_count,
        )
        return sampled, campaign_paths
    sampled = sample_cues(cues, config.sampling.num_cues, config.sampling.seed)
    return sampled, None


def _print_status(progress: Dict[str, Any], live: bool = False) -> None:
    message = (
        f"state={progress.get('state')} "
        f"completed={progress.get('completed_requests', 0)}/{progress.get('total_requests', 0)} "
        f"failed={progress.get('failed_requests', 0)} "
        f"remaining={progress.get('remaining_requests', 0)} "
        f"open_batches={progress.get('open_batches', 0)} "
        f"skipped_existing={progress.get('skipped_existing_requests', 0)} "
        f"cost_usd={progress.get('estimated_cost_usd', 0.0)}"
    )
    if live:
        print("\r" + message.ljust(120), end="", flush=True)
    else:
        print(message)


def cmd_start(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _apply_generation_overrides(config, args)
    run_dir = _resolve_run_dir(config, args.run_id, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sampled, campaign_paths = _resolve_sampled_cues(config)
    client = _build_client(config)
    runner = _build_runner(config, client)
    runner.campaign_paths = campaign_paths

    run_id = run_dir.name
    progress = runner.run_to_completion(
        run_dir=run_dir,
        run_id=run_id,
        config_path=args.config,
        raw_output_path=config.paths.raw_output_path,
        cues=sampled,
    )
    print(json.dumps({"run_dir": str(run_dir), "progress": progress}, indent=2))


def cmd_resume(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _apply_generation_overrides(config, args)
    run_dir = _resolve_run_dir(config, args.run_id, args.run_dir)
    if not (run_dir / "manifest.json").exists():
        raise ValueError(f"Cannot resume. No manifest found in {run_dir}")

    sampled, campaign_paths = _resolve_sampled_cues(config)
    client = _build_client(config)
    runner = _build_runner(config, client)
    runner.campaign_paths = campaign_paths
    progress = runner.run_to_completion(
        run_dir=run_dir,
        run_id=run_dir.name,
        config_path=args.config,
        raw_output_path=config.paths.raw_output_path,
        cues=sampled,
    )
    print(json.dumps({"run_dir": str(run_dir), "progress": progress}, indent=2))


def cmd_campaign_status(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    campaign_name = args.campaign_name or config.generation.campaign_name
    if not campaign_name:
        raise ValueError("campaign name is required (set in config or pass --campaign-name).")
    all_cues = read_cue_stats(config.paths.cue_stats_path)
    campaign_paths = init_or_load_campaign(
        campaign_root_dir=config.generation.campaign_root_dir,
        campaign_name=campaign_name,
        all_cues=all_cues,
        master_seed=config.generation.master_seed,
    )
    completed_ids = load_completed_ids(campaign_paths["index_path"])
    master_cues = select_cue_window(
        master_cues_path=campaign_paths["master_cues_path"],
        cue_start_index=0,
        cue_count=10_000_000,
    )
    print(
        json.dumps(
            {
                "campaign_name": campaign_name,
                "campaign_root": campaign_paths["root_dir"],
                "master_cues_total": len(master_cues),
                "completed_request_ids": len(completed_ids),
            },
            indent=2,
        )
    )


def cmd_pause(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    client = None
    if args.cancel_remote:
        if not args.config:
            raise ValueError("--config is required when using --cancel-remote")
        config = load_config(args.config)
        client = _build_client(config)
    GenerationRunner.pause_run(run_dir=run_dir, cancel_remote=args.cancel_remote, client=client)
    print(json.dumps({"run_dir": str(run_dir), "state": "paused"}, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    if args.session_cost_limit_usd is not None and not args.follow:
        raise ValueError("--session-cost-limit-usd can only be used with --follow")

    while True:
        progress = GenerationRunner.load_progress(run_dir)
        _print_status(progress, live=args.follow and not args.as_json)
        if args.as_json and not args.follow:
            print(json.dumps(progress, indent=2))
        if args.follow:
            if args.as_json:
                print(json.dumps(progress))
            # Session-scoped guard: never persisted.
            if (
                args.session_cost_limit_usd is not None
                and progress.get("state") == "running"
                and float(progress.get("estimated_cost_usd", 0.0)) >= args.session_cost_limit_usd
            ):
                GenerationRunner.pause_run(run_dir=run_dir, cancel_remote=False, client=None)
                if not args.as_json:
                    print("\npaused: session_cost_limit_reached")
                break
            if progress.get("state") in TERMINAL_STATES:
                if not args.as_json:
                    print()
                break
            time.sleep(args.refresh_sec)
            continue
        break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage LWOW Claude generation runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start a new generation run.")
    p_start.add_argument("--config", required=True, help="Path to YAML config.")
    p_start.add_argument("--run-id", default=None, help="Optional run ID.")
    p_start.add_argument("--run-dir", default=None, help="Optional run directory override.")
    p_start.add_argument("--campaign-name", default=None, help="Campaign name for cross-run dedup.")
    p_start.add_argument("--cue-start-index", type=int, default=None, help="Cue window start index.")
    p_start.add_argument("--cue-count", type=int, default=None, help="Cue window size override.")
    p_start.add_argument("--master-seed", type=int, default=None, help="Master cue ordering seed.")
    p_start.add_argument("--max-open-batches", type=int, default=None, help="Limit concurrent open batches.")
    p_start.set_defaults(func=cmd_start)

    p_resume = sub.add_parser("resume", help="Resume an existing run.")
    p_resume.add_argument("--config", required=True, help="Path to YAML config.")
    p_resume.add_argument("--run-id", default=None, help="Run ID under configured run root.")
    p_resume.add_argument("--run-dir", default=None, help="Run directory override.")
    p_resume.add_argument("--campaign-name", default=None, help="Campaign name for cross-run dedup.")
    p_resume.add_argument("--cue-start-index", type=int, default=None, help="Cue window start index.")
    p_resume.add_argument("--cue-count", type=int, default=None, help="Cue window size override.")
    p_resume.add_argument("--master-seed", type=int, default=None, help="Master cue ordering seed.")
    p_resume.add_argument("--max-open-batches", type=int, default=None, help="Limit concurrent open batches.")
    p_resume.set_defaults(func=cmd_resume)

    p_pause = sub.add_parser("pause", help="Pause an existing run.")
    p_pause.add_argument("--run-dir", required=True, help="Run directory.")
    p_pause.add_argument("--cancel-remote", action="store_true", help="Also cancel open remote batches.")
    p_pause.add_argument("--config", default=None, help="Config path (required for --cancel-remote).")
    p_pause.set_defaults(func=cmd_pause)

    p_status = sub.add_parser("status", help="Inspect run status.")
    p_status.add_argument("--run-dir", required=True, help="Run directory.")
    p_status.add_argument("--follow", action="store_true", help="Live status mode.")
    p_status.add_argument("--refresh-sec", type=float, default=2.0, help="Refresh interval for --follow.")
    p_status.add_argument(
        "--session-cost-limit-usd",
        type=float,
        default=None,
        help="Session-only cost cap in follow mode. Pauses when exceeded.",
    )
    p_status.add_argument("--as-json", action="store_true", help="Print JSON status.")
    p_status.set_defaults(func=cmd_status)

    p_campaign = sub.add_parser("campaign-status", help="Inspect campaign-level dedup state.")
    p_campaign.add_argument("--config", required=True, help="Path to YAML config.")
    p_campaign.add_argument("--campaign-name", default=None, help="Campaign name override.")
    p_campaign.set_defaults(func=cmd_campaign_status)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
