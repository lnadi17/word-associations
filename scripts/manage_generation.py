from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from lwow.campaign import init_or_load_campaign, select_cue_window
from lwow.clients.anthropic_client import AnthropicTextClient
from lwow.clients.openai_client import OpenAITextClient
from lwow.config import RunConfig, load_config
from lwow.generation import GenerationRunner
from lwow.io import read_cue_stats
from lwow.sampling import sample_cues


def _now_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def _build_client(config: RunConfig) -> AnthropicTextClient | OpenAITextClient:
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
        return OpenAITextClient(
            **common,
            api_base_url=base_url,
            reasoning_effort=g.reasoning_effort or None,
        )
    return AnthropicTextClient(**common, api_base_url=g.api_base_url)


def _build_runner(config: RunConfig, client: AnthropicTextClient | OpenAITextClient) -> GenerationRunner:
    g = config.generation
    return GenerationRunner(
        client=client,
        repetitions_per_cue=g.repetitions_per_cue,
        max_retries=g.max_retries,
        retry_backoff_sec=g.retry_backoff_sec,
        batch_request_limit=g.batch_request_limit,
        batch_poll_interval_sec=g.batch_poll_interval_sec,
        batch_timeout_sec=g.batch_timeout_sec,
        checkpoint_every_n_results=g.checkpoint_every_n_results,
        pricing={
            "input_per_mtok": g.input_cost_per_mtok,
            "output_per_mtok": g.output_cost_per_mtok,
        },
        campaign_name=g.campaign_name,
        max_open_batches=g.max_open_batches,
        submission_mode=g.submission_mode,
    )


def _apply_run_overrides(config: RunConfig, args: argparse.Namespace) -> None:
    """Apply optional CLI overrides. Used by interactive script for prompted values."""
    g = config.generation
    if getattr(args, "campaign_name", None):
        g.campaign_name = args.campaign_name
    if getattr(args, "cue_start_index", None) is not None:
        g.cue_start_index = args.cue_start_index
    if getattr(args, "cue_count", None) is not None:
        g.cue_count = args.cue_count
    if getattr(args, "max_open_batches", None) is not None:
        g.max_open_batches = args.max_open_batches


def _resolve_run_dir(config: RunConfig, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    return Path(config.generation.run_root_dir) / _now_run_id()


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


def _print_status(progress: Dict[str, Any]) -> None:
    ob = progress.get("open_batches", 0)
    max_ob = progress.get("max_open_batches", 0)
    ob_str = f"{ob}/{max_ob}" if max_ob > 0 else str(ob)
    print(
        f"state={progress.get('state')} "
        f"completed={progress.get('completed_requests', 0)}/{progress.get('total_requests', 0)} "
        f"failed={progress.get('failed_requests', 0)} "
        f"remaining={progress.get('remaining_requests', 0)} "
        f"open_batches={ob_str} "
        f"skipped_existing={progress.get('skipped_existing_requests', 0)} "
        f"cost_usd={progress.get('estimated_cost_usd', 0.0)}"
    )


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _apply_run_overrides(config, args)
    run_dir = _resolve_run_dir(config, args.run_dir)
    is_resume = (run_dir / "manifest.json").exists()
    if is_resume:
        run_dir = run_dir  # Must exist
        if getattr(args, "reconcile_only", False):
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["reconcile_only"] = True
            manifest_path.write_text(json.dumps(manifest, indent=2))
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

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
    with open(campaign_paths["master_cues_path"], "r", encoding="utf-8") as f:
        master_cues = json.load(f)
    with open(campaign_paths["index_path"], "r", encoding="utf-8") as f:
        idx = json.load(f)
    completed_by_cue = idx.get("completed_count_by_cue", {})
    reps = config.generation.repetitions_per_cue
    next_start = 0
    for cue in master_cues:
        if int(completed_by_cue.get(cue, 0)) >= reps:
            next_start += 1
        else:
            break
    completed_any = sum(1 for c in master_cues if int(completed_by_cue.get(c, 0)) > 0)
    completed_full = sum(1 for c in master_cues if int(completed_by_cue.get(c, 0)) >= reps)
    print(
        json.dumps(
            {
                "campaign_name": campaign_name,
                "campaign_root": campaign_paths["root_dir"],
                "master_cues_total": len(master_cues),
                "completed_request_ids": int(idx.get("total_completed", 0)),
                "completed_cues_any": completed_any,
                "completed_cues_full": completed_full,
                "next_start_index": next_start,
                "default_window_count": config.sampling.num_cues,
                "max_open_batches_default": config.generation.max_open_batches,
            },
            indent=2,
        )
    )


def cmd_pause(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    config = load_config(args.config)
    client = _build_client(config)
    GenerationRunner.pause_run(run_dir=run_dir, cancel_remote=True, client=client)
    print(json.dumps({"run_dir": str(run_dir), "state": "paused"}, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    progress = GenerationRunner.load_progress(run_dir)
    _print_status(progress)


def infer_campaign_from_config(config_path: str) -> str:
    """Infer campaign name from config: paths.dataset_name, or generation.campaign_name, or filename stem."""
    config = load_config(config_path)
    if config.paths.dataset_name:
        return config.paths.dataset_name
    if config.generation.campaign_name:
        return config.generation.campaign_name
    stem = Path(config_path).stem
    if stem.startswith("lwow_"):
        return stem[5:]  # lwow_gpt5_mini -> gpt5_mini
    return stem or "default"


def cmd_infer_campaign(args: argparse.Namespace) -> None:
    print(infer_campaign_from_config(args.config))


def cmd_list_runs(args: argparse.Namespace) -> None:
    """List run dirs, optionally filtered by config. One path per line."""
    runs_root = Path(args.runs_root)
    config_filter = args.config
    cfg_resolved = str(Path(config_filter).resolve()) if config_filter else None
    for run_dir in sorted(runs_root.iterdir(), key=lambda p: p.name, reverse=True):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            manifest_config = manifest.get("config_path", "")
            if cfg_resolved and manifest_config:
                man_resolved = str(Path(manifest_config).resolve())
                if cfg_resolved != man_resolved:
                    continue
        except (json.JSONDecodeError, OSError):
            continue
        print(run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage LWOW generation runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Start or resume a generation run.")
    p_run.add_argument("--config", required=True, help="Path to YAML config.")
    p_run.add_argument("--run-dir", default=None, help="Run directory. If present and has manifest, resume. Else create new.")
    p_run.add_argument("--campaign-name", default=None, help="Campaign name (overrides config).")
    p_run.add_argument("--cue-start-index", type=int, default=None, help="Cue window start (overrides config).")
    p_run.add_argument("--cue-count", type=int, default=None, help="Cue window size (overrides config).")
    p_run.add_argument("--max-open-batches", type=int, default=None, help="Max open batches (overrides config).")
    p_run.add_argument("--reconcile-only", action="store_true", help="Only collect open batches, do not submit new ones.")
    p_run.set_defaults(func=cmd_run)

    p_pause = sub.add_parser("pause", help="Pause an existing run (cancels open remote batches).")
    p_pause.add_argument("--run-dir", required=True, help="Run directory.")
    p_pause.add_argument("--config", required=True, help="Config path (needed to cancel remote batches).")
    p_pause.set_defaults(func=cmd_pause)

    p_status = sub.add_parser("status", help="Inspect run status.")
    p_status.add_argument("--run-dir", required=True, help="Run directory.")
    p_status.set_defaults(func=cmd_status)

    p_campaign = sub.add_parser("campaign-status", help="Inspect campaign-level dedup state.")
    p_campaign.add_argument("--config", required=True, help="Path to YAML config.")
    p_campaign.add_argument("--campaign-name", default=None, help="Campaign name override.")
    p_campaign.set_defaults(func=cmd_campaign_status)

    p_infer = sub.add_parser("infer-campaign", help="Print campaign name inferred from config.")
    p_infer.add_argument("--config", required=True, help="Path to YAML config.")
    p_infer.set_defaults(func=cmd_infer_campaign)

    p_list = sub.add_parser("list-runs", help="List run dirs, optionally filtered by config.")
    p_list.add_argument("--runs-root", default="data/runs", help="Runs directory.")
    p_list.add_argument("--config", default=None, help="Only runs that used this config.")
    p_list.set_defaults(func=cmd_list_runs)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
