#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_ROOT_DEFAULT="data/runs"

CLR_RESET="$(tput sgr0 2>/dev/null || true)"
CLR_BOLD="$(tput bold 2>/dev/null || true)"
CLR_BLUE="$(tput setaf 4 2>/dev/null || true)"
CLR_GREEN="$(tput setaf 2 2>/dev/null || true)"
CLR_YELLOW="$(tput setaf 3 2>/dev/null || true)"
CLR_RED="$(tput setaf 1 2>/dev/null || true)"
CLR_CYAN="$(tput setaf 6 2>/dev/null || true)"

CONFIG_PATH=""
CAMPAIGN_NAME=""

print_header() {
  clear
  printf "%s%sLWOW Interactive Campaign Console%s\n" "$CLR_BOLD" "$CLR_CYAN" "$CLR_RESET"
  printf "%s%s%s\n" "$CLR_BLUE" "----------------------------------------" "$CLR_RESET"
}

ensure_api_key() {
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
  fi

  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo
    echo "ANTHROPIC_API_KEY is not set."
    read -r -s -p "Enter ANTHROPIC_API_KEY (input hidden): " ANTHROPIC_API_KEY
    echo
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      echo "No API key provided. Exiting."
      exit 1
    fi
    export ANTHROPIC_API_KEY
  fi
}

pick_config() {
  local configs=()
  local cfg
  for cfg in configs/*.yaml; do
    [[ -e "$cfg" ]] || continue
    configs+=("$cfg")
  done
  if [[ "${#configs[@]}" -eq 0 ]]; then
    echo "No config files found in configs/*.yaml" >&2
    exit 1
  fi

  local sorted
  sorted="$(printf '%s\n' "${configs[@]}" | sort)"
  configs=()
  while IFS= read -r cfg; do
    [[ -n "$cfg" ]] && configs+=("$cfg")
  done <<< "$sorted"

  echo
  echo "Select config:"
  local i=1
  for cfg in "${configs[@]}"; do
    printf "  [%d] %s\n" "$i" "$cfg"
    ((i++))
  done
  while true; do
    read -r -p "Enter number: " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#configs[@]} )); then
      CONFIG_PATH="${configs[choice-1]}"
      return
    fi
    echo "Invalid choice."
  done
}

ensure_campaign_name() {
  local cfg_campaign
  cfg_campaign="$(CONFIG_PATH="$CONFIG_PATH" "$PYTHON_BIN" - <<'PY'
import os
from lwow.config import load_config
cfg = load_config(os.environ["CONFIG_PATH"])
print(cfg.generation.campaign_name or "")
PY
)"
  echo
  read -r -p "Campaign name [${cfg_campaign:-my_campaign}]: " entered
  if [[ -n "$entered" ]]; then
    CAMPAIGN_NAME="$entered"
  elif [[ -n "$cfg_campaign" ]]; then
    CAMPAIGN_NAME="$cfg_campaign"
  else
    CAMPAIGN_NAME="my_campaign"
  fi
}

campaign_snapshot() {
  CONFIG_PATH="$CONFIG_PATH" CAMPAIGN_NAME="$CAMPAIGN_NAME" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from lwow.campaign import init_or_load_campaign
from lwow.config import load_config
from lwow.io import read_cue_stats

cfg = load_config(os.environ["CONFIG_PATH"])
campaign_name = os.environ["CAMPAIGN_NAME"]
all_cues = read_cue_stats(cfg.paths.cue_stats_path)
paths = init_or_load_campaign(
    campaign_root_dir=cfg.generation.campaign_root_dir,
    campaign_name=campaign_name,
    all_cues=all_cues,
    master_seed=cfg.generation.master_seed,
)

with open(paths["master_cues_path"], "r", encoding="utf-8") as f:
    master = json.load(f)
with open(paths["index_path"], "r", encoding="utf-8") as f:
    idx = json.load(f)

completed_by_cue = idx.get("completed_count_by_cue", {})
reps = int(cfg.generation.repetitions_per_cue)

# Contiguous "next start" pointer from beginning of master list.
next_start = 0
for cue in master:
    if int(completed_by_cue.get(cue, 0)) >= reps:
        next_start += 1
    else:
        break

completed_any = sum(1 for cue in master if int(completed_by_cue.get(cue, 0)) > 0)
completed_full = sum(1 for cue in master if int(completed_by_cue.get(cue, 0)) >= reps)

print(json.dumps({
    "campaign_name": campaign_name,
    "campaign_root": paths["root_dir"],
    "master_cues_total": len(master),
    "completed_request_ids": int(idx.get("total_completed", 0)),
    "completed_cues_any": completed_any,
    "completed_cues_full": completed_full,
    "next_start_index": next_start,
    "default_window_count": int(cfg.sampling.num_cues),
    "max_open_batches_default": int(cfg.generation.max_open_batches),
}, indent=2))
PY
}

show_campaign_status() {
  print_header
  echo "Config: $CONFIG_PATH"
  echo "Campaign: $CAMPAIGN_NAME"
  echo "----------------------------------------"
  campaign_snapshot
  echo "----------------------------------------"
  read -r -p "Press Enter to continue..."
}

start_window_run() {
  local start_index="$1"
  local cue_count="$2"
  local max_open_batches="$3"
  local run_id
  run_id="$(date -u +"%Y%m%d-%H%M%S")"
  local run_dir="$RUNS_ROOT_DEFAULT/$run_id"
  mkdir -p "$run_dir"
  local log_path="$run_dir/interactive_runner.log"

  PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py start \
    --config "$CONFIG_PATH" \
    --run-dir "$run_dir" \
    --campaign-name "$CAMPAIGN_NAME" \
    --cue-start-index "$start_index" \
    --cue-count "$cue_count" \
    --max-open-batches "$max_open_batches" \
    >"$log_path" 2>&1 &
  local pid="$!"

  print_header
  printf "%sStarted run in background.%s\n" "$CLR_GREEN" "$CLR_RESET"
  echo "Run Dir: $run_dir"
  echo "PID: $pid"
  echo "Log: $log_path"
  echo
  read -r -p "Attach interactive run dashboard now? [Y/n]: " yn
  if [[ ! "$yn" =~ ^[Nn]$ ]]; then
    ./scripts/manage_generation_interactive.sh --attach-run-dir "$run_dir"
  fi
}

start_next_window() {
  local snapshot
  snapshot="$(campaign_snapshot)"
  local next_start default_count default_batches
  next_start="$(echo "$snapshot" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["next_start_index"])')"
  default_count="$(echo "$snapshot" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["default_window_count"])')"
  default_batches="$(echo "$snapshot" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["max_open_batches_default"])')"

  print_header
  echo "Starting next window"
  echo "Campaign: $CAMPAIGN_NAME"
  echo "Suggested start index: $next_start"
  read -r -p "Cue count [${default_count}]: " cue_count
  cue_count="${cue_count:-$default_count}"
  read -r -p "Max open batches [${default_batches}]: " max_batches
  max_batches="${max_batches:-$default_batches}"
  start_window_run "$next_start" "$cue_count" "$max_batches"
}

start_custom_window() {
  print_header
  echo "Starting custom window"
  echo "Campaign: $CAMPAIGN_NAME"
  read -r -p "Start index: " start_index
  read -r -p "Cue count: " cue_count
  read -r -p "Max open batches [1]: " max_batches
  max_batches="${max_batches:-1}"

  if [[ ! "$start_index" =~ ^[0-9]+$ ]] || [[ ! "$cue_count" =~ ^[0-9]+$ ]]; then
    echo "Start index and cue count must be non-negative integers."
    read -r -p "Press Enter to continue..."
    return
  fi
  start_window_run "$start_index" "$cue_count" "$max_batches"
}

main_menu() {
  while true; do
    print_header
    echo "Config: $CONFIG_PATH"
    echo "Campaign: $CAMPAIGN_NAME"
    echo
    echo "1) Show campaign status"
    echo "2) Start next window (auto index)"
    echo "3) Start custom window"
    echo "4) Attach existing run dashboard"
    echo "5) Change campaign name"
    echo "6) Exit"
    echo
    read -r -p "Select option: " option
    case "$option" in
      1) show_campaign_status ;;
      2) start_next_window ;;
      3) start_custom_window ;;
      4) ./scripts/manage_generation_interactive.sh ;;
      5) ensure_campaign_name ;;
      6) exit 0 ;;
      *) ;;
    esac
  done
}

main() {
  print_header
  ensure_api_key
  pick_config
  ensure_campaign_name
  main_menu
}

main "$@"
