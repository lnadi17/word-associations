#!/usr/bin/env bash
# Merged campaign + run interactive. Single entry point for generation workflow.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_interactive_common.sh
source "$SCRIPT_DIR/_interactive_common.sh"

CONFIG_PATH=""
CAMPAIGN_NAME=""
RUN_DIR=""
RUNNER_PID=""
RAW_OUTPUT_PATH=""
CUE_START_INDEX=""
CUE_COUNT=""
ATTACH_RUN_DIR=""

parse_cli_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --attach-run-dir)
        ATTACH_RUN_DIR="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        echo "Usage: $0 [--attach-run-dir data/runs/<run-id>]" >&2
        exit 1
        ;;
    esac
  done
}

infer_campaign_from_config() {
  CAMPAIGN_NAME="$(PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py infer-campaign --config "$CONFIG_PATH" 2>/dev/null)" || CAMPAIGN_NAME="default"
}

pick_existing_run_dir() {
  local runs=()
  local run
  if [[ -n "$CONFIG_PATH" ]] && [[ -f "$CONFIG_PATH" ]]; then
    while IFS= read -r run; do
      [[ -n "$run" ]] && runs+=("$run")
    done < <(PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py list-runs --config "$CONFIG_PATH" --runs-root "$RUNS_ROOT_DEFAULT" 2>/dev/null)
  fi
  if [[ "${#runs[@]}" -eq 0 ]]; then
    for run in "$RUNS_ROOT_DEFAULT"/*; do
      [[ -d "$run" ]] || continue
      [[ -f "$run/manifest.json" ]] || continue
      runs+=("$run")
    done
  fi
  if [[ "${#runs[@]}" -eq 0 ]]; then
    echo "No runs found." >&2
    return 1
  fi

  local sorted
  sorted="$(printf '%s\n' "${runs[@]}" | sort -r)"
  runs=()
  while IFS= read -r run; do
    [[ -n "$run" ]] && runs+=("$run")
  done <<< "$sorted"

  echo
  echo "Select run to attach:"
  local i=1
  for run in "${runs[@]}"; do
    printf "  [%d] %s\n" "$i" "$run"
    ((i++))
  done
  while true; do
    read -r -p "Enter number: " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#runs[@]} )); then
      RUN_DIR="${runs[choice-1]}"
      return 0
    fi
    echo "Invalid choice."
  done
}

campaign_snapshot() {
  PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py campaign-status \
    --config "$CONFIG_PATH" --campaign-name "$CAMPAIGN_NAME" 2>/dev/null
}

set_metadata_from_manifest() {
  local manifest_path="$RUN_DIR/manifest.json"
  if [[ ! -f "$manifest_path" ]]; then
    return 1
  fi
  local values
  values="$(MANIFEST_PATH="$manifest_path" "$PYTHON_BIN" -c "
import json, os
with open(os.environ['MANIFEST_PATH'], 'r', encoding='utf-8') as f:
    m = json.load(f)
print(m.get('config_path', ''))
print(m.get('raw_output_path', ''))
print(m.get('campaign_name', ''))
" 2>/dev/null)" || return 1
  CONFIG_PATH="$(echo "$values" | sed -n '1p')"
  RAW_OUTPUT_PATH="$(echo "$values" | sed -n '2p')"
  CAMPAIGN_NAME="$(echo "$values" | sed -n '3p')"
  return 0
}

set_metadata_from_config() {
  local values
  values="$(CONFIG_PATH="$CONFIG_PATH" "$PYTHON_BIN" -c "
import os
from lwow.config import load_config
cfg = load_config(os.environ['CONFIG_PATH'])
print(cfg.paths.raw_output_path)
print(cfg.generation.cue_start_index)
print(cfg.generation.cue_count)
" 2>/dev/null)"
  RAW_OUTPUT_PATH="$(echo "$values" | sed -n '1p')"
  CUE_START_INDEX="$(echo "$values" | sed -n '2p')"
  CUE_COUNT="$(echo "$values" | sed -n '3p')"
}

show_campaign_status() {
  print_header "Campaign Status"
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
  RUN_DIR="$RUNS_ROOT_DEFAULT/$run_id"
  mkdir -p "$RUN_DIR"
  local log_path="$RUN_DIR/interactive_runner.log"

  PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py run \
    --config "$CONFIG_PATH" \
    --run-dir "$RUN_DIR" \
    --campaign-name "$CAMPAIGN_NAME" \
    --cue-start-index "$start_index" \
    --cue-count "$cue_count" \
    --max-open-batches "$max_open_batches" \
    >"$log_path" 2>&1 &
  RUNNER_PID=$!

  set_metadata_from_config
  CUE_START_INDEX="$start_index"
  CUE_COUNT="$cue_count"
  echo
  printf "%sStarted run in background.%s\n" "$CLR_GREEN" "$CLR_RESET"
  echo "Run Dir: $RUN_DIR"
  echo "PID: $RUNNER_PID"
  enter_run_monitor
}

start_next_window() {
  local snapshot
  snapshot="$(campaign_snapshot)"
  if [[ -z "$snapshot" ]]; then
    echo "Failed to get campaign snapshot." >&2
    read -r -p "Press Enter to continue..."
    return
  fi
  local next_start default_count default_batches
  next_start="$(echo "$snapshot" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("next_start_index", 0))')"
  default_count="$(echo "$snapshot" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("default_window_count", 10))')"
  default_batches="$(echo "$snapshot" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("max_open_batches_default", 1))')"

  print_header "Start Next Window"
  echo "Campaign: $CAMPAIGN_NAME"
  echo "Suggested start index: $next_start"
  read -r -p "Cue count [${default_count}]: " cue_count
  cue_count="${cue_count:-$default_count}"
  read -r -p "Max open batches [${default_batches}]: " max_batches
  max_batches="${max_batches:-$default_batches}"
  start_window_run "$next_start" "$cue_count" "$max_batches"
}

runner_alive() {
  [[ -n "${RUNNER_PID:-}" ]] && kill -0 "$RUNNER_PID" 2>/dev/null
}

discover_runner_pid() {
  [[ -n "${RUNNER_PID:-}" ]] && return 0
  local lock_file="$RUN_DIR/.runner.lock"
  [[ ! -f "$lock_file" ]] && return 0
  local pid
  pid="$(lsof "$lock_file" 2>/dev/null | awk 'NR>1 {print $2; exit}' 2>/dev/null)" || pid=""
  [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && RUNNER_PID="$pid"
  return 0
}

read_status_line() {
  local out
  out="$(PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py status --run-dir "$RUN_DIR" 2>/dev/null || true)"
  STATUS_LINE="$(echo "$out" | sed -n '1p')"
  if [[ -z "$STATUS_LINE" ]]; then
    STATUS_LINE="state=initializing completed=0/0 failed=0 remaining=0 open_batches=0 cost_usd=0.0"
  fi
}

status_value() {
  local key="$1"
  echo "$STATUS_LINE" | sed -n "s/.*${key}=\\([^ ]*\\).*/\\1/p"
}

pause_run() {
  PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py pause --run-dir "$RUN_DIR" --config "$CONFIG_PATH" >/dev/null 2>&1 || true
}

render_dashboard() {
  [ -t 1 ] && clear || true
  print_header "Run Monitor"

  local state completed total failed remaining open_batches skipped_existing cost
  state="$(status_value state)"
  completed="$(echo "$(status_value completed)" | cut -d/ -f1)"
  total="$(echo "$(status_value completed)" | cut -d/ -f2)"
  failed="$(status_value failed)"
  remaining="$(status_value remaining)"
  open_batches="$(status_value open_batches)"
  skipped_existing="$(status_value skipped_existing)"
  cost="$(status_value cost_usd)"

  local state_color="$CLR_YELLOW"
  case "$state" in
    completed) state_color="$CLR_GREEN" ;;
    running) state_color="$CLR_CYAN" ;;
    paused) state_color="$CLR_YELLOW" ;;
    timed_out|failed) state_color="$CLR_RED" ;;
  esac

  printf "%sConfig:%s %s\n" "$CLR_BOLD" "$CLR_RESET" "$CONFIG_PATH"
  printf "%sRun Dir:%s %s\n" "$CLR_BOLD" "$CLR_RESET" "$RUN_DIR"
  printf "%sOutput CSV:%s %s\n" "$CLR_BOLD" "$CLR_RESET" "$RAW_OUTPUT_PATH"
  if [[ -n "${CAMPAIGN_NAME:-}" ]]; then
    printf "%sCampaign:%s %s (window start=%s count=%s)\n" "$CLR_BOLD" "$CLR_RESET" "$CAMPAIGN_NAME" "${CUE_START_INDEX:-0}" "${CUE_COUNT:-0}"
  fi
  printf "%sLog File:%s %s\n" "$CLR_BOLD" "$CLR_RESET" "$RUN_DIR/interactive_runner.log"
  echo
  printf "%sState:%s %s%s%s\n" "$CLR_BOLD" "$CLR_RESET" "$state_color" "${state:-unknown}" "$CLR_RESET"
  printf "%sProgress:%s %s/%s (failed=%s, remaining=%s, open_batches=%s)\n" \
    "$CLR_BOLD" "$CLR_RESET" "${completed:-0}" "${total:-0}" "${failed:-0}" "${remaining:-0}" "${open_batches:-0}"
  printf "%sSkipped Existing:%s %s\n" "$CLR_BOLD" "$CLR_RESET" "${skipped_existing:-0}"
  printf "%sEstimated Cost (USD):%s %s\n" "$CLR_BOLD" "$CLR_RESET" "${cost:-0.0}"

  if runner_alive; then
    printf "%sRunner Process:%s %s%s%s (active)\n" "$CLR_BOLD" "$CLR_RESET" "$CLR_GREEN" "$RUNNER_PID" "$CLR_RESET"
    if [[ "$state" == "paused" ]]; then
      printf "%s  Hint: to resume, kill the runner first: kill %s%s\n" "$CLR_BOLD" "$CLR_RESET" "$RUNNER_PID"
    fi
  else
    printf "%sRunner Process:%s %snot active%s\n" "$CLR_BOLD" "$CLR_RESET" "$CLR_YELLOW" "$CLR_RESET"
    if [[ "$state" == "paused" ]]; then
      printf "%s  Hint: press [r] to start a new runner and resume.%s\n" "$CLR_BOLD" "$CLR_RESET"
    fi
  fi

  echo
  printf "%sControls:%s [p]ause [x]hard-cancel [r]esume [k]ill-runner [l]og-tail [q]uit-monitor\n" "$CLR_BOLD" "$CLR_RESET"
}

show_log_tail() {
  [ -t 1 ] && clear || true
  print_header "Run Log"
  echo "Recent runner log:"
  echo "----------------------------------------"
  if [[ -f "$RUN_DIR/interactive_runner.log" ]]; then
    tail -n 40 "$RUN_DIR/interactive_runner.log"
  else
    echo "(log not created yet)"
  fi
  echo "----------------------------------------"
  read -r -p "Press Enter to return..."
}

start_run_or_show_error() {
  local log_path="$RUN_DIR/interactive_runner.log"
  mkdir -p "$RUN_DIR"

  PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py run \
    --config "$CONFIG_PATH" \
    --run-dir "$RUN_DIR" \
    --campaign-name "$CAMPAIGN_NAME" \
    --cue-start-index "${CUE_START_INDEX:-0}" \
    --cue-count "${CUE_COUNT:-0}" \
    >"$log_path" 2>&1 &
  RUNNER_PID=$!

  sleep 1
  if runner_alive; then
    return
  fi

  [ -t 1 ] && clear || true
  print_header "Run Monitor"
  printf "%sRunner failed to start.%s\n" "$CLR_RED" "$CLR_RESET"
  echo "Recent log:"
  echo "----------------------------------------"
  if [[ -f "$log_path" ]]; then
    tail -n 40 "$log_path"
  else
    echo "(no log available)"
  fi
  echo "----------------------------------------"
  read -r -p "Press Enter to return..."
}

enter_run_monitor() {
  set_metadata_from_manifest || set_metadata_from_config

  while true; do
    discover_runner_pid
    read_status_line
    render_dashboard
    if read -r -s -n 1 -t "$REFRESH_SEC" key; then
      case "$key" in
        p|P) pause_run ;;
        x|X) pause_run ;;
        r|R)
          if runner_alive; then
            [ -t 1 ] && clear || true
            print_header "Run Monitor"
            printf "%sRunner already active (PID %s). Press [k] to kill first.%s\n" "$CLR_YELLOW" "$RUNNER_PID" "$CLR_RESET"
            read -r -p "Press Enter to return..."
          else
            start_run_or_show_error
          fi
          ;;
        k|K)
          if runner_alive; then
            [ -t 1 ] && clear || true
            print_header "Run Monitor"
            printf "%sKill runner (PID %s)? [y/N]: " "$CLR_YELLOW" "$RUNNER_PID" "$CLR_RESET"
            read -r -n 1 yn
            echo
            if [[ "$yn" =~ ^[Yy]$ ]]; then
              kill "$RUNNER_PID" 2>/dev/null || true
              RUNNER_PID=""
              echo "Runner killed. Press [r] to resume."
              read -r -p "Press Enter to return..."
            fi
          fi
          ;;
        l|L) show_log_tail ;;
        q|Q)
          echo
          echo "Monitor exited. Run continues if runner is active."
          return
          ;;
      esac
    fi
  done
}

main_menu() {
  while true; do
    print_header "LWOW Interactive"
    echo "Config: $CONFIG_PATH"
    echo "Campaign: $CAMPAIGN_NAME"
    echo
    echo "1) Attach to existing run"
    echo "2) Start next window"
    echo "3) Show campaign status"
    echo "4) Change config"
    echo "5) Exit"
    echo
    read -r -p "Select option: " option
    case "$option" in
      1)
        if pick_existing_run_dir; then
          if set_metadata_from_manifest; then
            :
          elif [[ -n "$CONFIG_PATH" ]] && [[ -f "$CONFIG_PATH" ]]; then
            set_metadata_from_config
          else
            echo "Manifest invalid. Pick config."
            pick_config
            infer_campaign_from_config
            set_metadata_from_config
          fi
          RUNNER_PID=""
          enter_run_monitor
        fi
        ;;
      2)
        start_next_window
        ;;
      3)
        show_campaign_status
        ;;
      4)
        pick_config
        infer_campaign_from_config
        ;;
      5)
        exit 0
        ;;
      *)
        ;;
    esac
  done
}

main() {
  parse_cli_args "$@"
  print_header "LWOW Interactive"
  ensure_api_key
  pick_config
  infer_campaign_from_config

  if [[ -n "$ATTACH_RUN_DIR" ]]; then
    RUN_DIR="$ATTACH_RUN_DIR"
    if [[ ! -d "$RUN_DIR" ]]; then
      echo "Attach run directory not found: $RUN_DIR" >&2
      exit 1
    fi
    if set_metadata_from_manifest; then
      :
    elif [[ -z "$CONFIG_PATH" ]] || [[ ! -f "$CONFIG_PATH" ]]; then
      echo "Manifest missing. Pick config."
      pick_config
      infer_campaign_from_config
      set_metadata_from_config
    else
      set_metadata_from_config
    fi
    echo
    echo "Attached to run: $RUN_DIR"
    RUNNER_PID=""
    enter_run_monitor
  else
    main_menu
  fi
}

main "$@"
