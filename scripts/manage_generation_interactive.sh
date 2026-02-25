#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REFRESH_SEC="${REFRESH_SEC:-2}"
RUNS_ROOT_DEFAULT="data/runs"
ATTACH_RUN_DIR=""
CAMPAIGN_NAME=""
CUE_START_INDEX=""
CUE_COUNT=""

CLR_RESET="$(tput sgr0 2>/dev/null || true)"
CLR_BOLD="$(tput bold 2>/dev/null || true)"
CLR_BLUE="$(tput setaf 4 2>/dev/null || true)"
CLR_GREEN="$(tput setaf 2 2>/dev/null || true)"
CLR_YELLOW="$(tput setaf 3 2>/dev/null || true)"
CLR_RED="$(tput setaf 1 2>/dev/null || true)"
CLR_CYAN="$(tput setaf 6 2>/dev/null || true)"

print_header() {
  printf "%s%sLWOW Interactive Generation Console%s\n" "$CLR_BOLD" "$CLR_CYAN" "$CLR_RESET"
  printf "%s%s%s\n" "$CLR_BLUE" "----------------------------------------" "$CLR_RESET"
}

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

ensure_api_key() {
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ -f ".env" ]]; then
    # Export vars loaded from .env for child processes.
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

pick_mode() {
  echo
  echo "Choose mode:"
  echo "  [1] Start new run"
  echo "  [2] Attach existing run"
  while true; do
    read -r -p "Enter number: " choice
    case "$choice" in
      1) RUN_MODE="start"; return ;;
      2) RUN_MODE="attach"; return ;;
      *) echo "Invalid choice." ;;
    esac
  done
}

pick_existing_run_dir() {
  local runs=()
  local run
  for run in "$RUNS_ROOT_DEFAULT"/*; do
    [[ -d "$run" ]] || continue
    [[ -f "$run/manifest.json" ]] || continue
    runs+=("$run")
  done
  if [[ "${#runs[@]}" -eq 0 ]]; then
    echo "No existing runs with manifest found under $RUNS_ROOT_DEFAULT" >&2
    exit 1
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
      return
    fi
    echo "Invalid choice."
  done
}

pick_config() {
  local configs=()
  local cfg
  for cfg in configs/*.yaml; do
    [[ -e "$cfg" ]] || continue
    configs+=("$cfg")
  done
  if [[ "${#configs[@]}" -gt 0 ]]; then
    local sorted
    sorted="$(printf '%s\n' "${configs[@]}" | sort)"
    configs=()
    while IFS= read -r cfg; do
      [[ -n "$cfg" ]] && configs+=("$cfg")
    done <<< "$sorted"
  fi

  if [[ "${#configs[@]}" -eq 0 ]]; then
    echo "No config files found in configs/*.yaml" >&2
    exit 1
  fi

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

load_manifest_metadata() {
  local manifest_path="$RUN_DIR/manifest.json"
  if [[ ! -f "$manifest_path" ]]; then
    return 1
  fi
  local values
  values="$(MANIFEST_PATH="$manifest_path" "$PYTHON_BIN" - <<'PY'
import json, os
p = os.environ["MANIFEST_PATH"]
with open(p, "r", encoding="utf-8") as f:
    m = json.load(f)
print(m.get("config_path", ""))
print(m.get("raw_output_path", ""))
print(m.get("campaign_name", ""))
PY
)"
  CONFIG_PATH="$(echo "$values" | sed -n '1p')"
  RAW_OUTPUT_PATH="$(echo "$values" | sed -n '2p')"
  CAMPAIGN_NAME="$(echo "$values" | sed -n '3p')"
  return 0
}

set_raw_output_from_config() {
  values="$(CONFIG_PATH="$CONFIG_PATH" "$PYTHON_BIN" - <<'PY'
import os
from lwow.config import load_config
cfg = load_config(os.environ["CONFIG_PATH"])
print(cfg.paths.raw_output_path)
print(cfg.generation.campaign_name)
print(cfg.generation.cue_start_index)
print(cfg.generation.cue_count)
PY
)"
  RAW_OUTPUT_PATH="$(echo "$values" | sed -n '1p')"
  CAMPAIGN_NAME="$(echo "$values" | sed -n '2p')"
  CUE_START_INDEX="$(echo "$values" | sed -n '3p')"
  CUE_COUNT="$(echo "$values" | sed -n '4p')"
}

preview_config() {
  echo
  echo "Config preview: $CONFIG_PATH"
  echo "----------------------------------------"
  sed -n '1,200p' "$CONFIG_PATH"
  echo "----------------------------------------"
  read -r -p "Continue with this config? [y/N]: " yn
  if [[ ! "$yn" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
}

start_run_process() {
  local mode="$1" # start|resume
  mkdir -p "$RUN_DIR"
  local log_path="$RUN_DIR/interactive_runner.log"

  if [[ "$mode" == "start" ]]; then
    PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py start \
      --config "$CONFIG_PATH" \
      --run-dir "$RUN_DIR" \
      >"$log_path" 2>&1 &
  else
    PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py resume \
      --config "$CONFIG_PATH" \
      --run-dir "$RUN_DIR" \
      >"$log_path" 2>&1 &
  fi
  RUNNER_PID=$!
}

start_run_or_show_error() {
  local mode="$1"
  start_run_process "$mode"
  sleep 1
  if runner_alive; then
    return
  fi

  clear
  print_header
  printf "%sRunner failed to start.%s\n" "$CLR_RED" "$CLR_RESET"
  echo "Recent log:"
  echo "----------------------------------------"
  if [[ -f "$RUN_DIR/interactive_runner.log" ]]; then
    tail -n 40 "$RUN_DIR/interactive_runner.log"
  else
    echo "(no log available)"
  fi
  echo "----------------------------------------"
  read -r -p "Press Enter to continue..."
}

runner_alive() {
  if [[ -z "${RUNNER_PID:-}" ]]; then
    return 1
  fi
  kill -0 "$RUNNER_PID" >/dev/null 2>&1
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
  local val
  val="$(echo "$STATUS_LINE" | sed -n "s/.*${key}=\\([^ ]*\\).*/\\1/p")"
  echo "${val:-}"
}

pause_run() {
  PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py pause --run-dir "$RUN_DIR" >/dev/null 2>&1 || true
}

hard_cancel_run() {
  if [[ -z "${CONFIG_PATH:-}" ]] || [[ ! -f "$CONFIG_PATH" ]]; then
    clear
    print_header
    printf "%sHard cancel requires a valid config path.%s\n" "$CLR_RED" "$CLR_RESET"
    echo "Current CONFIG_PATH: ${CONFIG_PATH:-<unset>}"
    read -r -p "Press Enter to return..."
    return
  fi
  PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py pause \
    --run-dir "$RUN_DIR" \
    --cancel-remote \
    --config "$CONFIG_PATH" \
    >/dev/null 2>&1 || true
}

render_dashboard() {
  clear
  print_header

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
  else
    printf "%sRunner Process:%s %snot active%s\n" "$CLR_BOLD" "$CLR_RESET" "$CLR_YELLOW" "$CLR_RESET"
  fi

  echo
  printf "%sControls:%s [p]ause [x]hard-cancel [r]esume [f]ollow-status [l]og-tail [q]uit-monitor\n" "$CLR_BOLD" "$CLR_RESET"
}

show_log_tail() {
  clear
  print_header
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

follow_status() {
  clear
  echo "Attaching to live status (Ctrl+C to return to dashboard)..."
  PYTHONPATH=. "$PYTHON_BIN" scripts/manage_generation.py status --run-dir "$RUN_DIR" --follow || true
  sleep 1
}

main() {
  parse_cli_args "$@"
  print_header
  ensure_api_key

  if [[ -n "$ATTACH_RUN_DIR" ]]; then
    RUN_MODE="attach"
    RUN_DIR="$ATTACH_RUN_DIR"
    if [[ ! -d "$RUN_DIR" ]]; then
      echo "Attach run directory not found: $RUN_DIR" >&2
      exit 1
    fi
  else
    pick_mode
    if [[ "$RUN_MODE" == "attach" ]]; then
      pick_existing_run_dir
    fi
  fi

  if [[ "$RUN_MODE" == "attach" ]]; then
    if ! load_manifest_metadata; then
      echo "Manifest not found in $RUN_DIR; falling back to config picker."
      pick_config
      set_raw_output_from_config
    elif [[ -z "$CONFIG_PATH" ]] || [[ ! -f "$CONFIG_PATH" ]]; then
      echo "Manifest config path missing/unreadable; please pick config."
      pick_config
      set_raw_output_from_config
    fi
    echo
    echo "Attached to existing run: $RUN_DIR"
  else
    pick_config
    preview_config
    set_raw_output_from_config

    local run_id
    run_id="$(date -u +"%Y%m%d-%H%M%S")"
    RUN_DIR="$RUNS_ROOT_DEFAULT/$run_id"
    echo
    echo "Starting run..."
    start_run_or_show_error "start"
  fi

  while true; do
    read_status_line
    render_dashboard
    if read -r -s -n 1 -t "$REFRESH_SEC" key; then
      case "$key" in
        p|P)
          pause_run
          ;;
        x|X)
          hard_cancel_run
          ;;
        r|R)
          if ! runner_alive; then
            start_run_or_show_error "resume"
          fi
          ;;
        f|F)
          follow_status
          ;;
        l|L)
          show_log_tail
          ;;
        q|Q)
          echo
          echo "Monitor exited. Run continues if runner is active."
          exit 0
          ;;
      esac
    fi
  done
}

main "$@"
