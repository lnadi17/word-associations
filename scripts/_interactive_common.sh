#!/usr/bin/env bash
# Shared helpers for interactive scripts. Source with: source "$(dirname "${BASH_SOURCE[0]}")/_interactive_common.sh"

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REFRESH_SEC="${REFRESH_SEC:-2}"
RUNS_ROOT_DEFAULT="data/runs"

CLR_RESET="$(tput sgr0 2>/dev/null || true)"
CLR_BOLD="$(tput bold 2>/dev/null || true)"
CLR_BLUE="$(tput setaf 4 2>/dev/null || true)"
CLR_GREEN="$(tput setaf 2 2>/dev/null || true)"
CLR_YELLOW="$(tput setaf 3 2>/dev/null || true)"
CLR_RED="$(tput setaf 1 2>/dev/null || true)"
CLR_CYAN="$(tput setaf 6 2>/dev/null || true)"

print_header() {
  local title="${1:-LWOW Interactive}"
  [ -t 1 ] && clear || true
  printf "%s%s%s%s\n" "$CLR_BOLD" "$CLR_CYAN" "$title" "$CLR_RESET"
  printf "%s%s%s\n" "$CLR_BLUE" "----------------------------------------" "$CLR_RESET"
}

ensure_api_key() {
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
  fi

  if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo
    echo "ANTHROPIC_API_KEY or OPENAI_API_KEY is not set."
    read -r -s -p "Enter API key (input hidden): " API_KEY
    echo
    if [[ -z "${API_KEY:-}" ]]; then
      echo "No API key provided. Exiting."
      exit 1
    fi
    export ANTHROPIC_API_KEY="$API_KEY"
    export OPENAI_API_KEY="$API_KEY"
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
