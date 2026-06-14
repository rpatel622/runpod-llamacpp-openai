#!/usr/bin/env bash
set -euo pipefail

# Codespaces-friendly RunPod Flash deploy helper.
# It supports API-key auth without browser login, and a manual browser flow via
# `flash login --no-open` when preferred.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FLASH_DIR="$ROOT_DIR/flash_app"
VENV_DIR="$FLASH_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_FILE="$FLASH_DIR/.env"
GIT_EXCLUDE="$ROOT_DIR/.git/info/exclude"

cd "$FLASH_DIR"

say() {
  printf '\n==> %s\n' "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

append_git_exclude() {
  if [ -d "$ROOT_DIR/.git" ]; then
    mkdir -p "$(dirname "$GIT_EXCLUDE")"
    touch "$GIT_EXCLUDE"
    grep -qxF "flash_app/.env" "$GIT_EXCLUDE" || echo "flash_app/.env" >> "$GIT_EXCLUDE"
    grep -qxF "flash_app/.venv/" "$GIT_EXCLUDE" || echo "flash_app/.venv/" >> "$GIT_EXCLUDE"
  fi
}

check_python_version() {
  require_cmd "$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if major != 3 or minor not in (10, 11, 12, 13):
    raise SystemExit(f"Python 3.10-3.13 required; found {sys.version.split()[0]}")
print(f"Python {sys.version.split()[0]} OK")
PY
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  say "uv not found; installing uv into the Codespaces user environment"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv install completed but uv is still not on PATH"
}

ensure_venv_and_deps() {
  say "Creating/updating virtual environment"
  uv venv "$VENV_DIR" --python "$PYTHON_BIN"
  uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt
}

write_env_key() {
  local key="$1"
  local tmp
  tmp="$(mktemp)"

  if [ -f "$ENV_FILE" ]; then
    grep -v '^RUNPOD_API_KEY=' "$ENV_FILE" > "$tmp" || true
  fi

  printf 'RUNPOD_API_KEY=%s\n' "$key" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  append_git_exclude
}

load_env_file() {
  if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

auth_with_api_key() {
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    say "RUNPOD_API_KEY is already set in the environment"
    return
  fi

  if [ -f "$ENV_FILE" ] && grep -q '^RUNPOD_API_KEY=' "$ENV_FILE"; then
    load_env_file
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
      say "Loaded RUNPOD_API_KEY from flash_app/.env"
      return
    fi
  fi

  printf '\nPaste your RunPod API key. Input will be hidden.\n'
  printf 'RunPod API key: '
  IFS= read -r -s key
  printf '\n'

  [ -n "$key" ] || fail "RUNPOD_API_KEY cannot be empty"
  export RUNPOD_API_KEY="$key"

  printf 'Persist key to flash_app/.env for future Codespaces commands? [y/N]: '
  IFS= read -r persist
  case "${persist:-}" in
    y|Y|yes|YES)
      write_env_key "$key"
      say "Saved key to flash_app/.env and excluded it from git"
      ;;
    *)
      say "Using key only for this shell process"
      ;;
  esac
}

auth_with_browser_no_open() {
  say "Starting Flash browser auth without auto-open"
  printf 'Copy the URL printed by Flash, open it in your browser, authorize, then return here.\n'
  uv run flash login --no-open --timeout 900 --force
}

choose_auth() {
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    say "Using existing RUNPOD_API_KEY"
    return
  fi

  cat <<'EOF'

Choose RunPod Flash authentication method:

  1) Paste RUNPOD_API_KEY directly. Recommended for GitHub Codespaces.
  2) Browser authorization using `flash login --no-open`.

EOF

  printf 'Choice [1]: '
  IFS= read -r choice
  choice="${choice:-1}"

  case "$choice" in
    1) auth_with_api_key ;;
    2) auth_with_browser_no_open ;;
    *) fail "Invalid choice: $choice" ;;
  esac
}

print_config_summary() {
  load_env_file
  cat <<EOF

Flash deploy configuration:

  Directory:                  $FLASH_DIR
  Model:                      ${MODEL_NAME:-unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL}
  Image:                      ${RUNPOD_LLAMA_IMAGE:-ghcr.io/rpatel622/runpod-llamacpp-openai:latest}
  CPU adapter workers:         ${FLASH_LB_WORKERS_MIN:-1}-${FLASH_LB_WORKERS_MAX:-5}
  GPU queue workers:           ${FLASH_GPU_WORKERS_MIN:-0}-${FLASH_GPU_WORKERS_MAX:-3}
  GPU execution timeout ms:    ${FLASH_GPU_EXECUTION_TIMEOUT_MS:-600000}
  Llama startup timeout sec:   ${LLAMA_STARTUP_TIMEOUT:-1800}
  Queue warmup:                ${QUEUE_WARMUP:-0}

EOF
}

run_deploy() {
  printf 'Run `flash deploy --env production` now? [Y/n]: '
  IFS= read -r deploy_now
  case "${deploy_now:-Y}" in
    y|Y|yes|YES)
      say "Deploying Flash app"
      uv run flash deploy --env production
      ;;
    *)
      cat <<'EOF'

Skipped deploy. To deploy later:

  cd flash_app
  source .venv/bin/activate
  uv run flash deploy --env production

EOF
      ;;
  esac
}

main() {
  say "Preparing RunPod Flash deployment from GitHub Codespaces"
  check_python_version
  ensure_uv
  ensure_venv_and_deps
  load_env_file
  choose_auth
  print_config_summary
  run_deploy
}

main "$@"
