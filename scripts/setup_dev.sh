
#!/usr/bin/env bash
# ============================================================
# NeuroWorld AI
# File: run_local.sh
#
# Local Development & AI Execution Launcher
#
# Commands:
#   bash run_local.sh setup
#   bash run_local.sh api
#   bash run_local.sh health
#   bash run_local.sh simulation
#   bash run_local.sh imagination
#   bash run_local.sh train-world
#   bash run_local.sh train-tokenizer
#   bash run_local.sh benchmarks
#   bash run_local.sh trajectories
#   bash run_local.sh preprocess
#   bash run_local.sh datasets
#   bash run_local.sh tests
#   bash run_local.sh lint
#   bash run_local.sh help
#
# Compatible with Linux, macOS, WSL and Git Bash.
# ============================================================

set -Eeuo pipefail

ROOT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
    pwd
)"

cd "$ROOT_DIR"

COMMAND="${1:-api}"

if [[ $# -gt 0 ]]; then
    shift
fi

ARGS=("$@")

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"
LOG_LEVEL="${LOG_LEVEL:-info}"
API_APP="${API_APP:-}"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

info() {
    printf '[INFO] %s\n' "$*"
}

success() {
    printf '[OK] %s\n' "$*"
}

warn() {
    printf '[WARN] %s\n' "$*" >&2
}

error() {
    printf '[ERROR] %s\n' "$*" >&2
}

die() {
    error "$*"
    exit 1
}

on_error() {
    local exit_code="$1"
    local line="$2"

    error "Failed at line ${line} (exit ${exit_code})"
}

trap 'on_error "$?" "$LINENO"' ERR

# ------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------

load_env() {
    if [[ ! -f ".env" ]]; then
        return
    fi

    info "Reading .env"

    # Read literal KEY=VALUE entries.
    # Do not execute .env as a shell script.
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"

        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue

        if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            local key="${line%%=*}"
            local value="${line#*=}"

            if [[ ${#value} -ge 2 ]]; then
                if [[ "${value:0:1}" == '"' &&
                      "${value: -1}" == '"' ]]; then
                    value="${value:1:${#value}-2}"
                elif [[ "${value:0:1}" == "'" &&
                        "${value: -1}" == "'" ]]; then
                    value="${value:1:${#value}-2}"
                fi
            fi

            if [[ ! -v "$key" ]]; then
                printf -v "$key" '%s' "$value"
                export "$key"
            fi
        fi
    done < ".env"

    HOST="${HOST:-127.0.0.1}"
    PORT="${PORT:-8000}"
    RELOAD="${RELOAD:-true}"
    LOG_LEVEL="${LOG_LEVEL:-info}"
    API_APP="${API_APP:-}"
}

# ------------------------------------------------------------
# Python discovery
# ------------------------------------------------------------

find_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
        if [[ -x "$PYTHON_BIN" ]] ||
           command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            return
        fi

        die "Python executable not found: $PYTHON_BIN"
    fi

    if [[ -f "$VENV_DIR/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/$VENV_DIR/bin/python"

    elif [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
        PYTHON_BIN="$ROOT_DIR/$VENV_DIR/Scripts/python.exe"

    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"

    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"

    else
        die "Python not found. Run setup_dev.sh first."
    fi
}

check_python() {
    "$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "NeuroWorld AI requires Python 3.10+"
    )

print(
    f"[OK] Python {sys.version.split()[0]}"
)
PY
}

check_module() {
    local module="$1"

    "$PYTHON_BIN" - "$module" <<'PY'
import importlib.util
import sys

module_name = sys.argv[1]

try:
    available = (
        importlib.util.find_spec(module_name)
        is not None
    )
except (ImportError, ModuleNotFoundError, ValueError):
    available = False

sys.exit(0 if available else 1)
PY
}

require_module() {
    local module="$1"
    local package="${2:-$module}"

    if ! check_module "$module"; then
        die "Missing ${module}. Install: python -m pip install ${package}"
    fi
}

# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------

prepare_directories() {
    local directories=(
        "data/raw/videos"
        "data/processed/videos/train"
        "data/processed/videos/val"
        "data/processed/videos/test"
        "data/trajectories"
        "data/tokens/train"
        "data/tokens/val"
        "data/tokens/test"
        "models/checkpoints"
        "artifacts/tokenizer/checkpoints"
        "artifacts/world_model/checkpoints"
        "artifacts/imagination"
        "artifacts/simulations"
        "artifacts/evaluations/benchmarks"
        "artifacts/reports"
        "logs"
    )

    for directory in "${directories[@]}"; do
        mkdir -p "$directory"
    done
}

# ------------------------------------------------------------
# Script execution
# ------------------------------------------------------------

run_python_script() {
    local script="$1"
    shift

    if [[ ! -f "$script" ]]; then
        die "Required script not found: $script"
    fi

    find_python
    check_python

    info "Running: $script"

    "$PYTHON_BIN" "$script" "$@"
}

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

run_setup() {
    if [[ -f "setup_dev.sh" ]]; then
        info "Running development setup"
        bash setup_dev.sh "${ARGS[@]}"

    elif [[ -f "bootstrap.sh" ]]; then
        info "Running bootstrap"
        bash bootstrap.sh "${ARGS[@]}"

    else
        die "Neither setup_dev.sh nor bootstrap.sh exists"
    fi
}

# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------

detect_api_app() {
    if [[ -n "$API_APP" ]]; then
        return
    fi

    if [[ -f "neuroworld/api/main.py" ]]; then
        API_APP="neuroworld.api.main:app"

    elif [[ -f "app/main.py" ]]; then
        API_APP="app.main:app"

    elif [[ -f "main.py" ]]; then
        API_APP="main:app"

    else
        die "FastAPI entrypoint not found. Set API_APP."
    fi
}

run_api() {
    find_python
    check_python

    require_module "fastapi"
    require_module "uvicorn"

    detect_api_app

    local reload_args=()

    case "${RELOAD,,}" in
        true|yes|1|on)
            reload_args=(--reload)
            ;;
        false|no|0|off)
            ;;
        *)
            die "RELOAD must be true or false"
            ;;
    esac

    info "Starting NeuroWorld AI FastAPI"
    info "Application: $API_APP"
    info "URL: http://${HOST}:${PORT}"
    info "Docs: http://${HOST}:${PORT}/docs"

    exec "$PYTHON_BIN" -m uvicorn \
        "$API_APP" \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        "${reload_args[@]}" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Health
# ------------------------------------------------------------

run_health() {
    run_python_script \
        "scripts/health_check.py" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

run_simulation() {
    require_dependencies_for_simulation

    run_python_script \
        "scripts/run_simulation.py" \
        "${ARGS[@]}"
}

require_dependencies_for_simulation() {
    find_python
    require_module "numpy"
}

# ------------------------------------------------------------
# Imagination
# ------------------------------------------------------------

run_imagination() {
    find_python

    require_module "torch"

    run_python_script \
        "scripts/run_imagination.py" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# World-model training
# ------------------------------------------------------------

run_world_training() {
    find_python

    require_module "torch"
    require_module "numpy"

    run_python_script \
        "scripts/train_world_model.py" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Tokenizer training
# ------------------------------------------------------------

run_tokenizer_training() {
    find_python

    require_module "torch"
    require_module "numpy"

    run_python_script \
        "scripts/train_tokenizer.py" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------

run_benchmarks() {
    run_python_script \
        "scripts/run_benchmarks.py" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Trajectory generation
# ------------------------------------------------------------

run_trajectories() {
    run_python_script \
        "scripts/build_trajectories.py" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Video preprocessing
# ------------------------------------------------------------

run_preprocessing() {
    find_python

    require_module "cv2" "opencv-python"
    require_module "numpy"

    run_python_script \
        "scripts/preprocess_videos.py" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Dataset management
# ------------------------------------------------------------

run_datasets() {
    run_python_script \
        "scripts/download_datasets.py" \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

run_tests() {
    find_python
    require_module "pytest"

    if [[ ! -d "tests" ]]; then
        die "Tests directory not found"
    fi

    info "Running NeuroWorld AI tests"

    "$PYTHON_BIN" -m pytest \
        tests \
        "${ARGS[@]}"
}

# ------------------------------------------------------------
# Linting
# ------------------------------------------------------------

run_lint() {
    find_python
    require_module "ruff"

    info "Running Ruff"

    if [[ ${#ARGS[@]} -gt 0 ]]; then
        "$PYTHON_BIN" -m ruff check \
            "${ARGS[@]}"
    else
        "$PYTHON_BIN" -m ruff check \
            neuroworld scripts
    fi
}

# ------------------------------------------------------------
# Status
# ------------------------------------------------------------

show_status() {
    find_python
    check_python

    echo
    echo "=========================================="
    echo "       NEUROWORLD AI LOCAL STATUS"
    echo "=========================================="

    echo "Project: $ROOT_DIR"
    echo "Python: $PYTHON_BIN"
    echo "Environment: $VENV_DIR"
    echo "Host: $HOST"
    echo "Port: $PORT"

    echo
    echo "Available scripts:"

    local scripts=(
        "scripts/health_check.py"
        "scripts/run_simulation.py"
        "scripts/run_imagination.py"
        "scripts/train_world_model.py"
        "scripts/train_tokenizer.py"
        "scripts/run_benchmarks.py"
        "scripts/build_trajectories.py"
        "scripts/preprocess_videos.py"
        "scripts/download_datasets.py"
    )

    for script in "${scripts[@]}"; do
        if [[ -f "$script" ]]; then
            printf '  [OK] %s\n' "$script"
        else
            printf '  [MISSING] %s\n' "$script"
        fi
    done
}

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------

show_help() {
    cat <<'HELP'
========================================================
                NEUROWORLD AI
             LOCAL DEVELOPMENT CLI
========================================================

Usage:
    bash run_local.sh [COMMAND] [OPTIONS]

Commands:

  setup            Setup development environment
  api              Start FastAPI server
  status           Show environment status
  health           Run health diagnostics
  simulation       Run robotics simulation
  imagination      Run latent imagination
  train-world      Train world model
  train-tokenizer  Train VQ-VAE tokenizer
  benchmarks       Run evaluation benchmarks
  trajectories     Generate robot trajectories
  preprocess       Preprocess video datasets
  datasets         Download/manage datasets
  tests            Run unit/integration tests
  lint             Run Ruff code analysis
  help             Show this help

Examples:

  bash run_local.sh setup --dev --cpu

  bash run_local.sh api

  bash run_local.sh health --verbose

  bash run_local.sh simulation \
      --scenario warehouse \
      --robots 2 \
      --steps 500

  bash run_local.sh trajectories \
      --episodes 200

  bash run_local.sh train-world \
      --epochs 100 \
      --batch-size 128

  bash run_local.sh train-tokenizer \
      --epochs 50 \
      --batch-size 32

  bash run_local.sh imagination \
      --checkpoint models/checkpoints/world_model.pt \
      --plot

  bash run_local.sh benchmarks \
      --suite navigation \
      --episodes 20

  bash run_local.sh preprocess \
      --input data/raw/videos

  bash run_local.sh datasets list

  bash run_local.sh tests -v

  bash run_local.sh lint

Environment:

  HOST=127.0.0.1
  PORT=8000
  RELOAD=true
  LOG_LEVEL=info
  API_APP=neuroworld.api.main:app
  VENV_DIR=.venv

========================================================
HELP
}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

main() {
    load_env

    case "$COMMAND" in
        setup)
            run_setup
            ;;

        api|server|start)
            prepare_directories
            run_api
            ;;

        status)
            show_status
            ;;

        health)
            run_health
            ;;

        simulation|simulate)
            prepare_directories
            run_simulation
            ;;

        imagination|imagine)
            prepare_directories
            run_imagination
            ;;

        train-world|train-world-model)
            prepare_directories
            run_world_training
            ;;

        train-tokenizer|tokenizer)
            prepare_directories
            run_tokenizer_training
            ;;

        benchmarks|benchmark)
            prepare_directories
            run_benchmarks
            ;;

        trajectories|trajectory)
            prepare_directories
            run_trajectories
            ;;

        preprocess|videos)
            prepare_directories
            run_preprocessing
            ;;

        datasets|dataset)
            prepare_directories
            run_datasets
            ;;

        tests|test)
            run_tests
            ;;

        lint)
            run_lint
            ;;

        help|-h|--help)
            show_help
            ;;

        *)
            error "Unknown command: $COMMAND"
            show_help
            exit 2
            ;;
    esac
}

main
