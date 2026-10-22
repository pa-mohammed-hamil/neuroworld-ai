
#!/usr/bin/env bash
# ============================================================
# NeuroWorld AI — Bootstrap Script
# File: bootstrap.sh
#
# Usage:
#   bash bootstrap.sh
#   bash bootstrap.sh --dev
#   bash bootstrap.sh --test
#   bash bootstrap.sh --run
#   bash bootstrap.sh --help
#
# Supports Linux, macOS, and Windows Git Bash / WSL.
# ============================================================

set -Eeuo pipefail

PROJECT_NAME="NeuroWorld AI"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_DEV=false
RUN_TESTS=false
START_SERVER=false
SKIP_INSTALL=false

log() {
    printf '\033[1;36m[NeuroWorld]\033[0m %s\n' "$*"
}

success() {
    printf '\033[1;32m[OK]\033[0m %s\n' "$*"
}

warn() {
    printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2
}

error() {
    printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2
}

fail() {
    error "$*"
    exit 1
}

on_error() {
    error "Bootstrap failed at line $1."
}

trap 'on_error "$LINENO"' ERR

show_help() {
    cat <<'EOF'
NeuroWorld AI Bootstrap

Usage:
  bash bootstrap.sh [OPTIONS]

Options:
  --dev           Install development dependencies
  --test          Run available tests
  --run           Start the FastAPI development server
  --skip-install  Skip dependency installation
  --help          Show this help message

Examples:
  bash bootstrap.sh
  bash bootstrap.sh --dev --test
  bash bootstrap.sh --run
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dev)
                INSTALL_DEV=true
                ;;
            --test)
                RUN_TESTS=true
                ;;
            --run)
                START_SERVER=true
                ;;
            --skip-install)
                SKIP_INSTALL=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                fail "Unknown argument: $1"
                ;;
        esac
        shift
    done
}

find_python() {
    local candidate

    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c \
                'import sys; assert sys.version_info >= (3, 10)' \
                >/dev/null 2>&1; then
                PYTHON_CMD="$candidate"
                return 0
            fi
        fi
    done

    fail "Python 3.10+ is required."
}

show_python_info() {
    log "Checking Python installation..."

    "$PYTHON_CMD" --version

    "$PYTHON_CMD" - <<'PY'
import platform
import sys

print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("Architecture:", platform.machine())
PY
}

create_virtual_environment() {
    log "Preparing virtual environment..."

    if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        success "Virtual environment created."
    else
        success "Existing virtual environment detected."
    fi

    if [[ -x "$VENV_DIR/bin/python" ]]; then
        VENV_PYTHON="$VENV_DIR/bin/python"
    elif [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
        VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
    else
        fail "Virtual environment Python executable not found."
    fi

    "$VENV_PYTHON" -m pip --version \
        >/dev/null 2>&1 || \
        "$VENV_PYTHON" -m ensurepip --upgrade

    success "Python environment ready."
}

upgrade_pip() {
    log "Updating Python packaging tools..."

    "$VENV_PYTHON" -m pip install --upgrade \
        pip setuptools wheel
}

install_requirements() {
    if [[ "$SKIP_INSTALL" == true ]]; then
        warn "Dependency installation skipped."
        return
    fi

    upgrade_pip

    if [[ -f "requirements.txt" ]]; then
        log "Installing requirements.txt..."

        "$VENV_PYTHON" -m pip install \
            -r requirements.txt
    elif [[ -f "pyproject.toml" ]]; then
        log "Installing project from pyproject.toml..."

        "$VENV_PYTHON" -m pip install -e .
    else
        warn "No requirements.txt or pyproject.toml found."
        warn "Installing minimal development dependencies."

        "$VENV_PYTHON" -m pip install \
            numpy \
            pydantic \
            python-dotenv \
            fastapi \
            "uvicorn[standard]"
    fi

    if [[ "$INSTALL_DEV" == true ]]; then
        install_dev_dependencies
    fi

    success "Dependency installation finished."
}

install_dev_dependencies() {
    log "Installing development tools..."

    if [[ -f "requirements-dev.txt" ]]; then
        "$VENV_PYTHON" -m pip install \
            -r requirements-dev.txt
    else
        "$VENV_PYTHON" -m pip install \
            pytest \
            pytest-asyncio \
            pytest-cov \
            ruff \
            black \
            mypy
    fi
}

create_directories() {
    log "Creating project directories..."

    local directories=(
        "artifacts"
        "artifacts/checkpoints"
        "artifacts/experiments"
        "artifacts/world_runs"
        "artifacts/navigation"
        "artifacts/simulations"
        "artifacts/evaluations"
        "artifacts/reports"
        "artifacts/visualizations"
        "logs"
        "data/raw"
        "data/processed"
        "data/simulations"
        "data/trajectories"
        "data/datasets"
        "models/checkpoints"
        "models/exports"
        "models/registry"
        "configs"
        "scripts"
        "tests"
    )

    local directory

    for directory in "${directories[@]}"; do
        mkdir -p "$directory"
    done

    success "Project directories ready."
}

setup_environment_file() {
    log "Checking environment configuration..."

    if [[ -f ".env" ]]; then
        success ".env already exists."
        return
    fi

    if [[ -f ".env.example" ]]; then
        cp ".env.example" ".env"
        success ".env created from .env.example."
        return
    fi

    cat > ".env" <<'EOF'
# NeuroWorld AI local configuration
APP_NAME=NeuroWorld AI
APP_ENV=development
DEBUG=true

HOST=127.0.0.1
PORT=8000

LOG_LEVEL=INFO
LOG_DIR=logs

MODEL_DIR=models
DATA_DIR=data
ARTIFACT_DIR=artifacts

WORLD_WIDTH=10.0
WORLD_HEIGHT=10.0
SIMULATION_TIMESTEP=0.1

DEFAULT_LATENT_DIM=128
DEFAULT_ACTION_DIM=32

ENABLE_RENDERING=false
ENABLE_COLLISION_DETECTION=true
ENABLE_OBSTACLE_AVOIDANCE=true
EOF

    success "Default .env created."
}

setup_gitignore() {
    if [[ -f ".gitignore" ]]; then
        return
    fi

    cat > ".gitignore" <<'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/

# Secrets and local configuration
.env
.env.*
!.env.example

# Logs
logs/
*.log

# Generated data
artifacts/
models/checkpoints/
models/exports/
data/raw/
data/processed/
data/simulations/
data/trajectories/

# Editors
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
EOF

    success ".gitignore created."
}

check_imports() {
    log "Checking core imports..."

    "$VENV_PYTHON" - <<'PY'
import importlib.util

packages = [
    "numpy",
    "torch",
    "fastapi",
    "pydantic",
    "matplotlib",
    "PIL",
    "pytest",
]

for package in packages:
    installed = importlib.util.find_spec(package) is not None
    status = "OK" if installed else "NOT INSTALLED"
    print(f"{package:15} {status}")

print("\nNeuroWorld package check:")

spec = importlib.util.find_spec("neuroworld")

if spec is None:
    print("neuroworld      NOT FOUND")
    print("Check package layout or project installation.")
else:
    print("neuroworld      OK")
PY
}

check_project_structure() {
    log "Checking project modules..."

    local modules=(
        "neuroworld/environment/scenario.py"
        "neuroworld/environment/simulator.py"
        "neuroworld/environment/world_runner.py"
        "neuroworld/environment/navigation.py"
        "neuroworld/environment/obstacle_avoidance.py"
        "neuroworld/environment/object_interaction.py"
        "neuroworld/physics/collision.py"
        "neuroworld/physics/object_push.py"
        "neuroworld/agents/navigation_agent.py"
        "neuroworld/agents/world_agent.py"
    )

    local module

    for module in "${modules[@]}"; do
        if [[ -f "$module" ]]; then
            success "$module"
        else
            warn "Optional module missing: $module"
        fi
    done
}

run_tests() {
    log "Running project tests..."

    if ! "$VENV_PYTHON" -m pytest --version \
        >/dev/null 2>&1; then
        fail "pytest is not installed. Use --dev."
    fi

    if [[ ! -d "tests" ]]; then
        warn "No tests directory found."
        return
    fi

    if ! find tests -type f \
        \( -name 'test_*.py' -o -name '*_test.py' \) \
        -print -quit | grep -q .; then
        warn "No test files found."
        return
    fi

    "$VENV_PYTHON" -m pytest tests -v
}

start_server() {
    log "Starting NeuroWorld AI API..."

    local api_module=""

    if [[ -f "neuroworld/api/main.py" ]]; then
        api_module="neuroworld.api.main:app"
    elif [[ -f "app/main.py" ]]; then
        api_module="app.main:app"
    elif [[ -f "main.py" ]]; then
        api_module="main:app"
    else
        fail "No FastAPI entrypoint found."
    fi

    "$VENV_PYTHON" -c \
        "import uvicorn" \
        >/dev/null 2>&1 || \
        fail "uvicorn is not installed."

    local host="${HOST:-127.0.0.1}"
    local port="${PORT:-8000}"

    log "API module: $api_module"
    log "URL: http://${host}:${port}"

    exec "$VENV_PYTHON" -m uvicorn \
        "$api_module" \
        --host "$host" \
        --port "$port" \
        --reload
}

show_completion() {
    echo
    success "$PROJECT_NAME bootstrap completed."
    echo
    echo "Virtual environment: $VENV_DIR"
    echo "Python: $("$VENV_PYTHON" --version)"
    echo
    echo "Next commands:"
    echo "  source $VENV_DIR/bin/activate   # Linux/macOS/WSL"
    echo "  bash bootstrap.sh --test"
    echo "  bash bootstrap.sh --run"
    echo
}

main() {
    parse_args "$@"

    log "========================================"
    log "        NeuroWorld AI Bootstrap"
    log "========================================"

    find_python
    show_python_info
    create_virtual_environment
    install_requirements
    create_directories
    setup_environment_file
    setup_gitignore
    check_imports
    check_project_structure

    if [[ "$RUN_TESTS" == true ]]; then
        run_tests
    fi

    show_completion

    if [[ "$START_SERVER" == true ]]; then
        start_server
    fi
}

main "$@"
