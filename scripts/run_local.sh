
#!/usr/bin/env bash
# ============================================================
# NeuroWorld AI
# File: setup_dev.sh
#
# Development Environment Setup
#
# Features:
#   - Python 3.10+ validation
#   - Virtual environment creation
#   - pip and dependency installation
#   - PyTorch CPU/CUDA selection
#   - FastAPI development dependencies
#   - ML and world-model dependencies
#   - Dataset and artifact directories
#   - Environment configuration
#   - Git hooks (optional)
#   - Pre-commit setup (optional)
#   - Dependency verification
#   - Optional tests and health checks
#
# Usage:
#   bash setup_dev.sh
#   bash setup_dev.sh --dev
#   bash setup_dev.sh --cpu
#   bash setup_dev.sh --cuda
#   bash setup_dev.sh --all
#   bash setup_dev.sh --verify
#   bash setup_dev.sh --skip-install
#   bash setup_dev.sh --help
#
# Python 3.10+
# ============================================================

set -Eeuo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

cd "$PROJECT_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"

INSTALL_DEV=false
INSTALL_ALL=false
SKIP_INSTALL=false
VERIFY_ONLY=false
RUN_TESTS=false
RUN_HEALTH=false
SETUP_HOOKS=false

TORCH_MODE="auto"

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
    local line_number="$2"

    error "Setup failed at line ${line_number}"
    error "Exit code: ${exit_code}"
}

trap 'on_error "$?" "$LINENO"' ERR

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------

show_help() {
    cat <<'HELP'
NeuroWorld AI - Development Setup

Usage:
    bash setup_dev.sh [OPTIONS]

Options:
    --dev           Install development tools
    --all           Install core + development + extras
    --cpu           Install CPU-only PyTorch
    --cuda          Install CUDA PyTorch
    --auto          Use standard PyTorch installation
    --verify        Verify existing environment only
    --skip-install  Skip dependency installation
    --tests         Run pytest after setup
    --health        Run project health check
    --hooks         Configure pre-commit hooks
    --help          Show this help

Examples:
    bash setup_dev.sh

    bash setup_dev.sh --dev

    bash setup_dev.sh --cpu --dev

    bash setup_dev.sh --cuda --all

    bash setup_dev.sh --verify

    bash setup_dev.sh --dev --tests

Environment variables:
    PYTHON_BIN    Python executable
    VENV_DIR      Virtual environment directory
HELP
}

# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dev)
                INSTALL_DEV=true
                ;;
            --all)
                INSTALL_ALL=true
                INSTALL_DEV=true
                ;;
            --cpu)
                TORCH_MODE="cpu"
                ;;
            --cuda)
                TORCH_MODE="cuda"
                ;;
            --auto)
                TORCH_MODE="auto"
                ;;
            --verify)
                VERIFY_ONLY=true
                ;;
            --skip-install)
                SKIP_INSTALL=true
                ;;
            --tests)
                RUN_TESTS=true
                ;;
            --health)
                RUN_HEALTH=true
                ;;
            --hooks)
                SETUP_HOOKS=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
        shift
    done

    if [[ "$VERIFY_ONLY" == true ]]; then
        SKIP_INSTALL=true
    fi
}

# ------------------------------------------------------------
# Python discovery
# ------------------------------------------------------------

find_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
        command -v "$PYTHON_BIN" >/dev/null 2>&1 ||
            [[ -x "$PYTHON_BIN" ]] ||
            die "Python not found: $PYTHON_BIN"
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        die "Python is not installed"
    fi
}

check_python_version() {
    "$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "NeuroWorld AI requires Python 3.10+"
    )

print(
    "[OK] Python version:",
    sys.version.split()[0]
)
PY
}

# ------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------

create_virtualenv() {
    if [[ -d "$VENV_DIR" ]]; then
        success "Virtual environment exists"
        return
    fi

    info "Creating virtual environment"

    "$PYTHON_BIN" -m venv "$VENV_DIR"

    success "Virtual environment created"
}

activate_virtualenv() {
    local venv_python=""

    if [[ -f "$VENV_DIR/bin/python" ]]; then
        venv_python="$VENV_DIR/bin/python"
    elif [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
        venv_python="$VENV_DIR/Scripts/python.exe"
    else
        die "Virtual environment Python not found"
    fi

    PYTHON_BIN="$PROJECT_ROOT/$venv_python"

    check_python_version
}

# ------------------------------------------------------------
# Dependency installation
# ------------------------------------------------------------

upgrade_pip() {
    info "Upgrading pip and build tools"

    "$PYTHON_BIN" -m pip install \
        --upgrade \
        pip \
        setuptools \
        wheel
}

install_pytorch() {
    info "Installing PyTorch: $TORCH_MODE"

    case "$TORCH_MODE" in
        cpu)
            "$PYTHON_BIN" -m pip install \
                torch \
                torchvision \
                --index-url \
                https://download.pytorch.org/whl/cpu
            ;;

        cuda)
            # CUDA wheel availability depends on
            # OS, Python version and GPU driver.
            # Override this index if your supported
            # PyTorch/CUDA combination differs.
            local cuda_index="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

            "$PYTHON_BIN" -m pip install \
                torch \
                torchvision \
                --index-url "$cuda_index"
            ;;

        auto)
            "$PYTHON_BIN" -m pip install \
                torch \
                torchvision
            ;;
    esac
}

install_core_dependencies() {
    info "Installing NeuroWorld core dependencies"

    "$PYTHON_BIN" -m pip install \
        numpy \
        scipy \
        pandas \
        pydantic \
        python-dotenv \
        pyyaml \
        tqdm \
        rich \
        psutil
}

install_api_dependencies() {
    info "Installing FastAPI dependencies"

    "$PYTHON_BIN" -m pip install \
        fastapi \
        "uvicorn[standard]" \
        python-multipart \
        httpx \
        websockets
}

install_vision_dependencies() {
    info "Installing computer vision dependencies"

    "$PYTHON_BIN" -m pip install \
        opencv-python \
        pillow \
        imageio \
        imageio-ffmpeg \
        matplotlib
}

install_ml_dependencies() {
    info "Installing ML training dependencies"

    "$PYTHON_BIN" -m pip install \
        scikit-learn \
        tensorboard \
        safetensors
}

install_dev_dependencies() {
    info "Installing development dependencies"

    "$PYTHON_BIN" -m pip install \
        pytest \
        pytest-cov \
        pytest-asyncio \
        ruff \
        black \
        mypy \
        pre-commit \
        ipykernel
}

install_optional_dependencies() {
    info "Installing optional research dependencies"

    "$PYTHON_BIN" -m pip install \
        transformers \
        accelerate \
        datasets \
        einops \
        timm \
        huggingface-hub \
        mlflow
}

install_project() {
    if [[ -f "requirements.txt" ]]; then
        info "Installing requirements.txt"

        "$PYTHON_BIN" -m pip install \
            -r requirements.txt
    fi

    if [[ -f "pyproject.toml" ]] ||
       [[ -f "setup.py" ]]; then

        info "Installing NeuroWorld package"

        "$PYTHON_BIN" -m pip install \
            -e .
    fi
}

install_dependencies() {
    upgrade_pip

    install_pytorch
    install_core_dependencies
    install_api_dependencies
    install_vision_dependencies
    install_ml_dependencies

    if [[ "$INSTALL_DEV" == true ]]; then
        install_dev_dependencies
    fi

    if [[ "$INSTALL_ALL" == true ]]; then
        install_optional_dependencies
    fi

    install_project

    success "Dependency installation completed"
}

# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------

create_directories() {
    info "Creating project directories"

    local directories=(
        "data/raw"
        "data/raw/videos"
        "data/processed"
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
        "configs"
        "scripts"
        "tests"
    )

    for directory in "${directories[@]}"; do
        mkdir -p "$directory"
    done

    success "Project directories ready"
}

# ------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------

create_env_file() {
    if [[ -f ".env" ]]; then
        success ".env already exists"
        return
    fi

    info "Creating .env"

    cat > ".env" <<'ENV'
# ============================================================
# NeuroWorld AI - Local Development Configuration
# ============================================================

APP_NAME=NeuroWorld AI
APP_ENV=development
DEBUG=true

HOST=127.0.0.1
PORT=8000
RELOAD=true
LOG_LEVEL=info

# Data
DATA_DIR=data
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed

# Models
MODEL_DIR=models
CHECKPOINT_DIR=models/checkpoints

# Training
DEVICE=auto
RANDOM_SEED=42
BATCH_SIZE=32
LEARNING_RATE=0.0003

# Simulation
SIMULATION_TIMESTEP=0.1
SIMULATION_MAX_STEPS=500

# Outputs
ARTIFACT_DIR=artifacts
LOG_DIR=logs
ENV

    success ".env created"
}

# ------------------------------------------------------------
# Git configuration
# ------------------------------------------------------------

create_gitignore() {
    if [[ -f ".gitignore" ]]; then
        return
    fi

    cat > ".gitignore" <<'GIT'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Environments
.venv/
venv/
.env
.env.*

# IDE
.vscode/
.idea/

# Generated data and model outputs
data/raw/
data/processed/
data/trajectories/
data/tokens/
models/checkpoints/
artifacts/
logs/

# Notebook checkpoints
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
GIT

    success ".gitignore created"
}

setup_git_hooks() {
    if [[ "$SETUP_HOOKS" != true ]]; then
        return
    fi

    if [[ ! -d ".git" ]]; then
        warn "Not a Git repository; skipping hooks"
        return
    fi

    if [[ ! -f ".pre-commit-config.yaml" ]]; then
        warn "No .pre-commit-config.yaml found"
        return
    fi

    if ! "$PYTHON_BIN" -m pre_commit --version \
        >/dev/null 2>&1; then
        warn "pre-commit is not installed"
        return
    fi

    "$PYTHON_BIN" -m pre_commit install

    success "Pre-commit hooks configured"
}

# ------------------------------------------------------------
# Dependency verification
# ------------------------------------------------------------

verify_environment() {
    info "Verifying NeuroWorld environment"

    "$PYTHON_BIN" - <<'PY'
import importlib
import sys

modules = {
    "numpy": "NumPy",
    "torch": "PyTorch",
    "fastapi": "FastAPI",
    "uvicorn": "Uvicorn",
    "cv2": "OpenCV",
    "PIL": "Pillow",
    "matplotlib": "Matplotlib",
    "pydantic": "Pydantic",
}

failed = []

for module_name, label in modules.items():
    try:
        module = importlib.import_module(
            module_name
        )

        version = getattr(
            module,
            "__version__",
            "installed"
        )

        print(f"[OK] {label}: {version}")

    except Exception as exc:
        print(
            f"[FAIL] {label}: {exc}"
        )
        failed.append(label)

try:
    import torch

    print(
        "[INFO] CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        print(
            "[INFO] GPU:",
            torch.cuda.get_device_name(0)
        )

except ImportError:
    pass

if failed:
    print(
        "[ERROR] Missing or broken modules:",
        ", ".join(failed)
    )
    sys.exit(1)

print("[OK] Core environment verified")
PY
}

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

run_tests() {
    if [[ "$RUN_TESTS" != true ]]; then
        return
    fi

    if [[ ! -d "tests" ]]; then
        warn "No tests directory found"
        return
    fi

    info "Running pytest"

    "$PYTHON_BIN" -m pytest \
        tests \
        -v
}

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

run_health_check() {
    if [[ "$RUN_HEALTH" != true ]]; then
        return
    fi

    if [[ ! -f "scripts/health_check.py" ]]; then
        warn "Health-check script not found"
        return
    fi

    info "Running NeuroWorld health check"

    "$PYTHON_BIN" \
        scripts/health_check.py
}

# ------------------------------------------------------------
# Completion
# ------------------------------------------------------------

show_completion() {
    echo
    echo "=================================================="
    echo "        NEUROWORLD AI DEVELOPMENT SETUP"
    echo "=================================================="
    echo
    success "Development environment is ready"
    echo
    echo "Python:"
    "$PYTHON_BIN" --version
    echo
    echo "Virtual environment: $VENV_DIR"
    echo "Project root: $PROJECT_ROOT"
    echo
    echo "Next commands:"
    echo
    echo "  bash run_local.sh api"
    echo "  bash run_local.sh health"
    echo "  bash run_local.sh trajectories"
    echo "  bash run_local.sh benchmarks"
    echo
    echo "Training:"
    echo
    echo "  python scripts/train_tokenizer.py"
    echo "  python scripts/train_world_model.py"
    echo
    echo "Simulation:"
    echo
    echo "  python scripts/run_simulation.py"
    echo
    echo "Imagination:"
    echo
    echo "  python scripts/run_imagination.py --help"
    echo
    echo "=================================================="
}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

main() {
    parse_args "$@"

    info "NeuroWorld AI development setup"
    info "Project: $PROJECT_ROOT"

    find_python
    check_python_version

    if [[ "$VERIFY_ONLY" == true ]]; then
        if [[ -d "$VENV_DIR" ]]; then
            activate_virtualenv
        fi

        verify_environment
        success "Verification completed"
        exit 0
    fi

    create_directories
    create_env_file
    create_gitignore

    create_virtualenv
    activate_virtualenv

    if [[ "$SKIP_INSTALL" == false ]]; then
        install_dependencies
    else
        warn "Dependency installation skipped"
    fi

    verify_environment

    setup_git_hooks
    run_tests
    run_health_check

    show_completion
}

main "$@"
