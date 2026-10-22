
```python
#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/health_check.py

System Health Check and Diagnostics

Features:
- Python version verification
- Operating system information
- CPU and memory checks
- Disk space monitoring
- Python dependency checks
- PyTorch and CUDA diagnostics
- GPU memory information
- NeuroWorld module verification
- Dataset availability
- Model checkpoint discovery
- Configuration validation
- Optional FastAPI health endpoint
- JSON health reports
- CLI support
- Exit codes for CI/CD

Dependencies:
    Python 3.10+

Optional:
    psutil
    torch
    requests

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --verbose
    python scripts/health_check.py --json
    python scripts/health_check.py --check-api
    python scripts/health_check.py --strict
    python scripts/health_check.py --save-report

Exit codes:
    0 = Healthy or warnings only
    1 = Critical failure
    2 = Warnings detected in strict mode

This script performs read-only diagnostics,
except when explicitly saving a report.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import platform
import shutil
import sys
import time
import urllib.error
import urllib.request

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("neuroworld.health")


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    SKIPPED = "skipped"


@dataclass
class HealthCheckConfig:
    minimum_python: tuple[int, int] = (3, 10)

    minimum_ram_gb: float = 4.0
    recommended_ram_gb: float = 16.0

    minimum_disk_free_gb: float = 2.0
    recommended_disk_free_gb: float = 10.0

    require_torch: bool = False
    require_cuda: bool = False

    check_api: bool = False
    api_health_url: str = (
        "http://127.0.0.1:8000/health"
    )
    api_timeout_seconds: float = 3.0

    data_directory: str = "data"
    model_directory: str = "models"
    artifact_directory: str = "artifacts"

    report_directory: str = "artifacts/reports"

    strict: bool = False
    verbose: bool = False


@dataclass
class HealthCheckResult:
    name: str
    category: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(
        default_factory=dict
    )
    duration_ms: float = 0.0


@dataclass
class SystemHealthReport:
    project: str
    timestamp: str
    overall_status: HealthStatus

    total_checks: int
    healthy_checks: int
    warnings: int
    critical_failures: int
    skipped_checks: int

    duration_seconds: float

    checks: list[HealthCheckResult] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class NeuroWorldHealthChecker:
    """Read-only health diagnostics for NeuroWorld AI."""

    REQUIRED_PACKAGES = {
        "numpy": "numpy",
        "pydantic": "pydantic",
    }

    OPTIONAL_PACKAGES = {
        "torch": "torch",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "matplotlib": "matplotlib",
        "pillow": "PIL",
        "requests": "requests",
        "psutil": "psutil",
        "pytest": "pytest",
        "scipy": "scipy",
        "pandas": "pandas",
    }

    PROJECT_MODULES = {
        "scenario": (
            "neuroworld.environment.scenario"
        ),
        "simulator": (
            "neuroworld.environment.simulator"
        ),
        "navigation": (
            "neuroworld.environment.navigation"
        ),
        "obstacle_avoidance": (
            "neuroworld.environment.obstacle_avoidance"
        ),
        "object_interaction": (
            "neuroworld.environment.object_interaction"
        ),
        "world_runner": (
            "neuroworld.environment.world_runner"
        ),
        "collision": (
            "neuroworld.physics.collision"
        ),
        "object_push": (
            "neuroworld.physics.object_push"
        ),
        "navigation_agent": (
            "neuroworld.agents.navigation_agent"
        ),
        "world_agent": (
            "neuroworld.agents.world_agent"
        ),
    }

    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
    ) -> None:
        self.config = config or HealthCheckConfig()

        self.results: list[
            HealthCheckResult
        ] = []

        self.started_at = 0.0

    def _record(
        self,
        name: str,
        category: str,
        status: HealthStatus,
        message: str,
        details: Optional[
            dict[str, Any]
        ] = None,
        duration_ms: float = 0.0,
    ) -> HealthCheckResult:
        result = HealthCheckResult(
            name=name,
            category=category,
            status=status,
            message=message,
            details=details or {},
            duration_ms=duration_ms,
        )

        self.results.append(result)

        if self.config.verbose:
            LOGGER.info(
                "[%s] %s: %s",
                status.value.upper(),
                name,
                message,
            )

        return result

    @staticmethod
    def _bytes_to_gb(
        value: int | float,
    ) -> float:
        return round(
            value / (1024**3),
            2,
        )

    @staticmethod
    def _package_exists(
        module_name: str,
    ) -> bool:
        try:
            return (
                importlib.util.find_spec(
                    module_name
                ) is not None
            )
        except (
            ImportError,
            ModuleNotFoundError,
            ValueError,
        ):
            return False

    def check_python(
        self,
    ) -> HealthCheckResult:
        started = time.perf_counter()

        current = (
            sys.version_info.major,
            sys.version_info.minor,
        )

        required = (
            self.config.minimum_python
        )

        status = (
            HealthStatus.HEALTHY
            if current >= required
            else HealthStatus.CRITICAL
        )

        return self._record(
            name="Python Version",
            category="runtime",
            status=status,
            message=(
                f"Python {platform.python_version()}"
            ),
            details={
                "executable": sys.executable,
                "minimum_version": (
                    ".".join(
                        map(str, required)
                    )
                ),
                "implementation": (
                    platform.python_implementation()
                ),
            },
            duration_ms=(
                time.perf_counter() - started
            ) * 1000,
        )

    def check_operating_system(
        self,
    ) -> HealthCheckResult:
        started = time.perf_counter()

        return self._record(
            name="Operating System",
            category="system",
            status=HealthStatus.HEALTHY,
            message=platform.platform(),
            details={
                "system": platform.system(),
                "release": platform.release(),
                "architecture": (
                    platform.machine()
                ),
                "processor": (
                    platform.processor()
                ),
            },
            duration_ms=(
                time.perf_counter() - started
            ) * 1000,
        )

    def check_cpu(
        self,
    ) -> HealthCheckResult:
        started = time.perf_counter()

        logical_cores = (
            os.cpu_count() or 1
        )

        details: dict[str, Any] = {
            "logical_cores": logical_cores,
        }

        if self._package_exists("psutil"):
            import psutil

            details["physical_cores"] = (
                psutil.cpu_count(
                    logical=False
                )
            )

            details["cpu_usage_percent"] = (
                psutil.cpu_percent(
                    interval=0.1
                )
            )

        status = (
            HealthStatus.HEALTHY
            if logical_cores >= 2
            else HealthStatus.WARNING
        )

        return self._record(
            name="CPU",
            category="hardware",
            status=status,
            message=(
                f"{logical_cores} logical CPU cores"
            ),
            details=details,
            duration_ms=(
                time.perf_counter() - started
            ) * 1000,
        )

    def check_memory(
        self,
    ) -> HealthCheckResult:
        started = time.perf_counter()

        if not self._package_exists("psutil"):
            return self._record(
                name="System Memory",
                category="hardware",
                status=HealthStatus.SKIPPED,
                message=(
                    "Install psutil to inspect RAM"
                ),
            )

        import psutil

        memory = psutil.virtual_memory()

        total_gb = self._bytes_to_gb(
            memory.total
        )

        available_gb = self._bytes_to_gb(
            memory.available
        )

        if total_gb < self.config.minimum_ram_gb:
            status = HealthStatus.CRITICAL

        elif (
            total_gb
            < self.config.recommended_ram_gb
        ):
            status = HealthStatus.WARNING

        else:
            status = HealthStatus.HEALTHY

        return self._record(
            name="System Memory",
            category="hardware",
            status=status,
            message=(
                f"{total_gb} GB RAM, "
                f"{available_gb} GB available"
            ),
            details={
                "total_gb": total_gb,
                "available_gb": available_gb,
                "usage_percent": memory.percent,
                "minimum_gb": (
                    self.config.minimum_ram_gb
                ),
                "recommended_gb": (
                    self.config.recommended_ram_gb
                ),
            },
            duration_ms=(
                time.perf_counter() - started
            ) * 1000,
        )

    def check_disk(
        self,
    ) -> HealthCheckResult:
        started = time.perf_counter()

        usage = shutil.disk_usage(
            PROJECT_ROOT
        )

        free_gb = self._bytes_to_gb(
            usage.free
        )

        total_gb = self._bytes_to_gb(
            usage.total
        )

        if (
            free_gb
            < self.config.minimum_disk_free_gb
        ):
            status = HealthStatus.CRITICAL

        elif (
            free_gb
            < self.config.recommended_disk_free_gb
        ):
            status = HealthStatus.WARNING

        else:
            status = HealthStatus.HEALTHY

        return self._record(
            name="Disk Space",
            category="storage",
            status=status,
            message=(
                f"{free_gb} GB free "
                f"of {total_gb} GB"
            ),
            details={
                "total_gb": total_gb,
                "free_gb": free_gb,
                "used_gb": (
                    self._bytes_to_gb(
                        usage.used
                    )
                ),
                "project_root": str(
                    PROJECT_ROOT
                ),
            },
            duration_ms=(
                time.perf_counter() - started
            ) * 1000,
        )

    def check_dependencies(
        self,
    ) -> list[HealthCheckResult]:
        results = []

        for name, module in (
            self.REQUIRED_PACKAGES.items()
        ):
            installed = self._package_exists(
                module
            )

            results.append(
                self._record(
                    name=f"Dependency: {name}",
                    category="dependencies",
                    status=(
                        HealthStatus.HEALTHY
                        if installed
                        else HealthStatus.CRITICAL
                    ),
                    message=(
                        "Installed"
                        if installed
                        else "Required package missing"
                    ),
                )
            )

        for name, module in (
            self.OPTIONAL_PACKAGES.items()
        ):
            installed = self._package_exists(
                module
            )

            required = (
                name == "torch"
                and self.config.require_torch
            )

            if installed:
                status = HealthStatus.HEALTHY
            elif required:
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.SKIPPED

            results.append(
                self._record(
                    name=f"Dependency: {name}",
                    category="dependencies",
                    status=status,
                    message=(
                        "Installed"
                        if installed
                        else (
                            "Required package missing"
                            if required
                            else "Optional package missing"
                        )
                    ),
                )
            )

        return results

    def check_pytorch(
        self,
    ) -> HealthCheckResult:
        started = time.perf_counter()

        if not self._package_exists("torch"):
            return self._record(
                name="PyTorch",
                category="machine_learning",
                status=(
                    HealthStatus.CRITICAL
                    if self.config.require_torch
                    else HealthStatus.SKIPPED
                ),
                message="PyTorch is not installed",
            )

        try:
            import torch

            cuda_available = (
                torch.cuda.is_available()
            )

            details: dict[str, Any] = {
                "torch_version": torch.__version__,
                "cuda_available": cuda_available,
                "cuda_build_version": (
                    torch.version.cuda
                ),
                "gpu_count": (
                    torch.cuda.device_count()
                    if cuda_available
                    else 0
                ),
            }

            if cuda_available:
                devices = []

                for index in range(
                    torch.cuda.device_count()
                ):
                    properties = (
                        torch.cuda.get_device_properties(
                            index
                        )
                    )

                    devices.append(
                        {
                            "index": index,
                            "name": properties.name,
                            "total_memory_gb": (
                                self._bytes_to_gb(
                                    properties.total_memory
                                )
                            ),
                            "compute_capability": (
                                f"{properties.major}."
                                f"{properties.minor}"
                            ),
                        }
                    )

                details["devices"] = devices

            if (
                self.config.require_cuda
                and not cuda_available
            ):
                status = HealthStatus.CRITICAL
                message = (
                    "CUDA GPU required but unavailable"
                )

            else:
                status = HealthStatus.HEALTHY
                message = (
                    f"PyTorch {torch.__version__}, "
                    f"CUDA available: {cuda_available}"
                )

            return self._record(
                name="PyTorch",
                category="machine_learning",
                status=status,
                message=message,
                details=details,
                duration_ms=(
                    time.perf_counter() - started
                ) * 1000,
            )

        except Exception as exc:
            return self._record(
                name="PyTorch",
                category="machine_learning",
                status=HealthStatus.CRITICAL,
                message=(
                    f"PyTorch initialization failed: {exc}"
                ),
            )

    def check_project_modules(
        self,
    ) -> list[HealthCheckResult]:
        """
        Check source file availability without
        executing project module imports.
        """

        results = []

        for name, module in (
            self.PROJECT_MODULES.items()
        ):
            path = (
                PROJECT_ROOT
                / (
                    module.replace(
                        ".", "/"
                    ) + ".py"
                )
            )

            exists = path.is_file()

            results.append(
                self._record(
                    name=f"Module: {name}",
                    category="project",
                    status=(
                        HealthStatus.HEALTHY
                        if exists
                        else HealthStatus.WARNING
                    ),
                    message=(
                        "Source file found"
                        if exists
                        else "Module not found"
                    ),
                    details={
                        "module": module,
                        "path": str(path),
                    },
                )
            )

        return results

    def check_project_directories(
        self,
    ) -> list[HealthCheckResult]:
        directories = {
            "data": self.config.data_directory,
            "models": self.config.model_directory,
            "artifacts": (
                self.config.artifact_directory
            ),
            "configs": "configs",
            "scripts": "scripts",
        }

        results = []

        for name, relative_path in (
            directories.items()
        ):
            path = (
                PROJECT_ROOT / relative_path
            )

            exists = path.is_dir()

            results.append(
                self._record(
                    name=f"Directory: {name}",
                    category="filesystem",
                    status=(
                        HealthStatus.HEALTHY
                        if exists
                        else HealthStatus.WARNING
                    ),
                    message=(
                        "Directory exists"
                        if exists
                        else "Directory missing"
                    ),
                    details={
                        "path": str(path)
                    },
                )
            )

        return results

    def check_datasets(
        self,
    ) -> HealthCheckResult:
        started = time.perf_counter()

        directory = (
            PROJECT_ROOT
            / self.config.data_directory
        )

        if not directory.exists():
            return self._record(
                name="Datasets",
                category="data",
                status=HealthStatus.WARNING,
                message="Data directory not found",
            )

        extensions = {
            ".npz",
            ".npy",
            ".csv",
            ".jsonl",
            ".parquet",
            ".h5",
            ".hdf5",
        }

        files = [
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in extensions
            )
        ]

        total_size = sum(
            path.stat().st_size
            for path in files
        )

        return self._record(
            name="Datasets",
            category="data",
            status=(
                HealthStatus.HEALTHY
                if files
                else HealthStatus.WARNING
            ),
            message=(
                f"{len(files)} dataset files found"
            ),
            details={
                "file_count": len(files),
                "total_size_gb": (
                    self._bytes_to_gb(
                        total_size
                    )
                ),
                "sample_files": [
                    str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    )
                    for path in files[:10]
                ],
            },
            duration_ms=(
                time.perf_counter() - started
            ) * 1000,
        )

    def check_model_checkpoints(
        self,
    ) -> HealthCheckResult:
        directory = (
            PROJECT_ROOT
            / self.config.model_directory
        )

        if not directory.exists():
            return self._record(
                name="Model Checkpoints",
                category="machine_learning",
                status=HealthStatus.WARNING,
                message="Model directory missing",
            )

        extensions = {
            ".pt",
            ".pth",
            ".ckpt",
            ".safetensors",
            ".onnx",
        }

        files = [
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in extensions
            )
        ]

        return self._record(
            name="Model Checkpoints",
            category="machine_learning",
            status=(
                HealthStatus.HEALTHY
                if files
                else HealthStatus.WARNING
            ),
            message=(
                f"{len(files)} model files found"
            ),
            details={
                "count": len(files),
                "files": [
                    str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    )
                    for path in files[:20]
                ],
            },
        )

    def check_configuration(
        self,
    ) -> HealthCheckResult:
        files = [
            "pyproject.toml",
            "requirements.txt",
            ".env",
            ".env.example",
            "configs/datasets.json",
        ]

        found = [
            name
            for name in files
            if (
                PROJECT_ROOT / name
            ).is_file()
        ]

        if not (
            "requirements.txt" in found
            or "pyproject.toml" in found
        ):
            status = HealthStatus.WARNING
            message = (
                "No Python dependency manifest found"
            )
        else:
            status = HealthStatus.HEALTHY
            message = (
                "Project dependency manifest found"
            )

        return self._record(
            name="Configuration",
            category="configuration",
            status=status,
            message=message,
            details={
                "found_files": found,
            },
        )

    def check_api(
        self,
    ) -> HealthCheckResult:
        started = time.perf_counter()

        url = self.config.api_health_url

        if not url.startswith(
            ("http://", "https://")
        ):
            return self._record(
                name="FastAPI Health",
                category="api",
                status=HealthStatus.CRITICAL,
                message="Invalid API URL scheme",
            )

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json"
                },
                method="GET",
            )

            with urllib.request.urlopen(
                request,
                timeout=(
                    self.config.api_timeout_seconds
                ),
            ) as response:
                status_code = response.status

                body = response.read(
                    4096
                ).decode(
                    "utf-8",
                    errors="replace",
                )

            healthy = (
                200 <= status_code < 300
            )

            return self._record(
                name="FastAPI Health",
                category="api",
                status=(
                    HealthStatus.HEALTHY
                    if healthy
                    else HealthStatus.WARNING
                ),
                message=(
                    f"HTTP {status_code}"
                ),
                details={
                    "url": url,
                    "response": body,
                },
                duration_ms=(
                    time.perf_counter() - started
                ) * 1000,
            )

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            return self._record(
                name="FastAPI Health",
                category="api",
                status=HealthStatus.WARNING,
                message=(
                    f"API unavailable: {exc}"
                ),
                details={
                    "url": url
                },
            )

    def run_all(
        self,
    ) -> SystemHealthReport:
        self.results.clear()

        self.started_at = (
            time.perf_counter()
        )

        checks = [
            self.check_python,
            self.check_operating_system,
            self.check_cpu,
            self.check_memory,
            self.check_disk,
            self.check_dependencies,
            self.check_pytorch,
            self.check_project_modules,
            self.check_project_directories,
            self.check_datasets,
            self.check_model_checkpoints,
            self.check_configuration,
        ]

        if self.config.check_api:
            checks.append(
                self.check_api
            )

        for check in checks:
            try:
                check()

            except Exception as exc:
                LOGGER.exception(
                    "Health check failed: %s",
                    check.__name__,
                )

                self._record(
                    name=check.__name__,
                    category="internal",
                    status=HealthStatus.CRITICAL,
                    message=str(exc),
                )

        counts = {
            status: sum(
                result.status == status
                for result in self.results
            )
            for status in HealthStatus
        }

        if counts[HealthStatus.CRITICAL]:
            overall = HealthStatus.CRITICAL

        elif counts[HealthStatus.WARNING]:
            overall = HealthStatus.WARNING

        else:
            overall = HealthStatus.HEALTHY

        return SystemHealthReport(
            project="NeuroWorld AI",
            timestamp=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            overall_status=overall,
            total_checks=len(
                self.results
            ),
            healthy_checks=counts[
                HealthStatus.HEALTHY
            ],
            warnings=counts[
                HealthStatus.WARNING
            ],
            critical_failures=counts[
                HealthStatus.CRITICAL
            ],
            skipped_checks=counts[
                HealthStatus.SKIPPED
            ],
            duration_seconds=round(
                time.perf_counter()
                - self.started_at,
                3,
            ),
            checks=list(
                self.results
            ),
            metadata={
                "project_root": str(
                    PROJECT_ROOT
                ),
                "python_executable": (
                    sys.executable
                ),
                "platform": (
                    platform.platform()
                ),
            },
        )

    def save_report(
        self,
        report: SystemHealthReport,
    ) -> Path:
        directory = (
            PROJECT_ROOT
            / self.config.report_directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d_%H%M%S"
        )

        path = directory / (
            f"health_check_{timestamp}.json"
        )

        path.write_text(
            json.dumps(
                asdict(report),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return path

    @staticmethod
    def print_report(
        report: SystemHealthReport,
        verbose: bool = False,
    ) -> None:
        print()
        print("=" * 62)
        print("             NEUROWORLD AI HEALTH CHECK")
        print("=" * 62)

        print(
            "Overall status:",
            report.overall_status.value.upper(),
        )

        print(
            "Checks:",
            report.total_checks,
        )

        print(
            "Healthy:",
            report.healthy_checks,
        )

        print(
            "Warnings:",
            report.warnings,
        )

        print(
            "Critical:",
            report.critical_failures,
        )

        print(
            "Skipped:",
            report.skipped_checks,
        )

        print(
            "Duration:",
            f"{report.duration_seconds:.3f}s",
        )

        print("-" * 62)

        for result in report.checks:
            if (
                not verbose
                and result.status
                == HealthStatus.HEALTHY
            ):
                continue

            print(
                f"[{result.status.value.upper():8}] "
                f"{result.name}: "
                f"{result.message}"
            )

        print("=" * 62)

    def exit_code(
        self,
        report: SystemHealthReport,
    ) -> int:
        if report.critical_failures:
            return 1

        if (
            self.config.strict
            and report.warnings
        ):
            return 2

        return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NeuroWorld AI system health diagnostics"
        )
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show every health check",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report",
    )

    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save health report to disk",
    )

    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Check FastAPI health endpoint",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/health",
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
    )

    parser.add_argument(
        "--require-torch",
        action="store_true",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.WARNING
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    config = HealthCheckConfig(
        verbose=args.verbose,
        check_api=args.check_api,
        api_health_url=args.api_url,
        require_cuda=args.require_cuda,
        require_torch=(
            args.require_torch
            or args.require_cuda
        ),
        strict=args.strict,
    )

    checker = NeuroWorldHealthChecker(
        config
    )

    report = checker.run_all()

    if args.json:
        print(
            json.dumps(
                asdict(report),
                indent=2,
                default=str,
            )
        )

    else:
        checker.print_report(
            report,
            verbose=args.verbose,
        )

    if args.save_report:
        path = checker.save_report(
            report
        )

        print(
            "Report saved:",
            path,
            file=sys.stderr
            if args.json
            else sys.stdout,
        )

    return checker.exit_code(
        report
    )


if __name__ == "__main__":
    raise SystemExit(main())
```
