\
"""
NeuroWorld AI
File: training/datasets/dataset_validator.py

Dataset validation and quality reporting.

Features:
- NPZ dataset validation
- State/action/next-state shape checks
- NaN and Infinity detection
- Reward and terminal flag validation
- Episode continuity checks
- State-transition continuity checks
- Video tensor validation
- PyTorch Dataset sample validation
- Configurable limits and tolerances
- JSON quality reports

Dependencies:
    numpy
    torch
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class DatasetValidationConfig:
    state_dim: int | None = 128
    action_dim: int | None = 32

    min_samples: int = 1

    check_finite: bool = True
    check_episode_continuity: bool = True
    check_transition_continuity: bool = True

    transition_tolerance: float = 1e-4

    check_action_bounds: bool = False
    action_min: float = -1.0
    action_max: float = 1.0

    check_reward_bounds: bool = False
    reward_min: float = -1000.0
    reward_max: float = 1000.0

    video_channels: int = 3
    video_min_frames: int = 2

    check_video_range: bool = True
    video_min_value: float = 0.0
    video_max_value: float = 1.0

    max_sample_checks: int = 100

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError(
                "min_samples must be >= 1"
            )

        if self.transition_tolerance < 0:
            raise ValueError(
                "transition_tolerance must be >= 0"
            )

        if self.max_sample_checks < 1:
            raise ValueError(
                "max_sample_checks must be >= 1"
            )

        if self.action_min > self.action_max:
            raise ValueError(
                "Invalid action bounds"
            )

        if self.reward_min > self.reward_max:
            raise ValueError(
                "Invalid reward bounds"
            )


@dataclass
class ValidationReport:
    dataset_name: str
    valid: bool = True

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    statistics: dict[str, Any] = field(
        default_factory=dict
    )

    def add_error(
        self,
        message: str,
    ) -> None:
        self.valid = False
        self.errors.append(message)

    def add_warning(
        self,
        message: str,
    ) -> None:
        self.warnings.append(message)

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)

    def save_json(
        self,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
            )

        return destination


class DatasetValidator:
    """
    Validates NeuroWorld AI training data.

    NPZ transition schema:

    Required:
        states       [N, state_dim]
        actions      [N, action_dim]
        next_states  [N, state_dim]

    Optional:
        rewards      [N]
        terminated   [N]
        truncated    [N]
        episode_ids  [N]

    Episode transitions are assumed to be
    stored in chronological order.
    """

    REQUIRED_KEYS = (
        "states",
        "actions",
        "next_states",
    )

    OPTIONAL_KEYS = (
        "rewards",
        "terminated",
        "truncated",
        "episode_ids",
    )

    def __init__(
        self,
        config: DatasetValidationConfig | None = None,
    ) -> None:
        self.config = (
            config or DatasetValidationConfig()
        )

    @staticmethod
    def _array(
        value: Any,
    ) -> np.ndarray:
        if isinstance(value, Tensor):
            return value.detach().cpu().numpy()

        return np.asarray(value)

    @staticmethod
    def _is_numeric(
        array: np.ndarray,
    ) -> bool:
        return (
            np.issubdtype(
                array.dtype,
                np.number,
            )
            or np.issubdtype(
                array.dtype,
                np.bool_,
            )
        )

    def _validate_numeric(
        self,
        name: str,
        array: np.ndarray,
        report: ValidationReport,
    ) -> bool:
        if not self._is_numeric(array):
            report.add_error(
                f"{name} must contain numeric values"
            )
            return False

        if self.config.check_finite:
            if not np.isfinite(array).all():
                count = int(
                    np.count_nonzero(
                        ~np.isfinite(array)
                    )
                )

                report.add_error(
                    f"{name} contains {count} "
                    "non-finite values"
                )

                return False

        return True

    def _validate_matrix(
        self,
        name: str,
        array: np.ndarray,
        expected_dim: int | None,
        report: ValidationReport,
    ) -> bool:
        if array.ndim != 2:
            report.add_error(
                f"{name} must have shape [N, D], "
                f"received {array.shape}"
            )
            return False

        if (
            expected_dim is not None
            and array.shape[1] != expected_dim
        ):
            report.add_error(
                f"{name} expected feature dimension "
                f"{expected_dim}, received "
                f"{array.shape[1]}"
            )

            return False

        return self._validate_numeric(
            name,
            array,
            report,
        )

    def validate_transitions(
        self,
        data: Mapping[str, Any],
        *,
        dataset_name: str = "transitions",
    ) -> ValidationReport:
        report = ValidationReport(
            dataset_name=dataset_name
        )

        missing = [
            key
            for key in self.REQUIRED_KEYS
            if key not in data
        ]

        if missing:
            report.add_error(
                f"Missing required keys: {missing}"
            )
            return report

        try:
            states = self._array(
                data["states"]
            )
            actions = self._array(
                data["actions"]
            )
            next_states = self._array(
                data["next_states"]
            )
        except Exception as exc:
            report.add_error(
                f"Unable to read arrays: {exc}"
            )
            return report

        n = len(states) if states.ndim else 0

        report.statistics["samples"] = n

        if n < self.config.min_samples:
            report.add_error(
                f"Expected at least "
                f"{self.config.min_samples} samples"
            )

        shapes_valid = True

        for name, array, dim in (
            (
                "states",
                states,
                self.config.state_dim,
            ),
            (
                "actions",
                actions,
                self.config.action_dim,
            ),
            (
                "next_states",
                next_states,
                self.config.state_dim,
            ),
        ):
            if not self._validate_matrix(
                name,
                array,
                dim,
                report,
            ):
                shapes_valid = False

        if not shapes_valid:
            return report

        if (
            len(actions) != n
            or len(next_states) != n
        ):
            report.add_error(
                "Transition array lengths differ"
            )
            return report

        if (
            states.shape[1]
            != next_states.shape[1]
        ):
            report.add_error(
                "states and next_states "
                "feature dimensions differ"
            )
            return report

        report.statistics.update({
            "state_dim": int(
                states.shape[1]
            ),
            "action_dim": int(
                actions.shape[1]
            ),
            "state_mean": float(
                states.mean()
            ) if n else None,
            "state_std": float(
                states.std()
            ) if n else None,
            "action_mean": float(
                actions.mean()
            ) if n else None,
            "action_std": float(
                actions.std()
            ) if n else None,
        })

        # Optional arrays
        optional: dict[
            str,
            np.ndarray,
        ] = {}

        for name in self.OPTIONAL_KEYS:
            if name not in data:
                continue

            array = self._array(
                data[name]
            )

            if (
                array.ndim != 1
                or len(array) != n
            ):
                report.add_error(
                    f"{name} must have shape [{n}]"
                )
                continue

            if not self._validate_numeric(
                name,
                array,
                report,
            ):
                continue

            optional[name] = array

        if self.config.check_action_bounds:
            violations = np.count_nonzero(
                (actions < self.config.action_min)
                | (actions > self.config.action_max)
            )

            if violations:
                report.add_error(
                    f"Actions contain {violations} "
                    "out-of-bounds values"
                )

        rewards = optional.get(
            "rewards"
        )

        if (
            rewards is not None
            and self.config.check_reward_bounds
        ):
            violations = np.count_nonzero(
                (rewards < self.config.reward_min)
                | (rewards > self.config.reward_max)
            )

            if violations:
                report.add_error(
                    f"Rewards contain {violations} "
                    "out-of-bounds values"
                )

        for flag_name in (
            "terminated",
            "truncated",
        ):
            flags = optional.get(
                flag_name
            )

            if flags is not None:
                if not np.isin(
                    flags,
                    [0, 1],
                ).all():
                    report.add_error(
                        f"{flag_name} must be binary"
                    )

        episode_ids = optional.get(
            "episode_ids"
        )

        if episode_ids is not None:
            report.statistics["episodes"] = int(
                np.unique(
                    episode_ids
                ).size
            )

            if (
                self.config.check_episode_continuity
                and n > 1
            ):
                self._check_episode_continuity(
                    episode_ids,
                    report,
                )

        if (
            self.config.check_transition_continuity
            and n > 1
        ):
            if episode_ids is None:
                report.add_warning(
                    "Transition continuity skipped: "
                    "episode_ids are unavailable"
                )
            else:
                self._check_transition_continuity(
                    states,
                    next_states,
                    episode_ids,
                    optional.get("terminated"),
                    optional.get("truncated"),
                    report,
                )

        return report

    def _check_episode_continuity(
        self,
        episode_ids: np.ndarray,
        report: ValidationReport,
    ) -> None:
        """
        Check whether episode IDs appear in
        contiguous blocks.
        """
        seen: set[Any] = set()
        previous = None

        for episode in episode_ids:
            episode = (
                episode.item()
                if isinstance(
                    episode,
                    np.generic,
                )
                else episode
            )

            if episode != previous:
                if episode in seen:
                    report.add_error(
                        "Episode IDs are not "
                        "stored contiguously"
                    )
                    return

                seen.add(episode)
                previous = episode

    def _check_transition_continuity(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        episode_ids: np.ndarray,
        terminated: np.ndarray | None,
        truncated: np.ndarray | None,
        report: ValidationReport,
    ) -> None:
        """
        Check s[t+1] approximately equals
        next_states[t] within each episode.

        Assumes consecutive transitions have
        no missing timesteps.
        """
        same_episode = (
            episode_ids[:-1]
            == episode_ids[1:]
        )

        if terminated is not None:
            same_episode &= (
                ~terminated[:-1].astype(bool)
            )

        if truncated is not None:
            same_episode &= (
                ~truncated[:-1].astype(bool)
            )

        if not same_episode.any():
            return

        differences = np.max(
            np.abs(
                next_states[:-1]
                - states[1:]
            ),
            axis=1,
        )

        violations = (
            same_episode
            & (
                differences
                > self.config.transition_tolerance
            )
        )

        count = int(
            np.count_nonzero(violations)
        )

        report.statistics[
            "transition_mismatches"
        ] = count

        if count:
            report.add_warning(
                f"{count} consecutive transitions "
                "do not match within tolerance"
            )

    def validate_npz(
        self,
        path: str | Path,
    ) -> ValidationReport:
        source = Path(path)

        report = ValidationReport(
            dataset_name=str(source)
        )

        if not source.is_file():
            report.add_error(
                f"File not found: {source}"
            )
            return report

        try:
            with np.load(
                source,
                allow_pickle=False,
            ) as data:
                arrays = {
                    key: data[key]
                    for key in data.files
                }

            return self.validate_transitions(
                arrays,
                dataset_name=str(source),
            )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            report.add_error(
                f"Failed to load NPZ: {exc}"
            )

            return report

    def validate_video(
        self,
        video: Tensor | np.ndarray,
        *,
        dataset_name: str = "video",
    ) -> ValidationReport:
        """
        Expected video:
            [T, C, H, W]

        Floating-point values in [0, 1].
        """
        report = ValidationReport(
            dataset_name=dataset_name
        )

        array = self._array(video)

        if array.ndim != 4:
            report.add_error(
                "Video must have shape "
                "[T, C, H, W]"
            )
            return report

        frames, channels, height, width = (
            array.shape
        )

        report.statistics.update({
            "frames": int(frames),
            "channels": int(channels),
            "height": int(height),
            "width": int(width),
        })

        if frames < self.config.video_min_frames:
            report.add_error(
                "Video contains too few frames"
            )

        if channels != self.config.video_channels:
            report.add_error(
                f"Expected {self.config.video_channels} "
                f"channels, received {channels}"
            )

        if height < 1 or width < 1:
            report.add_error(
                "Invalid video resolution"
            )

        if not self._validate_numeric(
            "video",
            array,
            report,
        ):
            return report

        if self.config.check_video_range:
            if (
                np.any(
                    array < self.config.video_min_value
                )
                or np.any(
                    array > self.config.video_max_value
                )
            ):
                report.add_error(
                    "Video pixel values are outside "
                    "the configured range"
                )

        return report

    def validate_dataset(
        self,
        dataset: Dataset,
        *,
        dataset_name: str = "pytorch_dataset",
    ) -> ValidationReport:
        """
        Inspect PyTorch Dataset samples.

        This validates a bounded sample subset,
        not every element unless the dataset is
        small enough.
        """
        report = ValidationReport(
            dataset_name=dataset_name
        )

        total = len(dataset)

        report.statistics[
            "dataset_length"
        ] = total

        if total < self.config.min_samples:
            report.add_error(
                "Dataset contains too few samples"
            )
            return report

        count = min(
            total,
            self.config.max_sample_checks,
        )

        indices = np.linspace(
            0,
            total - 1,
            count,
            dtype=int,
        )

        checked = 0

        for index in indices:
            try:
                sample = dataset[
                    int(index)
                ]
            except Exception as exc:
                report.add_error(
                    f"Sample {index} failed: {exc}"
                )
                continue

            checked += 1

            if not isinstance(
                sample,
                Mapping,
            ):
                report.add_warning(
                    f"Sample {index} is not "
                    "a dictionary"
                )
                continue

            for name, value in sample.items():
                if not isinstance(
                    value,
                    (Tensor, np.ndarray),
                ):
                    continue

                array = self._array(
                    value
                )

                self._validate_numeric(
                    f"sample[{index}].{name}",
                    array,
                    report,
                )

            if (
                "states" in sample
                and "actions" in sample
            ):
                states = self._array(
                    sample["states"]
                )

                actions = self._array(
                    sample["actions"]
                )

                if (
                    self.config.state_dim is not None
                    and states.shape[-1]
                    != self.config.state_dim
                ):
                    report.add_error(
                        f"Sample {index}: "
                        "invalid state dimension"
                    )

                if (
                    self.config.action_dim is not None
                    and actions.shape[-1]
                    != self.config.action_dim
                ):
                    report.add_error(
                        f"Sample {index}: "
                        "invalid action dimension"
                    )

        report.statistics[
            "samples_checked"
        ] = checked

        return report

    @staticmethod
    def print_report(
        report: ValidationReport,
    ) -> None:
        print(
            f"\nDataset: {report.dataset_name}"
        )

        print(
            "Status:",
            "PASS" if report.valid else "FAIL",
        )

        print(
            "Statistics:",
            json.dumps(
                report.statistics,
                indent=2,
            ),
        )

        for error in report.errors:
            print(
                "[ERROR]",
                error,
            )

        for warning in report.warnings:
            print(
                "[WARNING]",
                warning,
            )


def validate_dataset_file(
    path: str | Path,
    config: DatasetValidationConfig | None = None,
) -> ValidationReport:
    validator = DatasetValidator(
        config
    )

    return validator.validate_npz(
        path
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n = 100

    states = rng.normal(
        size=(n, 128)
    ).astype(np.float32)

    actions = rng.uniform(
        -1,
        1,
        size=(n, 32),
    ).astype(np.float32)

    next_states = np.roll(
        states,
        shift=-1,
        axis=0,
    )

    rewards = rng.normal(
        size=n
    ).astype(np.float32)

    episode_ids = np.repeat(
        np.arange(5),
        20,
    )

    terminated = np.zeros(
        n,
        dtype=bool,
    )

    terminated[19::20] = True

    path = Path(
        "data/processed/validation_demo.npz"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        episode_ids=episode_ids,
        terminated=terminated,
    )

    validator = DatasetValidator(
        DatasetValidationConfig(
            state_dim=128,
            action_dim=32,
            check_action_bounds=True,
        )
    )

    report = validator.validate_npz(
        path
    )

    validator.print_report(
        report
    )

    report.save_json(
        "reports/dataset_validation.json"
    )
