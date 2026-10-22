
"""
NeuroWorld AI
File: training/callbacks/early_stopping.py

Advanced early stopping callback for PyTorch.

Features:
- Monitor validation loss or any numeric metric
- Minimize or maximize metrics
- Configurable patience
- Absolute or relative minimum improvement
- Warm-up epochs
- Optional baseline threshold
- NaN/Inf handling
- Consecutive bad-epoch tracking
- Save and restore callback state
- Integration with CheckpointCallback
- No external dependencies beyond Python

Python 3.11+
"""

from __future__ import annotations

import logging
import math

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal, Mapping


logger = logging.getLogger(__name__)


class StopReason(str, Enum):
    NONE = "none"
    PATIENCE_EXCEEDED = "patience_exceeded"
    NON_FINITE_METRIC = "non_finite_metric"
    METRIC_MISSING = "metric_missing"
    MANUAL_STOP = "manual_stop"


@dataclass
class EarlyStoppingConfig:
    monitor: str = "val_loss"

    mode: Literal["min", "max"] = "min"

    patience: int = 10

    min_delta: float = 0.0

    min_delta_mode: Literal[
        "absolute", "relative"
    ] = "absolute"

    warmup_epochs: int = 0

    baseline: float | None = None

    stop_on_nan: bool = True

    strict: bool = True

    verbose: bool = True

    restore_best_weights: bool = False

    def __post_init__(self) -> None:
        if not self.monitor.strip():
            raise ValueError(
                "monitor cannot be empty"
            )

        if self.mode not in {"min", "max"}:
            raise ValueError(
                "mode must be min or max"
            )

        if self.patience < 0:
            raise ValueError(
                "patience must be nonnegative"
            )

        if self.min_delta < 0:
            raise ValueError(
                "min_delta must be nonnegative"
            )

        if not math.isfinite(self.min_delta):
            raise ValueError(
                "min_delta must be finite"
            )

        if self.min_delta_mode not in {
            "absolute",
            "relative",
        }:
            raise ValueError(
                "Invalid min_delta_mode"
            )

        if self.warmup_epochs < 0:
            raise ValueError(
                "warmup_epochs must be nonnegative"
            )

        if (
            self.baseline is not None
            and not math.isfinite(self.baseline)
        ):
            raise ValueError(
                "baseline must be finite"
            )


@dataclass
class EarlyStoppingResult:
    epoch: int

    monitored_metric: str

    current_score: float | None

    best_score: float | None

    best_epoch: int | None

    bad_epochs: int

    patience: int

    improved: bool

    should_stop: bool

    reason: StopReason

    warmup_active: bool


class EarlyStopping:
    """
    Tracks a monitored metric and decides when
    to stop training.

    Epoch numbering is zero-based.

    Example:
        stopper = EarlyStopping(
            EarlyStoppingConfig(
                monitor="val_loss",
                patience=8,
                min_delta=0.001,
            )
        )

        result = stopper.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.42},
        )

        if result.should_stop:
            break

    Notes:
        restore_best_weights is configuration
        metadata only. Model weights are restored
        through CheckpointCallback.load_best().
    """

    STATE_VERSION = 1

    def __init__(
        self,
        config: EarlyStoppingConfig | None = None,
    ) -> None:
        self.config = (
            config or EarlyStoppingConfig()
        )

        self.reset()

    def reset(self) -> None:
        self.best_score: float | None = None

        self.best_epoch: int | None = None

        self.bad_epochs = 0

        self.last_epoch: int | None = None

        self.stopped_epoch: int | None = None

        self.should_stop = False

        self.stop_reason = StopReason.NONE

        self.history: list[dict[str, Any]] = []

    def _improvement_threshold(
        self,
        reference: float,
    ) -> float:
        if (
            self.config.min_delta_mode
            == "absolute"
        ):
            return self.config.min_delta

        return (
            abs(reference)
            * self.config.min_delta
        )

    def _is_improvement(
        self,
        score: float,
    ) -> bool:
        if self.best_score is None:
            if self.config.baseline is None:
                return True

            reference = self.config.baseline

        else:
            reference = self.best_score

        delta = self._improvement_threshold(
            reference
        )

        if self.config.mode == "min":
            return score < reference - delta

        return score > reference + delta

    def _record(
        self,
        result: EarlyStoppingResult,
    ) -> None:
        self.history.append(
            {
                "epoch": result.epoch,
                "score": result.current_score,
                "best_score": result.best_score,
                "best_epoch": result.best_epoch,
                "bad_epochs": result.bad_epochs,
                "improved": result.improved,
                "should_stop": result.should_stop,
                "reason": result.reason.value,
            }
        )

    def _make_result(
        self,
        *,
        epoch: int,
        score: float | None,
        improved: bool,
        warmup: bool,
    ) -> EarlyStoppingResult:
        result = EarlyStoppingResult(
            epoch=epoch,
            monitored_metric=self.config.monitor,
            current_score=score,
            best_score=self.best_score,
            best_epoch=self.best_epoch,
            bad_epochs=self.bad_epochs,
            patience=self.config.patience,
            improved=improved,
            should_stop=self.should_stop,
            reason=self.stop_reason,
            warmup_active=warmup,
        )

        self._record(result)

        return result

    def on_epoch_end(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, float],
    ) -> EarlyStoppingResult:
        """
        Evaluate stopping conditions after
        each completed validation epoch.

        Patience is the number of consecutive
        non-improving epochs allowed.

        Example:
            patience=3 means stop after
            3 consecutive non-improving epochs.
        """
        if epoch < 0:
            raise ValueError(
                "epoch must be nonnegative"
            )

        if (
            self.last_epoch is not None
            and epoch <= self.last_epoch
        ):
            raise ValueError(
                "Epoch must increase monotonically"
            )

        if self.should_stop:
            raise RuntimeError(
                "Early stopping already triggered. "
                "Call reset() before a new run."
            )

        self.last_epoch = epoch

        warmup = (
            epoch < self.config.warmup_epochs
        )

        if self.config.monitor not in metrics:
            if self.config.strict:
                raise KeyError(
                    f"Missing monitored metric: "
                    f"{self.config.monitor}"
                )

            if self.config.verbose:
                logger.warning(
                    "Metric '%s' missing at epoch %d",
                    self.config.monitor,
                    epoch,
                )

            return self._make_result(
                epoch=epoch,
                score=None,
                improved=False,
                warmup=warmup,
            )

        raw_score = metrics[
            self.config.monitor
        ]

        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Monitored metric must be numeric"
            ) from exc

        if not math.isfinite(score):
            if self.config.stop_on_nan:
                self.should_stop = True

                self.stopped_epoch = epoch

                self.stop_reason = (
                    StopReason.NON_FINITE_METRIC
                )

                if self.config.verbose:
                    logger.error(
                        "Training stopped at epoch %d: "
                        "non-finite %s=%s",
                        epoch,
                        self.config.monitor,
                        score,
                    )

            return self._make_result(
                epoch=epoch,
                score=score,
                improved=False,
                warmup=warmup,
            )

        improved = self._is_improvement(
            score
        )

        if improved:
            previous = self.best_score

            self.best_score = score

            self.best_epoch = epoch

            self.bad_epochs = 0

            if self.config.verbose:
                logger.info(
                    "Epoch %d: %s improved "
                    "from %s to %.6f",
                    epoch + 1,
                    self.config.monitor,
                    previous,
                    score,
                )

        elif not warmup:
            self.bad_epochs += 1

            if self.config.verbose:
                logger.info(
                    "Epoch %d: no improvement "
                    "(%d/%d bad epochs)",
                    epoch + 1,
                    self.bad_epochs,
                    self.config.patience,
                )

        if (
            not warmup
            and not improved
            and self.bad_epochs
            >= max(1, self.config.patience)
        ):
            self.should_stop = True

            self.stopped_epoch = epoch

            self.stop_reason = (
                StopReason.PATIENCE_EXCEEDED
            )

            if self.config.verbose:
                logger.warning(
                    "Early stopping at epoch %d. "
                    "Best %s=%.6f at epoch %s",
                    epoch + 1,
                    self.config.monitor,
                    self.best_score
                    if self.best_score is not None
                    else float("nan"),
                    (
                        self.best_epoch + 1
                        if self.best_epoch is not None
                        else "N/A"
                    ),
                )

        return self._make_result(
            epoch=epoch,
            score=score,
            improved=improved,
            warmup=warmup,
        )

    def manual_stop(self) -> None:
        self.should_stop = True

        self.stopped_epoch = self.last_epoch

        self.stop_reason = (
            StopReason.MANUAL_STOP
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "config": asdict(self.config),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "last_epoch": self.last_epoch,
            "stopped_epoch": self.stopped_epoch,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason.value,
            "history": list(self.history),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        strict_config: bool = True,
    ) -> None:
        version = state.get(
            "state_version"
        )

        if version != self.STATE_VERSION:
            raise ValueError(
                "Unsupported early stopping "
                "state version"
            )

        if (
            strict_config
            and state.get("config")
            != asdict(self.config)
        ):
            raise ValueError(
                "Early stopping configuration "
                "does not match checkpoint"
            )

        self.best_score = state.get(
            "best_score"
        )

        self.best_epoch = state.get(
            "best_epoch"
        )

        self.bad_epochs = int(
            state.get("bad_epochs", 0)
        )

        self.last_epoch = state.get(
            "last_epoch"
        )

        self.stopped_epoch = state.get(
            "stopped_epoch"
        )

        self.should_stop = bool(
            state.get("should_stop", False)
        )

        self.stop_reason = StopReason(
            state.get(
                "stop_reason",
                StopReason.NONE.value,
            )
        )

        self.history = list(
            state.get("history", [])
        )

    def summary(self) -> dict[str, Any]:
        return {
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "patience": self.config.patience,
            "best_score": self.best_score,
            "best_epoch": (
                self.best_epoch + 1
                if self.best_epoch is not None
                else None
            ),
            "bad_epochs": self.bad_epochs,
            "should_stop": self.should_stop,
            "stopped_epoch": (
                self.stopped_epoch + 1
                if self.stopped_epoch is not None
                else None
            ),
            "stop_reason": (
                self.stop_reason.value
            ),
            "evaluated_epochs": len(
                self.history
            ),
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )

    stopper = EarlyStopping(
        EarlyStoppingConfig(
            monitor="val_loss",
            mode="min",
            patience=3,
            min_delta=0.001,
            warmup_epochs=2,
        )
    )

    validation_losses = [
        0.82,
        0.65,
        0.52,
        0.43,
        0.39,
        0.391,
        0.392,
        0.395,
        0.398,
    ]

    for epoch, loss in enumerate(
        validation_losses
    ):
        result = stopper.on_epoch_end(
            epoch=epoch,
            metrics={"val_loss": loss},
        )

        print(
            f"Epoch={epoch + 1} "
            f"Loss={loss:.4f} "
            f"Bad={result.bad_epochs} "
            f"Stop={result.should_stop}"
        )

        if result.should_stop:
            break

    print(stopper.summary())
