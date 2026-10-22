
"""
NeuroWorld AI
File: neuroworld/prediction/uncertainty.py

Predictive Uncertainty & Confidence Estimation Engine.

Features:
    1. Epistemic uncertainty estimation
    2. Aleatoric uncertainty estimation
    3. Total predictive uncertainty
    4. Deep ensemble aggregation
    5. Monte Carlo dropout aggregation
    6. Gaussian prediction intervals
    7. Empirical uncertainty calibration
    8. Uncertainty-aware planning penalties
    9. Future rollout uncertainty analysis
   10. Per-horizon uncertainty summaries
   11. Risk classification
   12. NumPy and PyTorch support
   13. Input validation
   14. Demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.prediction.uncertainty

Conventions:
    ensemble_means: [M, B, ...]
    ensemble_variances: [M, B, ...]
    MC predictions: [S, B, ...]

    M = ensemble members
    S = Monte Carlo samples
    B = batch size

All returned uncertainty values are standard
deviations unless explicitly named variance.

Important:
    Statistical uncertainty is not a guarantee
    of safety or future prediction accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class UncertaintyError(Exception):
    """Base uncertainty estimation exception."""


class UncertaintyValidationError(UncertaintyError):
    """Invalid uncertainty inputs."""


# ============================================================
# ENUMERATIONS
# ============================================================

class UncertaintyType(str, Enum):
    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"
    TOTAL = "total"


class PredictionRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class UncertaintyConfig:
    epsilon: float = 1e-8

    confidence_level: float = 0.95

    moderate_threshold: float = 0.15
    high_threshold: float = 0.35
    critical_threshold: float = 0.65

    uncertainty_penalty_weight: float = 0.1

    max_acceptable_uncertainty: float = 0.5

    mc_samples: int = 20

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        if not 0 < self.confidence_level < 1:
            raise ValueError(
                "confidence_level must be in (0,1)"
            )

        if not (
            0
            <= self.moderate_threshold
            < self.high_threshold
            < self.critical_threshold
        ):
            raise ValueError(
                "Invalid risk thresholds"
            )

        if self.uncertainty_penalty_weight < 0:
            raise ValueError(
                "Penalty weight must be nonnegative"
            )

        if self.max_acceptable_uncertainty < 0:
            raise ValueError(
                "Maximum acceptable uncertainty "
                "must be nonnegative"
            )

        if self.mc_samples <= 0:
            raise ValueError(
                "mc_samples must be positive"
            )


# ============================================================
# TENSOR UTILITIES
# ============================================================

def as_tensor(
    value: Tensor | np.ndarray | list,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    tensor = torch.as_tensor(
        value,
        dtype=torch.float32,
        device=device,
    )

    if tensor.numel() == 0:
        raise UncertaintyValidationError(
            "Input cannot be empty"
        )

    if not torch.isfinite(tensor).all():
        raise UncertaintyValidationError(
            "Input contains non-finite values"
        )

    return tensor


def validate_matching_shapes(
    first: Tensor,
    second: Tensor,
) -> None:
    if first.shape != second.shape:
        raise UncertaintyValidationError(
            f"Shape mismatch: "
            f"{tuple(first.shape)} vs "
            f"{tuple(second.shape)}"
        )


def validate_nonnegative(
    values: Tensor,
    name: str,
) -> None:
    if (values < 0).any():
        raise UncertaintyValidationError(
            f"{name} must be nonnegative"
        )


# ============================================================
# OUTPUT
# ============================================================

@dataclass
class UncertaintyEstimate:
    mean: Tensor

    epistemic_variance: Tensor
    aleatoric_variance: Tensor
    total_variance: Tensor

    epistemic_std: Tensor
    aleatoric_std: Tensor
    total_std: Tensor

    def detach(self) -> "UncertaintyEstimate":
        return UncertaintyEstimate(
            **{
                name: value.detach()
                for name, value in vars(self).items()
            }
        )

    def to_numpy(
        self,
    ) -> dict[str, np.ndarray]:
        return {
            name: (
                value.detach()
                .cpu()
                .numpy()
            )
            for name, value in vars(self).items()
        }

    def summary(self) -> dict[str, float]:
        return {
            "mean_prediction": float(
                self.mean.mean().item()
            ),
            "mean_epistemic_std": float(
                self.epistemic_std.mean().item()
            ),
            "mean_aleatoric_std": float(
                self.aleatoric_std.mean().item()
            ),
            "mean_total_std": float(
                self.total_std.mean().item()
            ),
            "max_total_std": float(
                self.total_std.max().item()
            ),
        }


# ============================================================
# ENSEMBLE UNCERTAINTY
# ============================================================

def estimate_ensemble_uncertainty(
    ensemble_means: Tensor | np.ndarray,
    ensemble_variances: (
        Tensor | np.ndarray | None
    ) = None,
) -> UncertaintyEstimate:
    """
    Estimate uncertainty from ensemble predictions.

    ensemble_means:
        [M, B, ...]

    ensemble_variances:
        [M, B, ...] optional

    Epistemic variance:
        Var_m(E[Y | model_m])

    Aleatoric variance:
        Mean_m(Var[Y | model_m])

    Total variance:
        Epistemic + aleatoric

    Uses population variance (unbiased=False).
    """

    means = as_tensor(
        ensemble_means
    )

    if means.ndim < 2:
        raise UncertaintyValidationError(
            "Expected ensemble predictions "
            "with shape [M,B,...]"
        )

    mean_prediction = means.mean(
        dim=0
    )

    epistemic_variance = means.var(
        dim=0,
        unbiased=False,
    )

    if ensemble_variances is None:
        aleatoric_variance = (
            torch.zeros_like(
                mean_prediction
            )
        )

    else:
        variances = as_tensor(
            ensemble_variances,
            device=means.device,
        )

        validate_matching_shapes(
            means,
            variances,
        )

        validate_nonnegative(
            variances,
            "Ensemble variances",
        )

        aleatoric_variance = (
            variances.mean(
                dim=0
            )
        )

    total_variance = (
        epistemic_variance
        + aleatoric_variance
    )

    return UncertaintyEstimate(
        mean=mean_prediction,
        epistemic_variance=(
            epistemic_variance
        ),
        aleatoric_variance=(
            aleatoric_variance
        ),
        total_variance=total_variance,
        epistemic_std=(
            epistemic_variance
            .clamp_min(0)
            .sqrt()
        ),
        aleatoric_std=(
            aleatoric_variance
            .clamp_min(0)
            .sqrt()
        ),
        total_std=(
            total_variance
            .clamp_min(0)
            .sqrt()
        ),
    )


# ============================================================
# MONTE CARLO UNCERTAINTY
# ============================================================

def estimate_mc_uncertainty(
    predictions: Tensor | np.ndarray,
) -> UncertaintyEstimate:
    """
    Aggregate stochastic predictions.

    Input:
        [S,B,...]

    The observed spread combines whatever
    randomness exists in the samples.

    It is not automatically a pure estimate
    of epistemic uncertainty.
    """

    samples = as_tensor(
        predictions
    )

    if samples.ndim < 2:
        raise UncertaintyValidationError(
            "Expected MC samples [S,B,...]"
        )

    mean = samples.mean(
        dim=0
    )

    variance = samples.var(
        dim=0,
        unbiased=False,
    )

    zeros = torch.zeros_like(
        variance
    )

    std = variance.sqrt()

    return UncertaintyEstimate(
        mean=mean,
        epistemic_variance=variance,
        aleatoric_variance=zeros,
        total_variance=variance,
        epistemic_std=std,
        aleatoric_std=zeros,
        total_std=std,
    )


# ============================================================
# GAUSSIAN UNCERTAINTY
# ============================================================

def estimate_gaussian_uncertainty(
    mean: Tensor | np.ndarray,
    log_variance: Tensor | np.ndarray,
) -> UncertaintyEstimate:
    """
    Convert Gaussian mean/log-variance
    into an uncertainty estimate.

    Without an ensemble, model uncertainty
    cannot be separately identified.
    """

    mean = as_tensor(
        mean
    )

    log_variance = as_tensor(
        log_variance,
        device=mean.device,
    )

    validate_matching_shapes(
        mean,
        log_variance,
    )

    variance = log_variance.exp()

    std = variance.sqrt()

    zeros = torch.zeros_like(
        variance
    )

    return UncertaintyEstimate(
        mean=mean,
        epistemic_variance=zeros,
        aleatoric_variance=variance,
        total_variance=variance,
        epistemic_std=zeros,
        aleatoric_std=std,
        total_std=std,
    )


# ============================================================
# CONFIDENCE INTERVALS
# ============================================================

def gaussian_prediction_interval(
    mean: Tensor | np.ndarray,
    std: Tensor | np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """
    Construct symmetric Gaussian intervals.

    Example:
        confidence_level=0.95

        approximately mean +/- 1.96 * std

    Coverage depends on distributional
    assumptions and calibration.
    """

    if not 0 < confidence_level < 1:
        raise UncertaintyValidationError(
            "confidence_level must be in (0,1)"
        )

    mean = as_tensor(
        mean
    )

    std = as_tensor(
        std,
        device=mean.device,
    )

    validate_matching_shapes(
        mean,
        std,
    )

    validate_nonnegative(
        std,
        "Standard deviation",
    )

    standard_normal = torch.distributions.Normal(
        torch.tensor(
            0.0,
            device=mean.device,
        ),
        torch.tensor(
            1.0,
            device=mean.device,
        ),
    )

    quantile = (
        1.0 + confidence_level
    ) / 2.0

    z_score = standard_normal.icdf(
        torch.tensor(
            quantile,
            device=mean.device,
        )
    )

    lower = (
        mean - z_score * std
    )

    upper = (
        mean + z_score * std
    )

    return lower, upper


# ============================================================
# UNCERTAINTY RISK CLASSIFICATION
# ============================================================

class UncertaintyRiskAssessor:
    """
    Classifies scalar uncertainty magnitudes.

    Thresholds depend on target scaling
    and must be calibrated to the domain.
    """

    def __init__(
        self,
        config: UncertaintyConfig | None = None,
    ) -> None:
        self.config = (
            config or UncertaintyConfig()
        )

    def classify(
        self,
        uncertainty: float,
    ) -> PredictionRiskLevel:
        if not math.isfinite(
            uncertainty
        ) or uncertainty < 0:
            raise UncertaintyValidationError(
                "Uncertainty must be finite "
                "and nonnegative"
            )

        if (
            uncertainty
            >= self.config.critical_threshold
        ):
            return PredictionRiskLevel.CRITICAL

        if (
            uncertainty
            >= self.config.high_threshold
        ):
            return PredictionRiskLevel.HIGH

        if (
            uncertainty
            >= self.config.moderate_threshold
        ):
            return PredictionRiskLevel.MODERATE

        return PredictionRiskLevel.LOW

    def assess(
        self,
        estimate: UncertaintyEstimate,
    ) -> dict[str, Any]:
        score = float(
            estimate.total_std.mean().item()
        )

        risk = self.classify(
            score
        )

        return {
            "uncertainty_score": score,
            "risk_level": risk.value,
            "acceptable": (
                score
                <= self.config.max_acceptable_uncertainty
            ),
            **estimate.summary(),
        }


# ============================================================
# UNCERTAINTY-AWARE PLANNING
# ============================================================

def uncertainty_penalized_reward(
    rewards: Tensor | np.ndarray,
    uncertainty: Tensor | np.ndarray,
    *,
    penalty_weight: float = 0.1,
) -> Tensor:
    """
    Apply uncertainty penalty to rewards.

    adjusted_reward =
        reward - weight * uncertainty

    This modifies planning scores,
    not predicted physical states.
    """

    if penalty_weight < 0:
        raise UncertaintyValidationError(
            "penalty_weight must be nonnegative"
        )

    rewards = as_tensor(
        rewards
    )

    uncertainty = as_tensor(
        uncertainty,
        device=rewards.device,
    )

    validate_matching_shapes(
        rewards,
        uncertainty,
    )

    validate_nonnegative(
        uncertainty,
        "Uncertainty",
    )

    return (
        rewards
        - penalty_weight
        * uncertainty
    )


def uncertainty_constrained_mask(
    uncertainty: Tensor | np.ndarray,
    *,
    maximum_uncertainty: float,
) -> Tensor:
    """
    Return True for candidates satisfying
    the uncertainty threshold.
    """

    if maximum_uncertainty < 0:
        raise UncertaintyValidationError(
            "maximum_uncertainty must be nonnegative"
        )

    uncertainty = as_tensor(
        uncertainty
    )

    validate_nonnegative(
        uncertainty,
        "Uncertainty",
    )

    return (
        uncertainty
        <= maximum_uncertainty
    )


# ============================================================
# ROLLOUT UNCERTAINTY
# ============================================================

def summarize_rollout_uncertainty(
    uncertainty: Tensor | np.ndarray,
) -> dict[str, Any]:
    """
    Analyze uncertainty across future steps.

    Expected shape:
        [B,H,D]

    B = batch size
    H = horizon
    D = state/position dimensions
    """

    uncertainty = as_tensor(
        uncertainty
    )

    if uncertainty.ndim != 3:
        raise UncertaintyValidationError(
            "Expected rollout uncertainty [B,H,D]"
        )

    validate_nonnegative(
        uncertainty,
        "Rollout uncertainty",
    )

    horizon_means = uncertainty.mean(
        dim=(0, 2)
    )

    horizon_maxima = uncertainty.amax(
        dim=(0, 2)
    )

    return {
        "mean_uncertainty": float(
            uncertainty.mean().item()
        ),
        "maximum_uncertainty": float(
            uncertainty.max().item()
        ),
        "final_horizon_uncertainty": float(
            horizon_means[-1].item()
        ),
        "per_horizon_mean": (
            horizon_means.cpu().tolist()
        ),
        "per_horizon_max": (
            horizon_maxima.cpu().tolist()
        ),
    }


# ============================================================
# UNCERTAINTY CALIBRATION
# ============================================================

@dataclass
class CalibrationMetrics:
    empirical_coverage: float
    nominal_coverage: float
    coverage_error: float

    mean_interval_width: float
    gaussian_nll: float

    rmse: float
    mean_predicted_std: float

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in vars(self).items()
        }


def evaluate_gaussian_calibration(
    predictions: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
    predicted_std: Tensor | np.ndarray,
    *,
    confidence_level: float = 0.95,
    epsilon: float = 1e-8,
) -> CalibrationMetrics:
    """
    Evaluate regression uncertainty.

    Metrics:
        empirical interval coverage
        nominal interval coverage
        coverage error
        interval width
        Gaussian NLL
        RMSE

    Use a held-out dataset for meaningful
    calibration assessment.
    """

    predictions = as_tensor(
        predictions
    )

    targets = as_tensor(
        targets,
        device=predictions.device,
    )

    predicted_std = as_tensor(
        predicted_std,
        device=predictions.device,
    )

    validate_matching_shapes(
        predictions,
        targets,
    )

    validate_matching_shapes(
        predictions,
        predicted_std,
    )

    validate_nonnegative(
        predicted_std,
        "Predicted std",
    )

    lower, upper = gaussian_prediction_interval(
        predictions,
        predicted_std,
        confidence_level=confidence_level,
    )

    inside_interval = (
        (targets >= lower)
        & (targets <= upper)
    )

    empirical_coverage = float(
        inside_interval.float()
        .mean()
        .item()
    )

    interval_width = (
        upper - lower
    ).mean()

    variance = (
        predicted_std.square()
        .clamp_min(epsilon)
    )

    errors = (
        targets - predictions
    )

    nll = (
        0.5
        * (
            math.log(2.0 * math.pi)
            + variance.log()
            + errors.square() / variance
        )
    ).mean()

    rmse = (
        errors.square()
        .mean()
        .sqrt()
    )

    return CalibrationMetrics(
        empirical_coverage=(
            empirical_coverage
        ),
        nominal_coverage=(
            confidence_level
        ),
        coverage_error=abs(
            empirical_coverage
            - confidence_level
        ),
        mean_interval_width=float(
            interval_width.item()
        ),
        gaussian_nll=float(
            nll.item()
        ),
        rmse=float(
            rmse.item()
        ),
        mean_predicted_std=float(
            predicted_std.mean().item()
        ),
    )


# ============================================================
# SIMPLE UNCERTAINTY CALIBRATOR
# ============================================================

class VarianceScaleCalibrator:
    """
    Fits one global scale factor to
    predicted standard deviations.

    Calibrated std:
        predicted_std * scale

    This is a lightweight calibration
    baseline, not a complete solution
    for distribution shift.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        self.epsilon = epsilon
        self.scale = 1.0
        self.fitted = False

    def fit(
        self,
        predictions: Tensor | np.ndarray,
        targets: Tensor | np.ndarray,
        predicted_std: Tensor | np.ndarray,
    ) -> float:
        predictions = as_tensor(
            predictions
        )

        targets = as_tensor(
            targets,
            device=predictions.device,
        )

        predicted_std = as_tensor(
            predicted_std,
            device=predictions.device,
        )

        validate_matching_shapes(
            predictions,
            targets,
        )

        validate_matching_shapes(
            predictions,
            predicted_std,
        )

        validate_nonnegative(
            predicted_std,
            "Predicted std",
        )

        normalized_squared_error = (
            (targets - predictions).square()
            / predicted_std.square().clamp_min(
                self.epsilon
            )
        )

        scale = (
            normalized_squared_error
            .mean()
            .sqrt()
        )

        self.scale = max(
            float(scale.item()),
            self.epsilon,
        )

        self.fitted = True

        return self.scale

    def transform(
        self,
        predicted_std: Tensor | np.ndarray,
    ) -> Tensor:
        if not self.fitted:
            raise UncertaintyError(
                "Calibrator must be fitted first"
            )

        std = as_tensor(
            predicted_std
        )

        validate_nonnegative(
            std,
            "Predicted std",
        )

        return std * self.scale

    def state_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "scale": self.scale,
            "fitted": self.fitted,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        self.epsilon = float(
            state["epsilon"]
        )

        self.scale = float(
            state["scale"]
        )

        self.fitted = bool(
            state["fitted"]
        )


# ============================================================
# PREDICTION UNCERTAINTY ENGINE
# ============================================================

class PredictionUncertaintyEngine:
    """
    Unified uncertainty interface for:

        next_state.py
        future_state.py
        trajectory_prediction.py
        frame_prediction.py
        event_prediction.py

    Models can provide:
        ensemble means/variances
        stochastic prediction samples
        Gaussian mean/log-variance
    """

    def __init__(
        self,
        config: UncertaintyConfig | None = None,
    ) -> None:
        self.config = (
            config or UncertaintyConfig()
        )

        self.risk_assessor = (
            UncertaintyRiskAssessor(
                self.config
            )
        )

    def from_ensemble(
        self,
        means: Tensor | np.ndarray,
        variances: (
            Tensor | np.ndarray | None
        ) = None,
    ) -> UncertaintyEstimate:
        return estimate_ensemble_uncertainty(
            means,
            variances,
        )

    def from_mc_samples(
        self,
        predictions: Tensor | np.ndarray,
    ) -> UncertaintyEstimate:
        return estimate_mc_uncertainty(
            predictions
        )

    def from_gaussian(
        self,
        mean: Tensor | np.ndarray,
        log_variance: Tensor | np.ndarray,
    ) -> UncertaintyEstimate:
        return estimate_gaussian_uncertainty(
            mean,
            log_variance,
        )

    def prediction_interval(
        self,
        estimate: UncertaintyEstimate,
    ) -> tuple[Tensor, Tensor]:
        return gaussian_prediction_interval(
            estimate.mean,
            estimate.total_std,
            confidence_level=(
                self.config.confidence_level
            ),
        )

    def assess_risk(
        self,
        estimate: UncertaintyEstimate,
    ) -> dict[str, Any]:
        return self.risk_assessor.assess(
            estimate
        )

    def penalize_rewards(
        self,
        rewards: Tensor | np.ndarray,
        uncertainty: Tensor | np.ndarray,
    ) -> Tensor:
        return uncertainty_penalized_reward(
            rewards,
            uncertainty,
            penalty_weight=(
                self.config.uncertainty_penalty_weight
            ),
        )

    def acceptable_mask(
        self,
        uncertainty: Tensor | np.ndarray,
    ) -> Tensor:
        return uncertainty_constrained_mask(
            uncertainty,
            maximum_uncertainty=(
                self.config.max_acceptable_uncertainty
            ),
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - PREDICTION UNCERTAINTY"
    )
    print("=" * 65)

    config = UncertaintyConfig(
        confidence_level=0.95,
        uncertainty_penalty_weight=0.2,
        max_acceptable_uncertainty=0.5,
    )

    engine = PredictionUncertaintyEngine(
        config
    )

    ensemble_size = 5
    batch_size = 8
    state_dim = 16

    # --------------------------------------------------------
    # ENSEMBLE UNCERTAINTY
    # --------------------------------------------------------

    ensemble_means = torch.randn(
        ensemble_size,
        batch_size,
        state_dim,
    ) * 0.2

    ensemble_variances = (
        torch.rand(
            ensemble_size,
            batch_size,
            state_dim,
        )
        * 0.05
        + 0.01
    )

    estimate = engine.from_ensemble(
        ensemble_means,
        ensemble_variances,
    )

    print(
        "Ensemble uncertainty:",
        estimate.summary(),
    )

    # --------------------------------------------------------
    # CONFIDENCE INTERVAL
    # --------------------------------------------------------

    lower, upper = engine.prediction_interval(
        estimate
    )

    print(
        "Prediction interval shapes:",
        tuple(lower.shape),
        tuple(upper.shape),
    )

    # --------------------------------------------------------
    # RISK ASSESSMENT
    # --------------------------------------------------------

    risk = engine.assess_risk(
        estimate
    )

    print(
        "Risk assessment:",
        risk,
    )

    # --------------------------------------------------------
    # MONTE CARLO UNCERTAINTY
    # --------------------------------------------------------

    mc_predictions = torch.randn(
        20,
        batch_size,
        state_dim,
    ) * 0.1

    mc_estimate = engine.from_mc_samples(
        mc_predictions
    )

    print(
        "Monte Carlo uncertainty:",
        mc_estimate.summary(),
    )

    # --------------------------------------------------------
    # ROLLOUT UNCERTAINTY
    # --------------------------------------------------------

    rollout_std = torch.rand(
        batch_size,
        10,
        state_dim,
    ) * 0.3

    rollout_summary = (
        summarize_rollout_uncertainty(
            rollout_std
        )
    )

    print(
        "Rollout uncertainty:",
        rollout_summary,
    )

    # --------------------------------------------------------
    # REWARD PENALTY
    # --------------------------------------------------------

    rewards = torch.ones(
        batch_size
    )

    uncertainty_scores = (
        estimate.total_std.mean(
            dim=-1
        )
    )

    adjusted_rewards = engine.penalize_rewards(
        rewards,
        uncertainty_scores,
    )

    print(
        "Adjusted rewards:",
        adjusted_rewards.tolist(),
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    predictions = torch.randn(
        100,
        4,
    )

    targets = (
        predictions
        + torch.randn_like(
            predictions
        ) * 0.2
    )

    predicted_std = torch.full_like(
        predictions,
        0.15,
    )

    calibrator = VarianceScaleCalibrator()

    scale = calibrator.fit(
        predictions,
        targets,
        predicted_std,
    )

    calibrated_std = calibrator.transform(
        predicted_std
    )

    calibration_metrics = (
        evaluate_gaussian_calibration(
            predictions,
            targets,
            calibrated_std,
        )
    )

    print(
        "Calibration scale:",
        scale,
    )

    print(
        "Calibration metrics:",
        calibration_metrics.to_dict(),
    )

    print(
        "Prediction uncertainty demonstration completed"
    )


if __name__ == "__main__":
    main()
