
```python
"""
NeuroWorld AI
File: neuroworld/agents/evaluation_agent.py

Autonomous World Model Evaluation Agent

Features:
- One-step state prediction evaluation
- Multi-step rollout evaluation
- Latent-state MSE, MAE, RMSE
- Trajectory prediction accuracy
- Gaussian negative log-likelihood
- Predictive uncertainty calibration
- Empirical confidence interval coverage
- Temporal consistency
- Physical constraint violations
- Goal-reaching evaluation
- Planning success rate
- Model benchmarking and comparison
- Weighted evaluation scores
- JSON benchmark report export
- PyTorch CPU/CUDA support

Compatible with:
    DynamicsModel
    LatentDynamics
    StochasticDynamics
    RecurrentDynamics
    TransformerDynamics
    TransitionModel

Standard dynamics interface:
    mean, logvar = model(state, action)

Standard rollout interface:
    result = model.rollout(
        initial_state,
        actions,
        deterministic=True,
    )

Dependencies:
    pip install torch
"""

from __future__ import annotations

import json
import math
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class EvaluationAgentConfig:
    latent_dim: int = 128
    action_dim: int = 32

    confidence_level: float = 0.95
    calibration_bins: int = 10

    goal_tolerance: float = 0.5

    max_state_norm: Optional[float] = None
    max_transition_norm: Optional[float] = None

    prediction_weight: float = 0.30
    rollout_weight: float = 0.25
    calibration_weight: float = 0.15
    consistency_weight: float = 0.10
    physics_weight: float = 0.10
    planning_weight: float = 0.10

    score_scale: float = 1.0
    epsilon: float = 1e-8


@dataclass
class EvaluationBatch:
    """
    states: [B,H+1,D]
    actions: [B,H,A]

    Optional goals:
        [B,D]

    Optional success labels:
        [B] boolean
    """

    states: Tensor
    actions: Tensor
    goals: Optional[Tensor] = None
    success_labels: Optional[Tensor] = None


@dataclass
class PredictionMetrics:
    mse: float
    mae: float
    rmse: float

    gaussian_nll: Optional[float] = None
    mean_uncertainty: Optional[float] = None
    confidence_coverage: Optional[float] = None
    calibration_error: Optional[float] = None


@dataclass
class RolloutMetrics:
    mse: float
    mae: float
    rmse: float

    final_state_mse: float
    final_state_rmse: float

    trajectory_length: int
    temporal_consistency_error: float

    physical_violation_rate: Optional[float] = None


@dataclass
class PlanningMetrics:
    success_rate: float
    mean_goal_distance: float
    mean_final_goal_distance: float


@dataclass
class EvaluationReport:
    model_name: str
    timestamp: float

    prediction: PredictionMetrics
    rollout: RolloutMetrics

    planning: Optional[PlanningMetrics]

    overall_score: float

    evaluation_time_seconds: float

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


class EvaluationAgent:
    """
    Evaluation agent for latent world models.

    Metrics are calculated from held-out
    state-action trajectories.

    Lower is better:
        MSE
        MAE
        RMSE
        NLL
        Calibration error
        Temporal consistency error
        Physical violation rate
        Goal distance

    Higher is better:
        Confidence coverage only when close
        to the requested nominal coverage
        Planning success rate
        Overall normalized score

    The overall score is a configurable
    heuristic, not a standardized benchmark.
    """

    def __init__(
        self,
        config: Optional[EvaluationAgentConfig] = None,
        physical_constraint_fn: Optional[
            Callable[[Tensor], Tensor]
        ] = None,
    ):
        self.config = config or EvaluationAgentConfig()

        self.physical_constraint_fn = (
            physical_constraint_fn
        )

        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive"
            )

        if cfg.action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        if not 0 < cfg.confidence_level < 1:
            raise ValueError(
                "confidence_level must be in (0,1)"
            )

        if cfg.calibration_bins < 2:
            raise ValueError(
                "calibration_bins must be >= 2"
            )

        if cfg.goal_tolerance < 0:
            raise ValueError(
                "goal_tolerance must be nonnegative"
            )

        if cfg.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        if cfg.score_scale <= 0:
            raise ValueError(
                "score_scale must be positive"
            )

        for name in (
            "max_state_norm",
            "max_transition_norm",
        ):
            value = getattr(cfg, name)

            if value is not None and (
                not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be positive and finite"
                )

        weights = [
            cfg.prediction_weight,
            cfg.rollout_weight,
            cfg.calibration_weight,
            cfg.consistency_weight,
            cfg.physics_weight,
            cfg.planning_weight,
        ]

        if not all(
            math.isfinite(weight) and weight >= 0
            for weight in weights
        ):
            raise ValueError(
                "Evaluation weights must be "
                "finite and nonnegative"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one evaluation weight "
                "must be positive"
            )

    def _validate_batch(
        self,
        batch: EvaluationBatch,
    ) -> None:
        cfg = self.config

        states = batch.states
        actions = batch.actions

        if states.ndim != 3:
            raise ValueError(
                "states must be [B,H+1,D]"
            )

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,H,A]"
            )

        batch_size = states.shape[0]
        horizon = actions.shape[1]

        if batch_size < 1 or horizon < 1:
            raise ValueError(
                "Batch size and horizon must be positive"
            )

        if states.shape != (
            batch_size,
            horizon + 1,
            cfg.latent_dim,
        ):
            raise ValueError(
                "Invalid states shape"
            )

        if actions.shape != (
            batch_size,
            horizon,
            cfg.action_dim,
        ):
            raise ValueError(
                "Invalid actions shape"
            )

        if (
            states.device != actions.device
            or states.dtype != actions.dtype
        ):
            raise ValueError(
                "State/action device or dtype mismatch"
            )

        if batch.goals is not None:
            if batch.goals.shape != (
                batch_size,
                cfg.latent_dim,
            ):
                raise ValueError(
                    "goals must be [B,D]"
                )

            if (
                batch.goals.device != states.device
                or batch.goals.dtype != states.dtype
            ):
                raise ValueError(
                    "Goal device/dtype mismatch"
                )

        if batch.success_labels is not None:
            if batch.success_labels.shape != (
                batch_size,
            ):
                raise ValueError(
                    "success_labels must be [B]"
                )

        for name, tensor in (
            ("states", states),
            ("actions", actions),
            ("goals", batch.goals),
        ):
            if tensor is None:
                continue

            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    @staticmethod
    def _errors(
        predicted: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if predicted.shape != target.shape:
            raise ValueError(
                "Prediction and target shape mismatch"
            )

        difference = predicted - target

        mse = difference.square().mean()
        mae = difference.abs().mean()
        rmse = torch.sqrt(mse)

        return mse, mae, rmse

    @staticmethod
    def gaussian_nll(
        mean: Tensor,
        logvar: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Diagonal Gaussian negative log-likelihood.

        Only meaningful when logvar represents
        a predictive Gaussian distribution.
        """

        if (
            mean.shape != logvar.shape
            or mean.shape != target.shape
        ):
            raise ValueError(
                "Gaussian tensor shapes must match"
            )

        return (
            0.5
            * (
                math.log(2.0 * math.pi)
                + logvar
                + (target - mean).square()
                * torch.exp(-logvar)
            )
        ).mean()

    def confidence_coverage(
        self,
        mean: Tensor,
        logvar: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Empirical marginal Gaussian interval
        coverage over all batch elements and
        latent dimensions.
        """

        normal = torch.distributions.Normal(
            torch.tensor(
                0.0,
                device=mean.device,
                dtype=mean.dtype,
            ),
            torch.tensor(
                1.0,
                device=mean.device,
                dtype=mean.dtype,
            ),
        )

        probability = (
            1.0 + self.config.confidence_level
        ) / 2.0

        z = normal.icdf(
            torch.tensor(
                probability,
                device=mean.device,
                dtype=mean.dtype,
            )
        )

        std = torch.exp(
            0.5 * logvar
        )

        lower = mean - z * std
        upper = mean + z * std

        covered = (
            (target >= lower)
            & (target <= upper)
        )

        return covered.float().mean()

    def calibration_error(
        self,
        mean: Tensor,
        logvar: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Gaussian calibration error across
        nominal central confidence intervals.

        This is a marginal calibration metric.
        It does not measure joint trajectory
        calibration or epistemic uncertainty.
        """

        std = torch.exp(
            0.5 * logvar
        ).clamp_min(self.config.epsilon)

        normalized_error = (
            (target - mean).abs() / std
        )

        normal = torch.distributions.Normal(
            torch.tensor(
                0.0,
                device=mean.device,
                dtype=mean.dtype,
            ),
            torch.tensor(
                1.0,
                device=mean.device,
                dtype=mean.dtype,
            ),
        )

        errors = []

        for index in range(
            1,
            self.config.calibration_bins + 1,
        ):
            probability = (
                index
                / (self.config.calibration_bins + 1)
            )

            p = torch.tensor(
                (1.0 + probability) / 2.0,
                device=mean.device,
                dtype=mean.dtype,
            )

            z = normal.icdf(p)

            empirical = (
                normalized_error <= z
            ).float().mean()

            errors.append(
                torch.abs(
                    empirical - probability
                )
            )

        return torch.stack(errors).mean()

    def evaluate_prediction(
        self,
        model: nn.Module,
        batch: EvaluationBatch,
    ) -> PredictionMetrics:
        """
        Evaluate one-step transitions.

        Uses all transitions in the batch
        without autoregressive rollout.
        """

        self._validate_batch(batch)

        states = batch.states[:, :-1, :]
        actions = batch.actions
        targets = batch.states[:, 1:, :]

        batch_size, horizon, latent_dim = (
            states.shape
        )

        flat_states = states.reshape(
            batch_size * horizon,
            latent_dim,
        )

        flat_actions = actions.reshape(
            batch_size * horizon,
            self.config.action_dim,
        )

        flat_targets = targets.reshape(
            batch_size * horizon,
            latent_dim,
        )

        with torch.no_grad():
            result = model(
                flat_states,
                flat_actions,
            )

            if isinstance(result, tuple):
                if len(result) < 2:
                    raise ValueError(
                        "Model must return mean, logvar"
                    )

                mean, logvar = result[:2]
            else:
                # Deterministic model.
                mean = result
                logvar = None

            mse, mae, rmse = self._errors(
                mean,
                flat_targets,
            )

            if logvar is not None:
                nll = self.gaussian_nll(
                    mean,
                    logvar,
                    flat_targets,
                )

                coverage = self.confidence_coverage(
                    mean,
                    logvar,
                    flat_targets,
                )

                calibration = self.calibration_error(
                    mean,
                    logvar,
                    flat_targets,
                )

                uncertainty = (
                    logvar.exp().mean()
                )
            else:
                nll = None
                coverage = None
                calibration = None
                uncertainty = None

        return PredictionMetrics(
            mse=float(mse),
            mae=float(mae),
            rmse=float(rmse),
            gaussian_nll=(
                float(nll)
                if nll is not None
                else None
            ),
            mean_uncertainty=(
                float(uncertainty)
                if uncertainty is not None
                else None
            ),
            confidence_coverage=(
                float(coverage)
                if coverage is not None
                else None
            ),
            calibration_error=(
                float(calibration)
                if calibration is not None
                else None
            ),
        )

    def _get_rollout_states(
        self,
        model: nn.Module,
        batch: EvaluationBatch,
    ) -> Tensor:
        """
        Generate a deterministic rollout.

        Models with rollout() are preferred.
        Otherwise, use autoregressive forward().
        """

        initial_state = batch.states[:, 0, :]
        actions = batch.actions

        with torch.no_grad():

            if hasattr(model, "rollout"):
                output = model.rollout(
                    initial_state,
                    actions,
                    deterministic=True,
                )

                if hasattr(output, "states"):
                    return output.states

                if isinstance(output, Tensor):
                    return output

                raise TypeError(
                    "Unsupported rollout output"
                )

            current = initial_state
            states = [current]

            for step in range(actions.shape[1]):
                prediction = model(
                    current,
                    actions[:, step, :],
                )

                if isinstance(prediction, tuple):
                    current = prediction[0]
                else:
                    current = prediction

                states.append(current)

            return torch.stack(
                states,
                dim=1,
            )

    def physical_violation_rate(
        self,
        states: Tensor,
    ) -> Optional[float]:
        """
        Evaluate configured physical constraints.

        Built-in checks:
            max_state_norm
            max_transition_norm

        Custom physical_constraint_fn:
            Returns a boolean mask where True
            indicates a violation.

        If no constraints are configured,
        returns None.

        Latent-vector norms are only proxy
        constraints, not physical laws.
        """

        violation_masks = []

        if self.config.max_state_norm is not None:
            norms = torch.linalg.vector_norm(
                states,
                dim=-1,
            )

            violation_masks.append(
                norms > self.config.max_state_norm
            )

        if (
            self.config.max_transition_norm
            is not None
        ):
            differences = (
                states[:, 1:, :]
                - states[:, :-1, :]
            )

            transition_norms = (
                torch.linalg.vector_norm(
                    differences,
                    dim=-1,
                )
            )

            # Align transitions with states.
            padded = torch.zeros(
                states.shape[:2],
                dtype=torch.bool,
                device=states.device,
            )

            padded[:, 1:] = (
                transition_norms
                > self.config.max_transition_norm
            )

            violation_masks.append(padded)

        if self.physical_constraint_fn is not None:
            custom = self.physical_constraint_fn(
                states
            )

            if custom.shape != states.shape[:2]:
                raise ValueError(
                    "Custom physical constraint "
                    "must return [B,H+1]"
                )

            if custom.dtype != torch.bool:
                raise TypeError(
                    "Physical constraint mask "
                    "must be boolean"
                )

            violation_masks.append(custom)

        if not violation_masks:
            return None

        combined = torch.zeros(
            states.shape[:2],
            dtype=torch.bool,
            device=states.device,
        )

        for mask in violation_masks:
            combined |= mask

        return float(
            combined.float().mean()
        )

    def evaluate_rollout(
        self,
        model: nn.Module,
        batch: EvaluationBatch,
    ) -> RolloutMetrics:
        """
        Evaluate open-loop future prediction.

        The model predicts all future states
        starting from the first observed state.
        """

        self._validate_batch(batch)

        predicted = self._get_rollout_states(
            model,
            batch,
        )

        target = batch.states

        if predicted.shape != target.shape:
            raise ValueError(
                "Rollout shape mismatch: "
                f"predicted={tuple(predicted.shape)}, "
                f"target={tuple(target.shape)}"
            )

        mse, mae, rmse = self._errors(
            predicted[:, 1:, :],
            target[:, 1:, :],
        )

        final_mse, _, final_rmse = self._errors(
            predicted[:, -1, :],
            target[:, -1, :],
        )

        predicted_deltas = (
            predicted[:, 1:, :]
            - predicted[:, :-1, :]
        )

        target_deltas = (
            target[:, 1:, :]
            - target[:, :-1, :]
        )

        temporal_consistency = (
            predicted_deltas
            - target_deltas
        ).square().mean()

        physics_rate = (
            self.physical_violation_rate(
                predicted
            )
        )

        return RolloutMetrics(
            mse=float(mse),
            mae=float(mae),
            rmse=float(rmse),
            final_state_mse=float(final_mse),
            final_state_rmse=float(final_rmse),
            trajectory_length=(
                batch.actions.shape[1]
            ),
            temporal_consistency_error=float(
                temporal_consistency
            ),
            physical_violation_rate=physics_rate,
        )

    def evaluate_planning(
        self,
        predicted_states: Tensor,
        goals: Tensor,
        success_labels: Optional[Tensor] = None,
    ) -> PlanningMetrics:
        """
        Evaluate goal reaching.

        predicted_states:
            [B,H+1,D]

        goals:
            [B,D]

        Success is determined by the final
        predicted latent-state distance.

        If success_labels are supplied,
        the reported success rate uses those
        observed labels instead.

        Latent distance is meaningful only
        when the representation has a suitable
        goal-distance metric.
        """

        if predicted_states.ndim != 3:
            raise ValueError(
                "predicted_states must be [B,H+1,D]"
            )

        if goals.shape != (
            predicted_states.shape[0],
            predicted_states.shape[-1],
        ):
            raise ValueError(
                "goals must be [B,D]"
            )

        distances = torch.linalg.vector_norm(
            predicted_states
            - goals.unsqueeze(1),
            dim=-1,
        )

        final_distances = distances[:, -1]

        if success_labels is None:
            success = (
                final_distances
                <= self.config.goal_tolerance
            )
        else:
            if success_labels.shape != (
                predicted_states.shape[0],
            ):
                raise ValueError(
                    "success_labels must be [B]"
                )

            success = success_labels.bool()

        return PlanningMetrics(
            success_rate=float(
                success.float().mean()
            ),
            mean_goal_distance=float(
                distances.mean()
            ),
            mean_final_goal_distance=float(
                final_distances.mean()
            ),
        )

    def _score_component(
        self,
        error: float,
    ) -> float:
        """
        Convert a nonnegative error into
        a bounded score in (0,1].
        """

        if not math.isfinite(error):
            return 0.0

        return math.exp(
            -max(error, 0.0)
            / self.config.score_scale
        )

    def calculate_overall_score(
        self,
        prediction: PredictionMetrics,
        rollout: RolloutMetrics,
        planning: Optional[
            PlanningMetrics
        ] = None,
    ) -> float:
        """
        Weighted normalized evaluation score.

        Components unavailable for a model
        are excluded and remaining weights
        are renormalized.

        Returns:
            Score from 0 to 100.

        This is a custom engineering score,
        not an externally validated benchmark.
        """

        cfg = self.config

        components = [
            (
                cfg.prediction_weight,
                self._score_component(
                    prediction.rmse
                ),
            ),
            (
                cfg.rollout_weight,
                self._score_component(
                    rollout.rmse
                ),
            ),
            (
                cfg.consistency_weight,
                self._score_component(
                    rollout.temporal_consistency_error
                ),
            ),
        ]

        if prediction.calibration_error is not None:
            components.append(
                (
                    cfg.calibration_weight,
                    self._score_component(
                        prediction.calibration_error
                    ),
                )
            )

        if (
            rollout.physical_violation_rate
            is not None
        ):
            components.append(
                (
                    cfg.physics_weight,
                    1.0
                    - rollout.physical_violation_rate,
                )
            )

        if planning is not None:
            components.append(
                (
                    cfg.planning_weight,
                    planning.success_rate,
                )
            )

        total_weight = sum(
            weight
            for weight, _ in components
        )

        if total_weight <= 0:
            raise ValueError(
                "No positive score weights available"
            )

        weighted_score = sum(
            weight * score
            for weight, score in components
        ) / total_weight

        return float(
            100.0
            * min(
                max(weighted_score, 0.0),
                1.0,
            )
        )

    def evaluate(
        self,
        model: nn.Module,
        batch: EvaluationBatch,
        model_name: str = "world_model",
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> EvaluationReport:
        """
        Run a complete evaluation benchmark.
        """

        self._validate_batch(batch)

        start_time = time.perf_counter()

        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                prediction = (
                    self.evaluate_prediction(
                        model,
                        batch,
                    )
                )

                rollout = self.evaluate_rollout(
                    model,
                    batch,
                )

                planning = None

                if batch.goals is not None:
                    predicted_states = (
                        self._get_rollout_states(
                            model,
                            batch,
                        )
                    )

                    planning = (
                        self.evaluate_planning(
                            predicted_states,
                            batch.goals,
                            batch.success_labels,
                        )
                    )

                score = (
                    self.calculate_overall_score(
                        prediction,
                        rollout,
                        planning,
                    )
                )
        finally:
            model.train(was_training)

        elapsed = (
            time.perf_counter() - start_time
        )

        return EvaluationReport(
            model_name=model_name,
            timestamp=time.time(),
            prediction=prediction,
            rollout=rollout,
            planning=planning,
            overall_score=score,
            evaluation_time_seconds=elapsed,
            metadata=dict(metadata or {}),
        )

    def compare_models(
        self,
        models: Mapping[str, nn.Module],
        batch: EvaluationBatch,
    ) -> List[EvaluationReport]:
        """
        Benchmark several models on
        the same evaluation batch.

        Results are sorted by overall score.
        """

        if not models:
            raise ValueError(
                "At least one model required"
            )

        reports = []

        for name, model in models.items():
            report = self.evaluate(
                model=model,
                batch=batch,
                model_name=name,
            )

            reports.append(report)

        reports.sort(
            key=lambda report: (
                report.overall_score
            ),
            reverse=True,
        )

        return reports

    @staticmethod
    def export_report(
        report: EvaluationReport,
        path: str | Path,
    ) -> None:
        """
        Save benchmark report as JSON.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                asdict(report),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def summarize(
        report: EvaluationReport,
    ) -> Dict[str, Any]:
        """
        Compact evaluation summary.
        """

        return {
            "model_name": report.model_name,
            "overall_score": (
                report.overall_score
            ),
            "prediction_rmse": (
                report.prediction.rmse
            ),
            "rollout_rmse": (
                report.rollout.rmse
            ),
            "final_state_rmse": (
                report.rollout.final_state_rmse
            ),
            "calibration_error": (
                report.prediction.calibration_error
            ),
            "physical_violation_rate": (
                report.rollout.physical_violation_rate
            ),
            "planning_success_rate": (
                report.planning.success_rate
                if report.planning is not None
                else None
            ),
            "evaluation_time_seconds": (
                report.evaluation_time_seconds
            ),
        }


if __name__ == "__main__":

    from neuroworld.dynamics.transition_model import (
        TransitionModel,
        TransitionModelConfig,
    )

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    latent_dim = 128
    action_dim = 32

    model = TransitionModel(
        TransitionModelConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            backend="stochastic",
        )
    ).to(device)

    batch_size = 8
    horizon = 10

    # Synthetic test data.
    #
    # Replace with held-out real trajectories
    # for meaningful evaluation.
    states = torch.randn(
        batch_size,
        horizon + 1,
        latent_dim,
        device=device,
    )

    actions = torch.randn(
        batch_size,
        horizon,
        action_dim,
        device=device,
    )

    goals = torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )

    batch = EvaluationBatch(
        states=states,
        actions=actions,
        goals=goals,
    )

    evaluator = EvaluationAgent(
        EvaluationAgentConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            max_state_norm=30.0,
            max_transition_norm=15.0,
        )
    )

    report = evaluator.evaluate(
        model=model,
        batch=batch,
        model_name="NeuroWorld-TransitionModel",
    )

    print("Evaluation summary:")

    for key, value in (
        evaluator.summarize(report).items()
    ):
        print(f"{key}: {value}")

    evaluator.export_report(
        report,
        "artifacts/evaluation_report.json",
    )
```
