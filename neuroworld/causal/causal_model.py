
"""
NeuroWorld AI
File: neuroworld/causal/causal_model.py

Structural Causal Model (SCM) Engine

Features:
- Integration with causal_graph.py
- Linear and nonlinear structural equations
- Observational world-state simulation
- Hard do-interventions
- Individual counterfactual reasoning
- Average treatment effect (ATE) estimation
- Paired Monte Carlo simulations
- Intervention comparison
- Causal action evaluation
- Intervention sensitivity analysis
- Explicit exogenous noise control
- JSON model serialization
- Reproducible simulations
- Input validation
- CPU-only, lightweight implementation

Important:
    This is a structural causal simulator.

    Causal relationships must be supplied
    or justified independently. The model
    does not discover causal structure
    automatically from observational data.

Dependencies:
    Python 3.10+
"""

from __future__ import annotations

import json
import math
import random

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)

from neuroworld.causal.causal_graph import (
    CausalGraph,
    CausalSimulationResult,
    CounterfactualResult,
)


@dataclass
class CausalModelConfig:
    num_monte_carlo_samples: int = 1000
    random_seed: int = 42

    confidence_level: float = 0.95

    validate_finite_values: bool = True

    # Numerical tolerance for checking factual
    # consistency in counterfactual reasoning.
    factual_tolerance: float = 1e-5


@dataclass
class CausalEffectEstimate:
    treatment: str
    outcome: str

    treatment_value: float
    control_value: float

    average_treatment_effect: float

    standard_error: float
    confidence_interval_lower: float
    confidence_interval_upper: float

    num_samples: int

    treatment_mean: float
    control_mean: float


@dataclass
class InterventionEvaluation:
    intervention: Dict[str, float]

    mean_outcomes: Dict[str, float]
    std_outcomes: Dict[str, float]

    num_samples: int


@dataclass
class CausalActionResult:
    action_name: str

    intervention: Dict[str, float]

    expected_reward: float
    expected_risk: float

    score: float

    outcome_means: Dict[str, float] = field(
        default_factory=dict
    )


@dataclass
class CausalPlanningResult:
    best_action: CausalActionResult
    ranked_actions: List[CausalActionResult]


class CausalModel:
    """
    High-level structural causal model.

    Wraps CausalGraph and adds:

        - Monte Carlo simulation
        - Causal effect estimation
        - Intervention evaluation
        - Action ranking
        - Counterfactual analysis

    Default graph equations:

        X_i = bias_i
              + sum_j(weight_ji * X_j)
              + noise_i

    The graph may also contain user-defined
    structural functions.

    This class assumes the supplied graph
    represents a valid causal DAG.
    """

    def __init__(
        self,
        graph: Optional[CausalGraph] = None,
        config: Optional[CausalModelConfig] = None,
    ):
        self.graph = graph or CausalGraph()

        self.config = (
            config or CausalModelConfig()
        )

        self._validate_config()

        self._rng = random.Random(
            self.config.random_seed
        )

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.num_monte_carlo_samples <= 0:
            raise ValueError(
                "num_monte_carlo_samples "
                "must be positive"
            )

        if not 0 < cfg.confidence_level < 1:
            raise ValueError(
                "confidence_level must be in (0,1)"
            )

        if cfg.factual_tolerance < 0:
            raise ValueError(
                "factual_tolerance must be non-negative"
            )

    def _validate_node(
        self,
        node: str,
    ) -> None:
        if node not in self.graph.nodes:
            raise KeyError(
                f"Unknown causal variable: {node}"
            )

    def _validate_intervention(
        self,
        intervention: Mapping[str, float],
    ) -> Dict[str, float]:
        validated = {}

        for node, value in intervention.items():
            self._validate_node(node)

            value = float(value)

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite intervention: {node}"
                )

            validated[node] = value

        return validated

    def _sample_noise(
        self,
    ) -> Dict[str, float]:
        """
        Draw one exogenous noise realization
        using this model's random generator.
        """

        return {
            node: self._rng.gauss(
                0.0,
                self.graph._noise_std[node],
            )
            for node in self.graph.nodes
        }

    def reset_random_seed(
        self,
        seed: Optional[int] = None,
    ) -> None:
        """
        Reset the Monte Carlo random generator.
        """

        if seed is None:
            seed = self.config.random_seed

        self._rng.seed(seed)

    def simulate(
        self,
        interventions: Optional[
            Mapping[str, float]
        ] = None,
        noise: Optional[
            Mapping[str, float]
        ] = None,
    ) -> CausalSimulationResult:
        """
        Simulate one world state.

        If noise is omitted, sample a new
        exogenous noise realization.
        """

        intervention_values = (
            self._validate_intervention(
                interventions or {}
            )
        )

        if noise is None:
            noise = self._sample_noise()

        return self.graph.simulate(
            interventions=intervention_values,
            noise=noise,
        )

    def observe(
        self,
        noise: Optional[
            Mapping[str, float]
        ] = None,
    ) -> CausalSimulationResult:
        """
        Simulate the observational distribution.

        No causal intervention is applied.
        """

        return self.simulate(
            interventions={},
            noise=noise,
        )

    def do(
        self,
        interventions: Mapping[str, float],
        noise: Optional[
            Mapping[str, float]
        ] = None,
    ) -> CausalSimulationResult:
        """
        Apply hard causal interventions.

        Example:
            model.do({"robot_speed": 3.0})
        """

        return self.simulate(
            interventions=interventions,
            noise=noise,
        )

    def counterfactual(
        self,
        factual_values: Mapping[str, float],
        interventions: Mapping[str, float],
        noise: Optional[
            Mapping[str, float]
        ] = None,
    ) -> CounterfactualResult:
        """
        Abduction-action-prediction.

        Uses the same inferred or supplied
        exogenous disturbances in factual
        and counterfactual worlds.

        The graph's custom structural equations
        must follow the additive-noise convention
        used by CausalGraph.counterfactual().
        """

        intervention_values = (
            self._validate_intervention(
                interventions
            )
        )

        return self.graph.counterfactual(
            factual_values=factual_values,
            interventions=intervention_values,
            noise=noise,
        )

    def sample_worlds(
        self,
        num_samples: Optional[int] = None,
        interventions: Optional[
            Mapping[str, float]
        ] = None,
    ) -> List[Dict[str, float]]:
        """
        Sample multiple causal worlds.

        Returns:
            List of node-value dictionaries.
        """

        if num_samples is None:
            num_samples = (
                self.config.num_monte_carlo_samples
            )

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be positive"
            )

        intervention_values = (
            self._validate_intervention(
                interventions or {}
            )
        )

        worlds = []

        for _ in range(num_samples):
            result = self.simulate(
                interventions=intervention_values,
            )

            worlds.append(result.values)

        return worlds

    @staticmethod
    def _mean(
        values: Sequence[float],
    ) -> float:
        if not values:
            raise ValueError(
                "Cannot average empty values"
            )

        return sum(values) / len(values)

    @classmethod
    def _std(
        cls,
        values: Sequence[float],
        sample: bool = False,
    ) -> float:
        """
        Compute standard deviation.

        sample=True:
            Uses n-1 denominator.
        """

        n = len(values)

        if n == 0:
            raise ValueError(
                "Cannot compute empty std"
            )

        if sample and n < 2:
            return 0.0

        mean = cls._mean(values)

        denominator = (
            n - 1 if sample else n
        )

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / denominator

        return math.sqrt(
            max(variance, 0.0)
        )

    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float,
        control_value: float,
        num_samples: Optional[int] = None,
    ) -> CausalEffectEstimate:
        """
        Estimate average treatment effect.

        ATE = E[Y | do(T=treatment)]
              - E[Y | do(T=control)]

        Uses paired Monte Carlo sampling:
        each treatment/control pair shares
        identical exogenous noise.

        This reduces simulation variance.

        Confidence intervals use a normal
        approximation to the sample mean.
        """

        self._validate_node(treatment)
        self._validate_node(outcome)

        if num_samples is None:
            num_samples = (
                self.config.num_monte_carlo_samples
            )

        if num_samples < 2:
            raise ValueError(
                "ATE estimation requires "
                "at least two samples"
            )

        treatment_value = float(
            treatment_value
        )

        control_value = float(
            control_value
        )

        if not (
            math.isfinite(treatment_value)
            and math.isfinite(control_value)
        ):
            raise ValueError(
                "Treatment/control values "
                "must be finite"
            )

        treatment_outcomes = []
        control_outcomes = []
        paired_effects = []

        for _ in range(num_samples):
            noise = self._sample_noise()

            treated = self.graph.simulate(
                interventions={
                    treatment: treatment_value
                },
                noise=noise,
            )

            controlled = self.graph.simulate(
                interventions={
                    treatment: control_value
                },
                noise=noise,
            )

            treated_value = (
                treated.values[outcome]
            )

            controlled_value = (
                controlled.values[outcome]
            )

            treatment_outcomes.append(
                treated_value
            )

            control_outcomes.append(
                controlled_value
            )

            paired_effects.append(
                treated_value - controlled_value
            )

        ate = self._mean(paired_effects)

        standard_error = (
            self._std(
                paired_effects,
                sample=True,
            )
            / math.sqrt(num_samples)
        )

        # Normal quantile from the standard
        # library's NormalDist.
        from statistics import NormalDist

        confidence = self.config.confidence_level

        z_score = NormalDist().inv_cdf(
            (1.0 + confidence) / 2.0
        )

        margin = z_score * standard_error

        return CausalEffectEstimate(
            treatment=treatment,
            outcome=outcome,
            treatment_value=treatment_value,
            control_value=control_value,
            average_treatment_effect=ate,
            standard_error=standard_error,
            confidence_interval_lower=(
                ate - margin
            ),
            confidence_interval_upper=(
                ate + margin
            ),
            num_samples=num_samples,
            treatment_mean=self._mean(
                treatment_outcomes
            ),
            control_mean=self._mean(
                control_outcomes
            ),
        )

    def evaluate_intervention(
        self,
        intervention: Mapping[str, float],
        num_samples: Optional[int] = None,
        outcome_nodes: Optional[
            Sequence[str]
        ] = None,
    ) -> InterventionEvaluation:
        """
        Estimate outcome distributions under
        a fixed causal intervention.
        """

        if num_samples is None:
            num_samples = (
                self.config.num_monte_carlo_samples
            )

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be positive"
            )

        intervention_values = (
            self._validate_intervention(
                intervention
            )
        )

        if outcome_nodes is None:
            outcome_nodes = list(
                self.graph.nodes
            )

        for node in outcome_nodes:
            self._validate_node(node)

        outcomes = {
            node: []
            for node in outcome_nodes
        }

        for _ in range(num_samples):
            result = self.simulate(
                interventions=intervention_values,
            )

            for node in outcome_nodes:
                outcomes[node].append(
                    result.values[node]
                )

        means = {
            node: self._mean(values)
            for node, values in outcomes.items()
        }

        stds = {
            node: self._std(values)
            for node, values in outcomes.items()
        }

        return InterventionEvaluation(
            intervention=dict(
                intervention_values
            ),
            mean_outcomes=means,
            std_outcomes=stds,
            num_samples=num_samples,
        )

    def compare_interventions(
        self,
        interventions: Mapping[
            str,
            Mapping[str, float],
        ],
        outcome_nodes: Sequence[str],
        num_samples: Optional[int] = None,
    ) -> Dict[
        str,
        InterventionEvaluation,
    ]:
        """
        Compare named intervention scenarios.

        Uses common random noise across
        scenarios for paired comparisons.

        Example:
            {
                "normal": {"speed": 10},
                "slow": {"speed": 3}
            }
        """

        if not interventions:
            raise ValueError(
                "At least one intervention required"
            )

        if not outcome_nodes:
            raise ValueError(
                "At least one outcome node required"
            )

        for node in outcome_nodes:
            self._validate_node(node)

        if num_samples is None:
            num_samples = (
                self.config.num_monte_carlo_samples
            )

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be positive"
            )

        validated = {
            name: self._validate_intervention(values)
            for name, values in interventions.items()
        }

        samples = {
            name: {
                node: []
                for node in outcome_nodes
            }
            for name in validated
        }

        for _ in range(num_samples):
            noise = self._sample_noise()

            for name, intervention in (
                validated.items()
            ):
                result = self.graph.simulate(
                    interventions=intervention,
                    noise=noise,
                )

                for node in outcome_nodes:
                    samples[name][node].append(
                        result.values[node]
                    )

        evaluations = {}

        for name, intervention in validated.items():
            node_samples = samples[name]

            evaluations[name] = (
                InterventionEvaluation(
                    intervention=dict(intervention),
                    mean_outcomes={
                        node: self._mean(values)
                        for node, values
                        in node_samples.items()
                    },
                    std_outcomes={
                        node: self._std(values)
                        for node, values
                        in node_samples.items()
                    },
                    num_samples=num_samples,
                )
            )

        return evaluations

    def plan_actions(
        self,
        actions: Mapping[
            str,
            Mapping[str, float],
        ],
        reward_node: str,
        risk_node: Optional[str] = None,
        risk_penalty: float = 1.0,
        num_samples: Optional[int] = None,
    ) -> CausalPlanningResult:
        """
        Rank candidate interventions.

        Score:
            expected_reward
            - risk_penalty * expected_risk

        Actions are evaluated under the same
        sampled exogenous disturbances.

        This is a simple one-step causal planner,
        not a safety-certified controller.
        """

        if not actions:
            raise ValueError(
                "At least one action is required"
            )

        self._validate_node(reward_node)

        outcome_nodes = [reward_node]

        if risk_node is not None:
            self._validate_node(risk_node)

            if risk_node not in outcome_nodes:
                outcome_nodes.append(risk_node)

        if (
            not math.isfinite(risk_penalty)
            or risk_penalty < 0
        ):
            raise ValueError(
                "risk_penalty must be finite "
                "and non-negative"
            )

        evaluations = self.compare_interventions(
            interventions=actions,
            outcome_nodes=outcome_nodes,
            num_samples=num_samples,
        )

        ranked_actions = []

        for name, evaluation in (
            evaluations.items()
        ):
            expected_reward = (
                evaluation.mean_outcomes[
                    reward_node
                ]
            )

            expected_risk = (
                evaluation.mean_outcomes[
                    risk_node
                ]
                if risk_node is not None
                else 0.0
            )

            score = (
                expected_reward
                - risk_penalty * expected_risk
            )

            ranked_actions.append(
                CausalActionResult(
                    action_name=name,
                    intervention=dict(
                        evaluation.intervention
                    ),
                    expected_reward=(
                        expected_reward
                    ),
                    expected_risk=expected_risk,
                    score=score,
                    outcome_means=dict(
                        evaluation.mean_outcomes
                    ),
                )
            )

        ranked_actions.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return CausalPlanningResult(
            best_action=ranked_actions[0],
            ranked_actions=ranked_actions,
        )

    def intervention_sensitivity(
        self,
        treatment: str,
        outcome: str,
        values: Sequence[float],
        num_samples: Optional[int] = None,
    ) -> Dict[float, float]:
        """
        Estimate expected outcome for
        different treatment values.

        Uses paired disturbances across
        all treatment values.
        """

        self._validate_node(treatment)
        self._validate_node(outcome)

        if not values:
            raise ValueError(
                "At least one treatment value required"
            )

        if num_samples is None:
            num_samples = (
                self.config.num_monte_carlo_samples
            )

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be positive"
            )

        unique_values = [
            float(value)
            for value in dict.fromkeys(values)
        ]

        for value in unique_values:
            if not math.isfinite(value):
                raise ValueError(
                    "Treatment values must be finite"
                )

        results = {
            value: []
            for value in unique_values
        }

        for _ in range(num_samples):
            noise = self._sample_noise()

            for value in unique_values:
                simulation = self.graph.simulate(
                    interventions={
                        treatment: value
                    },
                    noise=noise,
                )

                results[value].append(
                    simulation.values[outcome]
                )

        return {
            value: self._mean(outcomes)
            for value, outcomes in results.items()
        }

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save configuration and causal graph.

        Custom Python structural functions
        cannot be serialized to JSON.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "config": {
                "num_monte_carlo_samples": (
                    self.config.num_monte_carlo_samples
                ),
                "random_seed": (
                    self.config.random_seed
                ),
                "confidence_level": (
                    self.config.confidence_level
                ),
                "validate_finite_values": (
                    self.config.validate_finite_values
                ),
                "factual_tolerance": (
                    self.config.factual_tolerance
                ),
            },
            "graph": self.graph.to_dict(),
        }

        path.write_text(
            json.dumps(
                data,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "CausalModel":
        """Restore a saved causal model."""

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        graph = CausalGraph.from_dict(
            data["graph"]
        )

        config = CausalModelConfig(
            **data["config"]
        )

        return cls(
            graph=graph,
            config=config,
        )

    def summary(self) -> Dict[str, Any]:
        """Return causal model information."""

        return {
            "graph": self.graph.summary(),
            "monte_carlo_samples": (
                self.config.num_monte_carlo_samples
            ),
            "confidence_level": (
                self.config.confidence_level
            ),
        }


if __name__ == "__main__":

    # ------------------------------------------------
    # Example: Robot Safety Causal Model
    # ------------------------------------------------

    graph = CausalGraph()

    for node in (
        "robot_speed",
        "obstacle_density",
        "collision_risk",
        "task_progress",
        "reward",
    ):
        graph.add_node(node)

    # Example structural relationships.
    graph.add_edge(
        "robot_speed",
        "collision_risk",
        weight=0.5,
    )

    graph.add_edge(
        "obstacle_density",
        "collision_risk",
        weight=0.8,
    )

    graph.add_edge(
        "robot_speed",
        "task_progress",
        weight=1.2,
    )

    graph.add_edge(
        "task_progress",
        "reward",
        weight=1.0,
    )

    graph.add_edge(
        "collision_risk",
        "reward",
        weight=-2.0,
    )

    graph.set_structural_equation(
        "robot_speed",
        bias=5.0,
        noise_std=0.5,
    )

    graph.set_structural_equation(
        "obstacle_density",
        bias=2.0,
        noise_std=0.2,
    )

    graph.set_structural_equation(
        "collision_risk",
        bias=0.0,
        noise_std=0.1,
    )

    graph.set_structural_equation(
        "task_progress",
        bias=0.0,
        noise_std=0.1,
    )

    graph.set_structural_equation(
        "reward",
        bias=0.0,
        noise_std=0.1,
    )

    model = CausalModel(
        graph=graph,
        config=CausalModelConfig(
            num_monte_carlo_samples=500,
            random_seed=42,
        ),
    )

    print("Model summary:")
    print(model.summary())

    # Observational simulation.
    factual = model.observe()

    print("\nObserved world:")
    print(factual.values)

    # Hard intervention.
    intervention = model.do(
        {"robot_speed": 3.0}
    )

    print("\nIntervened world:")
    print(intervention.values)

    # Counterfactual.
    counterfactual = model.counterfactual(
        factual_values=factual.values,
        interventions={
            "robot_speed": 3.0
        },
    )

    print("\nCounterfactual effects:")
    print(counterfactual.effects)

    # Average treatment effect.
    ate = model.estimate_ate(
        treatment="robot_speed",
        outcome="collision_risk",
        treatment_value=8.0,
        control_value=3.0,
        num_samples=500,
    )

    print("\nAverage treatment effect:")
    print(
        ate.average_treatment_effect
    )

    print(
        "Confidence interval:",
        (
            ate.confidence_interval_lower,
            ate.confidence_interval_upper,
        ),
    )

    # Evaluate multiple robot actions.
    planning = model.plan_actions(
        actions={
            "slow": {
                "robot_speed": 2.0
            },
            "normal": {
                "robot_speed": 5.0
            },
            "fast": {
                "robot_speed": 8.0
            },
        },
        reward_node="reward",
        risk_node="collision_risk",
        risk_penalty=2.0,
        num_samples=500,
    )

    print("\nBest action:")
    print(
        planning.best_action.action_name
    )

    for action in planning.ranked_actions:
        print(
            action.action_name,
            "score:",
            round(action.score, 4),
        )

    # Sensitivity analysis.
    sensitivity = (
        model.intervention_sensitivity(
            treatment="robot_speed",
            outcome="collision_risk",
            values=[
                1.0,
                3.0,
                5.0,
                7.0,
                9.0,
            ],
            num_samples=500,
        )
    )

    print("\nSpeed sensitivity:")
    print(sensitivity)

    # Save model.
    model.save(
        "artifacts/causal_model.json"
    )

    # Restore model.
    restored = CausalModel.load(
        "artifacts/causal_model.json"
    )

    print("\nRestored model:")
    print(restored.summary())
