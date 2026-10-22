
"""
NeuroWorld AI
File: neuroworld/causal/intervention.py

Causal Intervention & Action Simulation Engine

Features:
- Hard do-interventions
- Soft structural interventions
- Multiple simultaneous interventions
- Paired Monte Carlo scenario comparison
- Average intervention effect estimation
- Risk-aware action scoring
- Intervention feasibility constraints
- Intervention cost estimation
- Causal effect tracing
- JSON-compatible reports
- Reproducible simulations

Dependencies:
    Python 3.10+

Requires:
    neuroworld/causal/causal_graph.py
    neuroworld/causal/causal_model.py

Important:
    Causal results are valid only under the
    assumptions of the supplied structural
    causal model.

    This module is for simulation and analysis.
    It does not execute physical robot actions.
"""

from __future__ import annotations

import json
import math
import random

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from statistics import NormalDist, mean, stdev
from typing import Any, Dict, List, Mapping, Optional, Sequence

from neuroworld.causal.causal_graph import (
    CausalGraph,
)
from neuroworld.causal.causal_model import (
    CausalModel,
)


class InterventionType(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class InterventionConfig:
    num_samples: int = 500
    random_seed: int = 42

    confidence_level: float = 0.95

    risk_penalty: float = 1.0
    cost_penalty: float = 0.01

    enforce_constraints: bool = True


@dataclass
class InterventionConstraint:
    node: str

    minimum: Optional[float] = None
    maximum: Optional[float] = None

    allowed_values: Optional[
        List[float]
    ] = None


@dataclass
class InterventionSpecification:
    name: str

    intervention_type: InterventionType = (
        InterventionType.HARD
    )

    # Hard intervention:
    #   node -> replacement value
    values: Dict[str, float] = field(
        default_factory=dict
    )

    # Soft intervention:
    #   node -> replacement bias
    bias_overrides: Dict[str, float] = field(
        default_factory=dict
    )

    # Soft intervention:
    #   "source->target" -> replacement weight
    weight_overrides: Dict[str, float] = field(
        default_factory=dict
    )

    description: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class InterventionResult:
    name: str

    intervention_type: str

    values: Dict[str, float]

    effects: Dict[str, float]

    factual_values: Dict[str, float]

    intervention_cost: float

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class InterventionStatistics:
    name: str

    mean_outcomes: Dict[str, float]
    std_outcomes: Dict[str, float]

    mean_effects: Dict[str, float]

    confidence_intervals: Dict[
        str,
        List[float],
    ]

    num_samples: int


@dataclass
class InterventionActionScore:
    name: str

    expected_reward: float
    expected_risk: float

    intervention_cost: float

    score: float


@dataclass
class InterventionRanking:
    best_action: InterventionActionScore

    ranked_actions: List[
        InterventionActionScore
    ]


class InterventionEngine:
    """
    Structural causal intervention engine.

    Hard intervention:
        do(X=x)

        Replaces the structural equation
        for X with a constant value.

    Soft intervention:
        Modifies structural parameters
        without deleting incoming edges.

    Soft interventions are supported for
    the graph's built-in linear equations.

    Custom structural functions are not
    modified by bias/weight overrides.
    """

    def __init__(
        self,
        causal_model: CausalModel,
        config: Optional[
            InterventionConfig
        ] = None,
        constraints: Optional[
            Sequence[InterventionConstraint]
        ] = None,
    ):
        self.causal_model = causal_model
        self.graph: CausalGraph = (
            causal_model.graph
        )

        self.config = (
            config or InterventionConfig()
        )

        self.constraints = {
            constraint.node: constraint
            for constraint in (constraints or [])
        }

        self._validate_config()

        self._rng = random.Random(
            self.config.random_seed
        )

        for constraint in self.constraints.values():
            self._validate_constraint(
                constraint
            )

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.num_samples <= 0:
            raise ValueError(
                "num_samples must be positive"
            )

        if not 0 < cfg.confidence_level < 1:
            raise ValueError(
                "confidence_level must be in (0,1)"
            )

        for name, value in (
            ("risk_penalty", cfg.risk_penalty),
            ("cost_penalty", cfg.cost_penalty),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite "
                    "and non-negative"
                )

    def _validate_constraint(
        self,
        constraint: InterventionConstraint,
    ) -> None:
        self._require_node(
            constraint.node
        )

        if (
            constraint.minimum is not None
            and not math.isfinite(
                constraint.minimum
            )
        ):
            raise ValueError(
                "Constraint minimum must be finite"
            )

        if (
            constraint.maximum is not None
            and not math.isfinite(
                constraint.maximum
            )
        ):
            raise ValueError(
                "Constraint maximum must be finite"
            )

        if (
            constraint.minimum is not None
            and constraint.maximum is not None
            and constraint.minimum
            > constraint.maximum
        ):
            raise ValueError(
                "Constraint minimum exceeds maximum"
            )

        if constraint.allowed_values is not None:
            if not constraint.allowed_values:
                raise ValueError(
                    "allowed_values cannot be empty"
                )

            if not all(
                math.isfinite(v)
                for v in constraint.allowed_values
            ):
                raise ValueError(
                    "allowed_values must be finite"
                )

    def _require_node(
        self,
        node: str,
    ) -> None:
        if node not in self.graph.nodes:
            raise KeyError(
                f"Unknown causal node: {node}"
            )

    def _validate_values(
        self,
        values: Mapping[str, float],
    ) -> Dict[str, float]:
        result = {}

        for node, value in values.items():
            self._require_node(node)

            value = float(value)

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite intervention: {node}"
                )

            result[node] = value

        return result

    def _check_constraints(
        self,
        values: Mapping[str, float],
    ) -> None:
        if not self.config.enforce_constraints:
            return

        for node, value in values.items():
            constraint = self.constraints.get(
                node
            )

            if constraint is None:
                continue

            if (
                constraint.minimum is not None
                and value < constraint.minimum
            ):
                raise ValueError(
                    f"{node} below minimum"
                )

            if (
                constraint.maximum is not None
                and value > constraint.maximum
            ):
                raise ValueError(
                    f"{node} above maximum"
                )

            if constraint.allowed_values is not None:
                if not any(
                    math.isclose(
                        value,
                        allowed,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    for allowed in (
                        constraint.allowed_values
                    )
                ):
                    raise ValueError(
                        f"{node} value not allowed"
                    )

    def _validate_specification(
        self,
        spec: InterventionSpecification,
    ) -> None:
        if not spec.name:
            raise ValueError(
                "Intervention name cannot be empty"
            )

        spec.intervention_type = (
            InterventionType(
                spec.intervention_type
            )
        )

        values = self._validate_values(
            spec.values
        )

        self._check_constraints(values)

        if (
            spec.intervention_type
            == InterventionType.HARD
        ):
            if (
                spec.bias_overrides
                or spec.weight_overrides
            ):
                raise ValueError(
                    "Hard interventions cannot "
                    "contain soft parameter overrides"
                )

        elif (
            spec.intervention_type
            == InterventionType.SOFT
        ):
            if spec.values:
                raise ValueError(
                    "Soft interventions cannot "
                    "contain hard value assignments"
                )

            for node, bias in (
                spec.bias_overrides.items()
            ):
                self._require_node(node)

                if not math.isfinite(bias):
                    raise ValueError(
                        "Bias override must be finite"
                    )

                if node in self.graph._functions:
                    raise ValueError(
                        "Cannot override bias of "
                        "a custom structural function"
                    )

            for edge_name, weight in (
                spec.weight_overrides.items()
            ):
                if not math.isfinite(weight):
                    raise ValueError(
                        "Weight override must be finite"
                    )

                source, target = (
                    self._parse_edge_name(
                        edge_name
                    )
                )

                if (source, target) not in (
                    self.graph.edges
                ):
                    raise KeyError(
                        f"Unknown edge: {edge_name}"
                    )

                if target in self.graph._functions:
                    raise ValueError(
                        "Cannot override an incoming "
                        "weight of a custom "
                        "structural function"
                    )

    @staticmethod
    def _parse_edge_name(
        edge_name: str,
    ) -> tuple[str, str]:
        parts = edge_name.split("->")

        if len(parts) != 2:
            raise ValueError(
                "Edge key must use source->target"
            )

        source = parts[0].strip()
        target = parts[1].strip()

        if not source or not target:
            raise ValueError(
                "Invalid edge key"
            )

        return source, target

    def _sample_noise(
        self,
    ) -> Dict[str, float]:
        return {
            node: self._rng.gauss(
                0.0,
                self.graph._noise_std[node],
            )
            for node in self.graph.nodes
        }

    def _simulate_specification(
        self,
        spec: InterventionSpecification,
        noise: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Evaluate one intervention.

        Soft interventions are applied without
        mutating the original causal graph.
        """

        if (
            spec.intervention_type
            == InterventionType.HARD
        ):
            return self.graph.simulate(
                interventions=spec.values,
                noise=noise,
            ).values

        # Evaluate a soft-intervention world
        # using modified structural parameters.
        order = self.graph.topological_sort()

        values: Dict[str, float] = {}

        for node in order:

            if node in self.graph._functions:
                values[node] = (
                    self.graph._evaluate_node(
                        node,
                        values,
                        noise.get(node, 0.0),
                    )
                )
                continue

            bias = spec.bias_overrides.get(
                node,
                self.graph._biases[node],
            )

            result = (
                bias + noise.get(node, 0.0)
            )

            for parent in self.graph._parents[node]:

                edge_key = f"{parent}->{node}"

                weight = (
                    spec.weight_overrides.get(
                        edge_key,
                        self.graph.edges[
                            (parent, node)
                        ].weight,
                    )
                )

                result += (
                    weight * values[parent]
                )

            if not math.isfinite(result):
                raise ValueError(
                    f"Non-finite outcome: {node}"
                )

            values[node] = float(result)

        return values

    @staticmethod
    def intervention_cost(
        factual_values: Mapping[str, float],
        spec: InterventionSpecification,
    ) -> float:
        """
        Basic quadratic intervention cost.

        Hard:
            Sum squared changes in target values.

        Soft:
            Sum squared parameter modifications.

        Soft parameter costs are computed
        separately in _soft_intervention_cost().
        """

        if (
            spec.intervention_type
            == InterventionType.HARD
        ):
            return sum(
                (
                    value
                    - factual_values[node]
                ) ** 2
                for node, value in (
                    spec.values.items()
                )
            )

        return 0.0

    def _soft_intervention_cost(
        self,
        spec: InterventionSpecification,
    ) -> float:
        cost = 0.0

        for node, new_bias in (
            spec.bias_overrides.items()
        ):
            old_bias = (
                self.graph._biases[node]
            )

            cost += (
                new_bias - old_bias
            ) ** 2

        for edge_name, new_weight in (
            spec.weight_overrides.items()
        ):
            source, target = (
                self._parse_edge_name(
                    edge_name
                )
            )

            old_weight = (
                self.graph.edges[
                    (source, target)
                ].weight
            )

            cost += (
                new_weight - old_weight
            ) ** 2

        return cost

    def apply(
        self,
        spec: InterventionSpecification,
        factual_values: Optional[
            Mapping[str, float]
        ] = None,
        noise: Optional[
            Mapping[str, float]
        ] = None,
    ) -> InterventionResult:
        """
        Apply a single intervention.

        If factual_values are provided and noise
        is omitted, infer additive exogenous noise
        to obtain an individual counterfactual.

        Otherwise, sample or reuse explicit noise.

        For custom nonlinear equations, automatic
        abduction requires additive noise.
        """

        self._validate_specification(spec)

        if factual_values is not None:
            factual = self._validate_values(
                factual_values
            )

            if set(factual) != set(self.graph.nodes):
                raise ValueError(
                    "Complete factual values required"
                )

            if noise is None:
                inferred_noise = {}

                for node in (
                    self.graph.topological_sort()
                ):
                    baseline = (
                        self.graph._evaluate_node(
                            node=node,
                            values=factual,
                            noise=0.0,
                        )
                    )

                    inferred_noise[node] = (
                        factual[node] - baseline
                    )

                noise = inferred_noise

        if noise is None:
            noise = self._sample_noise()

        if factual_values is None:
            factual = self.graph.simulate(
                noise=noise
            ).values

        counterfactual = (
            self._simulate_specification(
                spec,
                noise,
            )
        )

        effects = {
            node: (
                counterfactual[node]
                - factual[node]
            )
            for node in self.graph.nodes
        }

        if (
            spec.intervention_type
            == InterventionType.HARD
        ):
            cost = self.intervention_cost(
                factual,
                spec,
            )
        else:
            cost = self._soft_intervention_cost(
                spec
            )

        return InterventionResult(
            name=spec.name,
            intervention_type=(
                spec.intervention_type.value
            ),
            values=counterfactual,
            effects=effects,
            factual_values=dict(factual),
            intervention_cost=cost,
            metadata=dict(spec.metadata),
        )

    def compare(
        self,
        specifications: Sequence[
            InterventionSpecification
        ],
        outcome_nodes: Optional[
            Sequence[str]
        ] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[
        str,
        InterventionStatistics,
    ]:
        """
        Compare interventions with paired
        Monte Carlo simulations.

        Every scenario receives the same
        exogenous noise for each sample.

        This makes differences between
        scenarios easier to estimate.
        """

        if not specifications:
            raise ValueError(
                "At least one specification required"
            )

        names = [
            spec.name
            for spec in specifications
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Intervention names must be unique"
            )

        for spec in specifications:
            self._validate_specification(spec)

        if outcome_nodes is None:
            outcome_nodes = list(
                self.graph.nodes
            )

        if not outcome_nodes:
            raise ValueError(
                "At least one outcome node required"
            )

        for node in outcome_nodes:
            self._require_node(node)

        if num_samples is None:
            num_samples = self.config.num_samples

        if num_samples < 2:
            raise ValueError(
                "At least two samples required"
            )

        outcome_samples = {
            spec.name: {
                node: []
                for node in outcome_nodes
            }
            for spec in specifications
        }

        effect_samples = {
            spec.name: {
                node: []
                for node in outcome_nodes
            }
            for spec in specifications
        }

        for _ in range(num_samples):

            noise = self._sample_noise()

            factual = self.graph.simulate(
                noise=noise
            ).values

            for spec in specifications:

                result = (
                    self._simulate_specification(
                        spec,
                        noise,
                    )
                )

                for node in outcome_nodes:

                    outcome_samples[
                        spec.name
                    ][node].append(
                        result[node]
                    )

                    effect_samples[
                        spec.name
                    ][node].append(
                        result[node]
                        - factual[node]
                    )

        z_score = NormalDist().inv_cdf(
            (
                1.0
                + self.config.confidence_level
            ) / 2.0
        )

        statistics = {}

        for spec in specifications:

            outcomes = outcome_samples[
                spec.name
            ]

            effects = effect_samples[
                spec.name
            ]

            means = {
                node: mean(values)
                for node, values in outcomes.items()
            }

            stds = {
                node: stdev(values)
                for node, values in outcomes.items()
            }

            mean_effects = {
                node: mean(values)
                for node, values in effects.items()
            }

            confidence_intervals = {}

            for node, values in effects.items():

                effect_mean = mean(values)

                standard_error = (
                    stdev(values)
                    / math.sqrt(num_samples)
                )

                margin = (
                    z_score * standard_error
                )

                confidence_intervals[node] = [
                    effect_mean - margin,
                    effect_mean + margin,
                ]

            statistics[spec.name] = (
                InterventionStatistics(
                    name=spec.name,
                    mean_outcomes=means,
                    std_outcomes=stds,
                    mean_effects=mean_effects,
                    confidence_intervals=(
                        confidence_intervals
                    ),
                    num_samples=num_samples,
                )
            )

        return statistics

    def rank_actions(
        self,
        specifications: Sequence[
            InterventionSpecification
        ],
        reward_node: str,
        risk_node: Optional[str] = None,
        num_samples: Optional[int] = None,
    ) -> InterventionRanking:
        """
        Rank intervention scenarios.

        Score:
            expected_reward
            - risk_penalty * expected_risk
            - cost_penalty * intervention_cost

        The cost term is evaluated relative
        to the expected factual world.
        """

        self._require_node(reward_node)

        outcome_nodes = [reward_node]

        if risk_node is not None:
            self._require_node(risk_node)

            if risk_node not in outcome_nodes:
                outcome_nodes.append(risk_node)

        stats = self.compare(
            specifications=specifications,
            outcome_nodes=outcome_nodes,
            num_samples=num_samples,
        )

        # Expected factual world using a separate
        # Monte Carlo sample.
        factual_worlds = [
            self.graph.simulate(
                noise=self._sample_noise()
            ).values
            for _ in range(
                num_samples
                or self.config.num_samples
            )
        ]

        factual_means = {
            node: mean(
                world[node]
                for world in factual_worlds
            )
            for node in self.graph.nodes
        }

        ranked = []

        for spec in specifications:

            evaluation = stats[spec.name]

            reward = (
                evaluation.mean_outcomes[
                    reward_node
                ]
            )

            risk = (
                evaluation.mean_outcomes[
                    risk_node
                ]
                if risk_node is not None
                else 0.0
            )

            if (
                spec.intervention_type
                == InterventionType.HARD
            ):
                cost = self.intervention_cost(
                    factual_means,
                    spec,
                )
            else:
                cost = self._soft_intervention_cost(
                    spec
                )

            score = (
                reward
                - self.config.risk_penalty
                * risk
                - self.config.cost_penalty
                * cost
            )

            ranked.append(
                InterventionActionScore(
                    name=spec.name,
                    expected_reward=reward,
                    expected_risk=risk,
                    intervention_cost=cost,
                    score=score,
                )
            )

        ranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return InterventionRanking(
            best_action=ranked[0],
            ranked_actions=ranked,
        )

    def causal_effect_paths(
        self,
        intervention_node: str,
        outcome_node: str,
        max_paths: int = 100,
    ) -> List[List[str]]:
        """
        Find directed paths through which
        an intervention could affect an outcome.

        Paths indicate possible causal routes,
        not proof of a nonzero numerical effect.
        """

        self._require_node(intervention_node)
        self._require_node(outcome_node)

        return self.graph.causal_paths(
            source=intervention_node,
            target=outcome_node,
            max_paths=max_paths,
        )

    @staticmethod
    def export_report(
        result: InterventionResult,
        path: str | Path,
    ) -> None:
        """Save an intervention report as JSON."""

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                asdict(result),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":

    # -----------------------------------------
    # Example: Industrial Robot Safety
    # -----------------------------------------

    graph = CausalGraph()

    for node in (
        "robot_speed",
        "motor_temperature",
        "collision_risk",
        "task_progress",
        "reward",
    ):
        graph.add_node(node)

    graph.add_edge(
        "robot_speed",
        "motor_temperature",
        weight=0.6,
    )

    graph.add_edge(
        "robot_speed",
        "collision_risk",
        weight=0.4,
    )

    graph.add_edge(
        "robot_speed",
        "task_progress",
        weight=1.1,
    )

    graph.add_edge(
        "motor_temperature",
        "collision_risk",
        weight=0.1,
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
        noise_std=0.2,
    )

    graph.set_structural_equation(
        "motor_temperature",
        bias=25.0,
        noise_std=0.5,
    )

    graph.set_structural_equation(
        "collision_risk",
        noise_std=0.1,
    )

    graph.set_structural_equation(
        "task_progress",
        noise_std=0.1,
    )

    graph.set_structural_equation(
        "reward",
        noise_std=0.1,
    )

    causal_model = CausalModel(graph)

    engine = InterventionEngine(
        causal_model=causal_model,
        config=InterventionConfig(
            num_samples=500,
            random_seed=42,
            risk_penalty=2.0,
            cost_penalty=0.01,
        ),
        constraints=[
            InterventionConstraint(
                node="robot_speed",
                minimum=0.0,
                maximum=10.0,
            )
        ],
    )

    # Hard interventions.
    slow = InterventionSpecification(
        name="slow",
        intervention_type=(
            InterventionType.HARD
        ),
        values={
            "robot_speed": 2.0
        },
    )

    normal = InterventionSpecification(
        name="normal",
        values={
            "robot_speed": 5.0
        },
    )

    fast = InterventionSpecification(
        name="fast",
        values={
            "robot_speed": 8.0
        },
    )

    # Soft intervention:
    # Change how speed affects temperature.
    improved_cooling = (
        InterventionSpecification(
            name="improved_cooling",
            intervention_type=(
                InterventionType.SOFT
            ),
            weight_overrides={
                "robot_speed->motor_temperature": 0.3
            },
        )
    )

    factual = causal_model.observe()

    result = engine.apply(
        slow,
        factual_values=factual.values,
    )

    print("Hard intervention result:")
    print(result.values)

    print("\nCounterfactual effects:")
    print(result.effects)

    soft_result = engine.apply(
        improved_cooling,
        factual_values=factual.values,
    )

    print("\nSoft intervention result:")
    print(soft_result.values)

    specifications = [
        slow,
        normal,
        fast,
        improved_cooling,
    ]

    statistics = engine.compare(
        specifications=specifications,
        outcome_nodes=[
            "collision_risk",
            "reward",
        ],
        num_samples=500,
    )

    print("\nIntervention statistics:")

    for name, stat in statistics.items():
        print(
            name,
            stat.mean_outcomes,
        )

    ranking = engine.rank_actions(
        specifications=specifications,
        reward_node="reward",
        risk_node="collision_risk",
        num_samples=500,
    )

    print("\nBest intervention:")
    print(ranking.best_action.name)

    print("\nRanked interventions:")

    for action in ranking.ranked_actions:
        print(
            action.name,
            round(action.score, 4),
        )

    print("\nCausal paths:")

    print(
        engine.causal_effect_paths(
            "robot_speed",
            "collision_risk",
        )
    )

    engine.export_report(
        result,
        "artifacts/intervention_report.json",
    )
