
"""
NeuroWorld AI
File: neuroworld/causal/counterfactual_reasoner.py

Counterfactual Reasoning Engine

Features:
- Abduction-action-prediction reasoning
- Factual-world reconstruction
- Hard do-interventions
- Alternative action evaluation
- Individual counterfactual effects
- Multiple scenario comparison
- Target outcome optimization
- Risk-aware counterfactual ranking
- Causal explanation generation
- Monte Carlo counterfactual evaluation
- Common-random-number comparisons
- Counterfactual consistency validation
- JSON-compatible result export

Dependencies:
    Python 3.10+

Requires:
    neuroworld/causal/causal_graph.py
    neuroworld/causal/causal_model.py

Important:
    Counterfactual conclusions depend on the
    correctness of the supplied causal graph
    and structural equations.

    Custom structural functions must use
    additive exogenous noise to support
    automatic noise abduction.
"""

from __future__ import annotations

import json
import math

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from neuroworld.causal.causal_graph import (
    CausalGraph,
    CounterfactualResult,
)

from neuroworld.causal.causal_model import (
    CausalModel,
)


@dataclass
class CounterfactualReasonerConfig:
    num_monte_carlo_samples: int = 500

    risk_penalty: float = 1.0
    action_cost_weight: float = 0.0

    factual_tolerance: float = 1e-5

    maximize_outcome: bool = True

    random_seed: int = 42


@dataclass
class CounterfactualScenario:
    name: str
    interventions: Dict[str, float]

    description: str = ""
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class CounterfactualExplanation:
    scenario_name: str

    factual_values: Dict[str, float]
    counterfactual_values: Dict[str, float]

    interventions: Dict[str, float]

    changed_variables: Dict[str, float]

    causal_paths: Dict[
        str,
        List[List[str]],
    ]

    explanation: str


@dataclass
class CounterfactualScenarioResult:
    name: str

    interventions: Dict[str, float]

    outcome_value: float
    factual_outcome: float
    outcome_effect: float

    risk_value: float
    action_cost: float

    score: float

    values: Dict[str, float]
    effects: Dict[str, float]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class CounterfactualRanking:
    best_scenario: CounterfactualScenarioResult

    ranked_scenarios: List[
        CounterfactualScenarioResult
    ]

    target_outcome: str
    risk_node: Optional[str]


@dataclass
class MonteCarloCounterfactualResult:
    scenario_name: str
    outcome_node: str

    mean_effect: float
    std_effect: float

    mean_factual_outcome: float
    mean_counterfactual_outcome: float

    num_samples: int


class CounterfactualReasoner:
    """
    Structural counterfactual reasoning engine.

    Three stages:

    1. Abduction:
       Infer exogenous disturbances from
       the observed factual world.

    2. Action:
       Replace selected structural equations
       with interventions.

    3. Prediction:
       Recompute the world using the same
       exogenous disturbances.

    Supports individual and population-level
    counterfactual analysis.
    """

    def __init__(
        self,
        causal_model: CausalModel,
        config: Optional[
            CounterfactualReasonerConfig
        ] = None,
    ):
        self.causal_model = causal_model
        self.graph: CausalGraph = (
            causal_model.graph
        )

        self.config = (
            config
            or CounterfactualReasonerConfig()
        )

        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.num_monte_carlo_samples <= 0:
            raise ValueError(
                "num_monte_carlo_samples "
                "must be positive"
            )

        if (
            not math.isfinite(cfg.risk_penalty)
            or cfg.risk_penalty < 0
        ):
            raise ValueError(
                "risk_penalty must be finite "
                "and non-negative"
            )

        if (
            not math.isfinite(cfg.action_cost_weight)
            or cfg.action_cost_weight < 0
        ):
            raise ValueError(
                "action_cost_weight must be finite "
                "and non-negative"
            )

        if (
            not math.isfinite(cfg.factual_tolerance)
            or cfg.factual_tolerance < 0
        ):
            raise ValueError(
                "factual_tolerance must be finite "
                "and non-negative"
            )

    def _validate_node(
        self,
        node: str,
    ) -> None:
        if node not in self.graph.nodes:
            raise KeyError(
                f"Unknown causal variable: {node}"
            )

    def _validate_values(
        self,
        values: Mapping[str, float],
        require_complete: bool = False,
    ) -> Dict[str, float]:

        result = {}

        for node, value in values.items():
            self._validate_node(node)

            number = float(value)

            if not math.isfinite(number):
                raise ValueError(
                    f"Non-finite value for {node}"
                )

            result[node] = number

        if require_complete:
            if set(result) != set(self.graph.nodes):
                raise ValueError(
                    "A complete factual world "
                    "is required"
                )

        return result

    def _infer_noise(
        self,
        factual_values: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Infer additive exogenous disturbances.

        For each structural equation:

            observed = f(parents, 0) + noise

        Therefore:

            noise = observed - f(parents, 0)

        Requires complete factual observations.
        """

        factual = self._validate_values(
            factual_values,
            require_complete=True,
        )

        inferred = {}

        for node in self.graph.topological_sort():

            baseline = self.graph._evaluate_node(
                node=node,
                values=factual,
                noise=0.0,
            )

            inferred[node] = (
                factual[node] - baseline
            )

        return inferred

    def reconstruct_factual_world(
        self,
        factual_values: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Reconstruct factual observations
        from inferred exogenous noise.
        """

        factual = self._validate_values(
            factual_values,
            require_complete=True,
        )

        noise = self._infer_noise(factual)

        reconstructed = self.graph.simulate(
            noise=noise
        ).values

        for node in self.graph.nodes:
            if not math.isclose(
                reconstructed[node],
                factual[node],
                rel_tol=self.config.factual_tolerance,
                abs_tol=self.config.factual_tolerance,
            ):
                raise ValueError(
                    "Factual reconstruction failed "
                    f"for node {node}"
                )

        return reconstructed

    def reason(
        self,
        factual_values: Mapping[str, float],
        interventions: Mapping[str, float],
    ) -> CounterfactualResult:
        """
        Evaluate an individual counterfactual.

        Example:

            What if robot_speed had been 3
            instead of its observed value?
        """

        factual = self._validate_values(
            factual_values,
            require_complete=True,
        )

        intervention = self._validate_values(
            interventions
        )

        self.reconstruct_factual_world(
            factual
        )

        return self.causal_model.counterfactual(
            factual_values=factual,
            interventions=intervention,
        )

    def explain(
        self,
        factual_values: Mapping[str, float],
        scenario: CounterfactualScenario,
        max_paths: int = 100,
    ) -> CounterfactualExplanation:
        """
        Generate a structured causal explanation.

        Reports:
            - Interventions
            - Changed variables
            - Directed causal paths
            - Counterfactual differences
        """

        result = self.reason(
            factual_values=factual_values,
            interventions=scenario.interventions,
        )

        tolerance = self.config.factual_tolerance

        changed = {
            node: effect
            for node, effect in result.effects.items()
            if abs(effect) > tolerance
        }

        paths: Dict[
            str,
            List[List[str]],
        ] = {}

        for changed_node in changed:

            all_paths = []

            for intervention_node in (
                scenario.interventions
            ):
                if (
                    intervention_node
                    == changed_node
                ):
                    all_paths.append(
                        [intervention_node]
                    )

                elif self.graph.has_path(
                    intervention_node,
                    changed_node,
                ):
                    all_paths.extend(
                        self.graph.causal_paths(
                            intervention_node,
                            changed_node,
                            max_paths=max_paths,
                        )
                    )

            paths[changed_node] = (
                all_paths[:max_paths]
            )

        if not changed:
            explanation = (
                f"Scenario '{scenario.name}' "
                "produced no numerical changes "
                "above the configured tolerance."
            )
        else:
            parts = [
                f"Scenario '{scenario.name}' "
                "applies interventions:"
            ]

            for node, value in (
                scenario.interventions.items()
            ):
                parts.append(
                    f"{node} = {value:g}"
                )

            parts.append(
                "Model-predicted changes:"
            )

            for node, effect in changed.items():
                direction = (
                    "increased"
                    if effect > 0
                    else "decreased"
                )

                parts.append(
                    f"{node} {direction} by "
                    f"{abs(effect):.4f}"
                )

            explanation = " ".join(parts)

        return CounterfactualExplanation(
            scenario_name=scenario.name,
            factual_values=dict(
                result.factual_values
            ),
            counterfactual_values=dict(
                result.counterfactual_values
            ),
            interventions=dict(
                scenario.interventions
            ),
            changed_variables=changed,
            causal_paths=paths,
            explanation=explanation,
        )

    @staticmethod
    def _action_cost(
        factual_values: Mapping[str, float],
        interventions: Mapping[str, float],
    ) -> float:
        """
        Simple squared intervention magnitude.

        Uses factual values as the baseline.
        """

        return sum(
            (
                value - factual_values[node]
            ) ** 2
            for node, value in interventions.items()
        )

    def compare_scenarios(
        self,
        factual_values: Mapping[str, float],
        scenarios: Sequence[
            CounterfactualScenario
        ],
        target_outcome: str,
        risk_node: Optional[str] = None,
        maximize: Optional[bool] = None,
    ) -> CounterfactualRanking:
        """
        Evaluate and rank alternative worlds.

        Score:
            outcome_utility
            - risk_penalty * risk_value
            - action_cost_weight * action_cost

        outcome_utility:
            + outcome if maximizing
            - outcome if minimizing

        The risk variable must use a consistent
        interpretation where larger means worse.
        """

        if not scenarios:
            raise ValueError(
                "At least one scenario required"
            )

        self._validate_node(target_outcome)

        if risk_node is not None:
            self._validate_node(risk_node)

        factual = self._validate_values(
            factual_values,
            require_complete=True,
        )

        if maximize is None:
            maximize = (
                self.config.maximize_outcome
            )

        names = [s.name for s in scenarios]

        if len(names) != len(set(names)):
            raise ValueError(
                "Scenario names must be unique"
            )

        ranked = []

        for scenario in scenarios:

            result = self.reason(
                factual_values=factual,
                interventions=(
                    scenario.interventions
                ),
            )

            outcome = (
                result.counterfactual_values[
                    target_outcome
                ]
            )

            factual_outcome = (
                factual[target_outcome]
            )

            risk = (
                result.counterfactual_values[
                    risk_node
                ]
                if risk_node is not None
                else 0.0
            )

            cost = self._action_cost(
                factual,
                scenario.interventions,
            )

            utility = (
                outcome
                if maximize
                else -outcome
            )

            score = (
                utility
                - self.config.risk_penalty * risk
                - self.config.action_cost_weight
                * cost
            )

            ranked.append(
                CounterfactualScenarioResult(
                    name=scenario.name,
                    interventions=dict(
                        scenario.interventions
                    ),
                    outcome_value=outcome,
                    factual_outcome=(
                        factual_outcome
                    ),
                    outcome_effect=(
                        outcome - factual_outcome
                    ),
                    risk_value=risk,
                    action_cost=cost,
                    score=score,
                    values=dict(
                        result.counterfactual_values
                    ),
                    effects=dict(
                        result.effects
                    ),
                    metadata=dict(
                        scenario.metadata
                    ),
                )
            )

        ranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return CounterfactualRanking(
            best_scenario=ranked[0],
            ranked_scenarios=ranked,
            target_outcome=target_outcome,
            risk_node=risk_node,
        )

    def monte_carlo_counterfactual(
        self,
        scenario: CounterfactualScenario,
        outcome_node: str,
        num_samples: Optional[int] = None,
    ) -> MonteCarloCounterfactualResult:
        """
        Population-level intervention effect.

        Samples factual worlds and evaluates
        their counterfactual alternatives using
        identical exogenous disturbances.

        This estimates a model-implied average
        intervention effect.

        It does not estimate an individual
        counterfactual for a particular person
        or observed robot trajectory.
        """

        self._validate_node(outcome_node)

        intervention = self._validate_values(
            scenario.interventions
        )

        if num_samples is None:
            num_samples = (
                self.config.num_monte_carlo_samples
            )

        if num_samples < 2:
            raise ValueError(
                "At least two samples required"
            )

        factual_outcomes = []
        counterfactual_outcomes = []
        effects = []

        # Independent local random stream.
        import random

        rng = random.Random(
            self.config.random_seed
        )

        for _ in range(num_samples):

            noise = {
                node: rng.gauss(
                    0.0,
                    self.graph._noise_std[node],
                )
                for node in self.graph.nodes
            }

            factual = self.graph.simulate(
                noise=noise
            )

            counterfactual = self.graph.simulate(
                interventions=intervention,
                noise=noise,
            )

            factual_value = (
                factual.values[outcome_node]
            )

            alternative_value = (
                counterfactual.values[
                    outcome_node
                ]
            )

            factual_outcomes.append(
                factual_value
            )

            counterfactual_outcomes.append(
                alternative_value
            )

            effects.append(
                alternative_value - factual_value
            )

        from statistics import (
            mean,
            stdev,
        )

        return MonteCarloCounterfactualResult(
            scenario_name=scenario.name,
            outcome_node=outcome_node,
            mean_effect=mean(effects),
            std_effect=stdev(effects),
            mean_factual_outcome=mean(
                factual_outcomes
            ),
            mean_counterfactual_outcome=mean(
                counterfactual_outcomes
            ),
            num_samples=num_samples,
        )

    def evaluate_action_candidates(
        self,
        factual_values: Mapping[str, float],
        action_node: str,
        candidate_values: Sequence[float],
        target_outcome: str,
        risk_node: Optional[str] = None,
        maximize: Optional[bool] = None,
    ) -> CounterfactualRanking:
        """
        Evaluate alternative action values.

        Example:
            action_node = "robot_speed"
            candidate_values = [1, 3, 5, 7]
        """

        self._validate_node(action_node)

        if not candidate_values:
            raise ValueError(
                "candidate_values cannot be empty"
            )

        scenarios = [
            CounterfactualScenario(
                name=f"{action_node}_{index}",
                interventions={
                    action_node: float(value)
                },
                metadata={
                    "candidate_value": float(value)
                },
            )
            for index, value in enumerate(
                candidate_values
            )
        ]

        return self.compare_scenarios(
            factual_values=factual_values,
            scenarios=scenarios,
            target_outcome=target_outcome,
            risk_node=risk_node,
            maximize=maximize,
        )

    def counterfactual_effects(
        self,
        factual_values: Mapping[str, float],
        interventions: Mapping[str, float],
        outcome_nodes: Optional[
            Sequence[str]
        ] = None,
    ) -> Dict[str, float]:
        """
        Return counterfactual effects for
        selected outcome variables.
        """

        result = self.reason(
            factual_values,
            interventions,
        )

        if outcome_nodes is None:
            return dict(result.effects)

        for node in outcome_nodes:
            self._validate_node(node)

        return {
            node: result.effects[node]
            for node in outcome_nodes
        }

    def export_explanation(
        self,
        explanation: CounterfactualExplanation,
        path: str | Path,
    ) -> None:
        """
        Export a counterfactual explanation
        as a JSON file.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                asdict(explanation),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":

    # -----------------------------------------
    # Example: Robot Collision Avoidance
    # -----------------------------------------

    graph = CausalGraph()

    for node in (
        "robot_speed",
        "obstacle_density",
        "collision_risk",
        "task_progress",
        "reward",
    ):
        graph.add_node(node)

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
        noise_std=0.2,
    )

    graph.set_structural_equation(
        "obstacle_density",
        bias=2.0,
        noise_std=0.1,
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

    reasoner = CounterfactualReasoner(
        causal_model=causal_model,
        config=CounterfactualReasonerConfig(
            num_monte_carlo_samples=500,
            risk_penalty=1.0,
            action_cost_weight=0.01,
        ),
    )

    # Observe a factual world.
    factual = causal_model.observe()

    print("Factual world:")
    print(factual.values)

    # Individual counterfactual.
    alternative = reasoner.reason(
        factual_values=factual.values,
        interventions={
            "robot_speed": 2.0
        },
    )

    print("\nCounterfactual world:")
    print(
        alternative.counterfactual_values
    )

    print("\nCounterfactual effects:")
    print(alternative.effects)

    # Explain why outcomes changed.
    scenario = CounterfactualScenario(
        name="slow_robot",
        interventions={
            "robot_speed": 2.0
        },
        description=(
            "Reduce robot speed to avoid "
            "potential collisions"
        ),
    )

    explanation = reasoner.explain(
        factual_values=factual.values,
        scenario=scenario,
    )

    print("\nExplanation:")
    print(explanation.explanation)

    # Rank candidate robot actions.
    ranking = (
        reasoner.evaluate_action_candidates(
            factual_values=factual.values,
            action_node="robot_speed",
            candidate_values=[
                1.0,
                2.0,
                3.0,
                5.0,
                7.0,
            ],
            target_outcome="reward",
            risk_node="collision_risk",
        )
    )

    print("\nBest counterfactual action:")
    print(
        ranking.best_scenario.name
    )

    print("\nRanked actions:")

    for result in ranking.ranked_scenarios:
        print(
            result.name,
            "score:",
            round(result.score, 4),
            "risk:",
            round(result.risk_value, 4),
        )

    # Monte Carlo population analysis.
    monte_carlo = (
        reasoner.monte_carlo_counterfactual(
            scenario=scenario,
            outcome_node="collision_risk",
            num_samples=500,
        )
    )

    print("\nMonte Carlo mean effect:")
    print(monte_carlo.mean_effect)

    # Export explanation.
    reasoner.export_explanation(
        explanation,
        "artifacts/counterfactual_explanation.json",
    )
