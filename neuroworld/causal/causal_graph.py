
"""
NeuroWorld AI
File: neuroworld/causal/causal_graph.py

Causal Graph Intelligence Engine

Features:
- Directed acyclic causal graphs (DAG)
- Node and causal-edge management
- Cycle detection
- Topological sorting
- Parent, child, ancestor, descendant queries
- Directed causal path discovery
- Markov blanket
- D-separation via ancestral moralization
- Structural causal model (SCM) simulation
- Hard do-interventions
- Abduction-action-prediction counterfactuals
- Graph export/import using JSON
- Graph validation and visualization export
- No external graph-library dependency

Important:
    Graph edges encode assumed causal relationships.
    Observational correlations alone do not
    establish causal direction.

Dependencies:
    Python 3.10+
"""

from __future__ import annotations

import json
import math
import random

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)


@dataclass
class CausalNode:
    name: str
    description: str = ""
    category: str = "state"
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class CausalEdge:
    source: str
    target: str

    # Optional linear SCM coefficient.
    weight: float = 1.0

    description: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class CausalSimulationResult:
    values: Dict[str, float]
    interventions: Dict[str, float]
    topological_order: List[str]


@dataclass
class CounterfactualResult:
    factual_values: Dict[str, float]
    counterfactual_values: Dict[str, float]
    interventions: Dict[str, float]
    effects: Dict[str, float]


StructuralFunction = Callable[
    [Dict[str, float], float],
    float,
]


class CausalGraph:
    """
    Directed acyclic graph for causal modeling.

    Supports:
        - Causal structure
        - Graph reasoning
        - Linear structural equations
        - Custom structural equations
        - Hard interventions
        - Counterfactual evaluation

    Default structural equation:

        X_i = bias_i
              + sum_j(weight_ji * X_j)
              + noise_i

    For nonlinear relationships, register a
    custom structural function with:

        set_structural_function(node, function)

    Custom function signature:

        function(parent_values, noise) -> float
    """

    def __init__(self):
        self.nodes: Dict[
            str, CausalNode
        ] = {}

        self.edges: Dict[
            Tuple[str, str], CausalEdge
        ] = {}

        self._parents: Dict[
            str, Set[str]
        ] = {}

        self._children: Dict[
            str, Set[str]
        ] = {}

        self._biases: Dict[
            str, float
        ] = {}

        self._noise_std: Dict[
            str, float
        ] = {}

        self._functions: Dict[
            str, StructuralFunction
        ] = {}

    def add_node(
        self,
        name: str,
        description: str = "",
        category: str = "state",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """Add a causal variable."""

        if not isinstance(name, str) or not name:
            raise ValueError(
                "Node name must be non-empty"
            )

        if name in self.nodes:
            raise ValueError(
                f"Node already exists: {name}"
            )

        self.nodes[name] = CausalNode(
            name=name,
            description=description,
            category=category,
            metadata=dict(metadata or {}),
        )

        self._parents[name] = set()
        self._children[name] = set()

        self._biases[name] = 0.0
        self._noise_std[name] = 0.0

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        description: str = "",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """Add a directed causal relationship."""

        self._require_node(source)
        self._require_node(target)

        if source == target:
            raise ValueError(
                "Self-loops are not allowed"
            )

        if not math.isfinite(weight):
            raise ValueError(
                "Edge weight must be finite"
            )

        key = (source, target)

        if key in self.edges:
            raise ValueError(
                "Causal edge already exists"
            )

        # Adding source -> target creates a cycle
        # if target already reaches source.
        if self.has_path(target, source):
            raise ValueError(
                f"Adding {source} -> {target} "
                "would create a cycle"
            )

        self.edges[key] = CausalEdge(
            source=source,
            target=target,
            weight=float(weight),
            description=description,
            metadata=dict(metadata or {}),
        )

        self._parents[target].add(source)
        self._children[source].add(target)

    def remove_edge(
        self,
        source: str,
        target: str,
    ) -> None:
        key = (source, target)

        if key not in self.edges:
            raise KeyError(
                f"Unknown edge: {source} -> {target}"
            )

        del self.edges[key]

        self._parents[target].remove(source)
        self._children[source].remove(target)

    def remove_node(
        self,
        name: str,
    ) -> None:
        self._require_node(name)

        for parent in list(
            self._parents[name]
        ):
            self.remove_edge(parent, name)

        for child in list(
            self._children[name]
        ):
            self.remove_edge(name, child)

        del self.nodes[name]
        del self._parents[name]
        del self._children[name]

        self._biases.pop(name, None)
        self._noise_std.pop(name, None)
        self._functions.pop(name, None)

    def _require_node(
        self,
        name: str,
    ) -> None:
        if name not in self.nodes:
            raise KeyError(
                f"Unknown causal node: {name}"
            )

    def parents(
        self,
        node: str,
    ) -> Set[str]:
        self._require_node(node)
        return set(self._parents[node])

    def children(
        self,
        node: str,
    ) -> Set[str]:
        self._require_node(node)
        return set(self._children[node])

    def ancestors(
        self,
        node: str,
    ) -> Set[str]:
        """All upstream causal ancestors."""

        self._require_node(node)

        visited: Set[str] = set()
        stack = list(self._parents[node])

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)

            stack.extend(
                self._parents[current]
            )

        return visited

    def descendants(
        self,
        node: str,
    ) -> Set[str]:
        """All downstream causal descendants."""

        self._require_node(node)

        visited: Set[str] = set()
        stack = list(self._children[node])

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)

            stack.extend(
                self._children[current]
            )

        return visited

    def has_path(
        self,
        source: str,
        target: str,
    ) -> bool:
        """Check directed reachability."""

        self._require_node(source)
        self._require_node(target)

        if source == target:
            return True

        visited = {source}
        stack = [source]

        while stack:
            current = stack.pop()

            for child in self._children[current]:
                if child == target:
                    return True

                if child not in visited:
                    visited.add(child)
                    stack.append(child)

        return False

    def causal_paths(
        self,
        source: str,
        target: str,
        max_paths: int = 1000,
    ) -> List[List[str]]:
        """
        Enumerate directed causal paths.

        max_paths limits potentially exponential
        path enumeration.
        """

        self._require_node(source)
        self._require_node(target)

        if max_paths <= 0:
            raise ValueError(
                "max_paths must be positive"
            )

        results: List[List[str]] = []

        stack = [
            (source, [source])
        ]

        while stack and len(results) < max_paths:
            current, path = stack.pop()

            if current == target:
                results.append(path)
                continue

            for child in sorted(
                self._children[current],
                reverse=True,
            ):
                if child not in path:
                    stack.append(
                        (child, path + [child])
                    )

        return results

    def topological_sort(
        self,
    ) -> List[str]:
        """Return a valid causal evaluation order."""

        indegree = {
            node: len(self._parents[node])
            for node in self.nodes
        }

        queue = deque(
            sorted(
                node
                for node, degree in indegree.items()
                if degree == 0
            )
        )

        result: List[str] = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for child in sorted(
                self._children[current]
            ):
                indegree[child] -= 1

                if indegree[child] == 0:
                    queue.append(child)

        if len(result) != len(self.nodes):
            raise ValueError(
                "Graph contains a directed cycle"
            )

        return result

    def markov_blanket(
        self,
        node: str,
    ) -> Set[str]:
        """
        Markov blanket:
        parents, children, and co-parents
        of children.
        """

        self._require_node(node)

        blanket = (
            self.parents(node)
            | self.children(node)
        )

        for child in self._children[node]:
            blanket.update(
                self._parents[child]
            )

        blanket.discard(node)

        return blanket

    def d_separated(
        self,
        x: Iterable[str],
        y: Iterable[str],
        conditioned_on: Iterable[str] = (),
    ) -> bool:
        """
        Test DAG d-separation using the
        ancestral moral graph method.

        1. Take ancestors of X, Y, and Z.
        2. Build their induced subgraph.
        3. Moralize by connecting co-parents.
        4. Remove conditioned nodes Z.
        5. Test undirected connectivity.

        Returns True if X and Y are d-separated
        given Z in this DAG.
        """

        x_set = set(x)
        y_set = set(y)
        z_set = set(conditioned_on)

        for node in x_set | y_set | z_set:
            self._require_node(node)

        if not x_set or not y_set:
            raise ValueError(
                "X and Y must be non-empty"
            )

        if (
            x_set & y_set
            or x_set & z_set
            or y_set & z_set
        ):
            raise ValueError(
                "X, Y, and Z must be disjoint"
            )

        relevant = x_set | y_set | z_set

        ancestral_nodes = set(relevant)

        for node in relevant:
            ancestral_nodes.update(
                self.ancestors(node)
            )

        undirected = {
            node: set()
            for node in ancestral_nodes
        }

        # Convert directed edges into
        # undirected edges.
        for source, target in self.edges:
            if (
                source in ancestral_nodes
                and target in ancestral_nodes
            ):
                undirected[source].add(target)
                undirected[target].add(source)

        # Moralize: connect co-parents.
        for node in ancestral_nodes:
            parents = sorted(
                self._parents[node]
                & ancestral_nodes
            )

            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    first = parents[i]
                    second = parents[j]

                    undirected[first].add(second)
                    undirected[second].add(first)

        # Conditioning removes Z from
        # the moralized graph.
        visited = set(z_set)

        queue = deque(
            sorted(x_set)
        )

        visited.update(x_set)

        while queue:
            current = queue.popleft()

            if current in y_set:
                return False

            for neighbor in undirected[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return True

    def set_structural_equation(
        self,
        node: str,
        bias: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        """
        Configure a linear structural equation.

        X = bias + sum(weight * parent) + noise
        """

        self._require_node(node)

        if not math.isfinite(bias):
            raise ValueError(
                "bias must be finite"
            )

        if (
            not math.isfinite(noise_std)
            or noise_std < 0
        ):
            raise ValueError(
                "noise_std must be finite "
                "and non-negative"
            )

        self._biases[node] = float(bias)
        self._noise_std[node] = float(noise_std)

    def set_structural_function(
        self,
        node: str,
        function: StructuralFunction,
    ) -> None:
        """
        Register a nonlinear structural equation.

        function(parent_values, noise) -> value

        The function must be deterministic for
        fixed parent values and noise to support
        meaningful counterfactual replay.
        """

        self._require_node(node)

        if not callable(function):
            raise TypeError(
                "function must be callable"
            )

        self._functions[node] = function

    def _evaluate_node(
        self,
        node: str,
        values: Dict[str, float],
        noise: float,
    ) -> float:
        parent_values = {
            parent: values[parent]
            for parent in self._parents[node]
        }

        if node in self._functions:
            result = self._functions[node](
                parent_values,
                noise,
            )
        else:
            result = self._biases[node] + noise

            for parent in self._parents[node]:
                edge = self.edges[
                    (parent, node)
                ]

                result += (
                    edge.weight
                    * parent_values[parent]
                )

        result = float(result)

        if not math.isfinite(result):
            raise ValueError(
                f"Non-finite value for node {node}"
            )

        return result

    def _validate_assignments(
        self,
        assignments: Mapping[str, float],
    ) -> Dict[str, float]:
        validated = {}

        for node, value in assignments.items():
            self._require_node(node)

            value = float(value)

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite assignment: {node}"
                )

            validated[node] = value

        return validated

    def sample_noise(
        self,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Sample exogenous disturbances.

        Uses a local random generator.
        """

        rng = random.Random(seed)

        return {
            node: rng.gauss(
                0.0,
                self._noise_std[node],
            )
            for node in self.nodes
        }

    def simulate(
        self,
        interventions: Optional[
            Mapping[str, float]
        ] = None,
        noise: Optional[
            Mapping[str, float]
        ] = None,
        seed: Optional[int] = None,
    ) -> CausalSimulationResult:
        """
        Evaluate the structural causal model.

        interventions:
            Hard do-interventions.

        Example:
            do(temperature=100)

        Hard interventions replace the target
        node's structural equation.

        noise:
            Explicit exogenous disturbances.
            Missing entries default to zero.

        If noise is None, disturbances are
        sampled using sample_noise().
        """

        interventions = self._validate_assignments(
            interventions or {}
        )

        if noise is None:
            disturbances = self.sample_noise(seed)
        else:
            disturbances = self._validate_assignments(
                noise
            )

        order = self.topological_sort()

        values: Dict[str, float] = {}

        for node in order:
            if node in interventions:
                values[node] = interventions[node]
                continue

            values[node] = self._evaluate_node(
                node=node,
                values=values,
                noise=disturbances.get(node, 0.0),
            )

        return CausalSimulationResult(
            values=values,
            interventions=dict(interventions),
            topological_order=order,
        )

    def intervene(
        self,
        interventions: Mapping[str, float],
        noise: Optional[
            Mapping[str, float]
        ] = None,
        seed: Optional[int] = None,
    ) -> CausalSimulationResult:
        """
        Apply do-interventions and simulate.
        """

        return self.simulate(
            interventions=interventions,
            noise=noise,
            seed=seed,
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
        Compute a counterfactual using
        abduction-action-prediction.

        Abduction:
            Infer exogenous noise from a complete
            factual assignment.

        Action:
            Replace selected structural equations
            with do-interventions.

        Prediction:
            Re-evaluate the model with the same
            inferred exogenous disturbances.

        Requirements:
            - Complete factual assignment
            - Acyclic causal graph
            - Deterministic structural functions
            - Additive exogenous noise

        This is a model-based counterfactual,
        not an independently verified causal fact.
        """

        factual = self._validate_assignments(
            factual_values
        )

        interventions = self._validate_assignments(
            interventions
        )

        if set(factual) != set(self.nodes):
            missing = set(self.nodes) - set(factual)
            extra = set(factual) - set(self.nodes)

            raise ValueError(
                "A complete factual assignment "
                "is required. "
                f"Missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        if noise is None:
            # For additive-noise SCMs:
            #
            # observed_value =
            #     structural_function(parents, 0)
            #     + exogenous_noise
            #
            # Custom structural functions must
            # follow this additive-noise convention.
            inferred_noise = {}

            for node in self.topological_sort():
                baseline = self._evaluate_node(
                    node=node,
                    values=factual,
                    noise=0.0,
                )

                inferred_noise[node] = (
                    factual[node] - baseline
                )
        else:
            inferred_noise = (
                self._validate_assignments(noise)
            )

            if set(inferred_noise) != set(self.nodes):
                raise ValueError(
                    "Explicit counterfactual noise "
                    "must cover every node"
                )

            # Ensure supplied disturbances reproduce
            # the factual observations.
            reproduced = self.simulate(
                noise=inferred_noise
            ).values

            for node in self.nodes:
                if not math.isclose(
                    reproduced[node],
                    factual[node],
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                ):
                    raise ValueError(
                        "Provided noise does not "
                        "reproduce factual values"
                    )

        simulated = self.simulate(
            interventions=interventions,
            noise=inferred_noise,
        )

        effects = {
            node: (
                simulated.values[node]
                - factual[node]
            )
            for node in self.nodes
        }

        return CounterfactualResult(
            factual_values=dict(factual),
            counterfactual_values=simulated.values,
            interventions=dict(interventions),
            effects=effects,
        )

    def adjacency_matrix(
        self,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Return weighted adjacency matrix.

        Matrix[i][j] represents the edge
        from node i to node j.
        """

        names = sorted(self.nodes)

        index = {
            name: i
            for i, name in enumerate(names)
        }

        matrix = [
            [0.0] * len(names)
            for _ in names
        ]

        for edge in self.edges.values():
            matrix[
                index[edge.source]
            ][
                index[edge.target]
            ] = edge.weight

        return names, matrix

    def to_dot(self) -> str:
        """
        Export Graphviz DOT source.

        Can be rendered externally using Graphviz.
        """

        lines = [
            "digraph CausalGraph {",
            "    rankdir=LR;",
        ]

        for node in sorted(self.nodes):
            lines.append(
                f"    {json.dumps(node)};"
            )

        for edge in sorted(
            self.edges.values(),
            key=lambda e: (e.source, e.target),
        ):
            lines.append(
                "    "
                f"{json.dumps(edge.source)} -> "
                f"{json.dumps(edge.target)} "
                f'[label="{edge.weight:g}"];'
            )

        lines.append("}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize graph structure and linear SCM
        parameters.

        Custom Python structural functions are
        not serializable.
        """

        if self._functions:
            raise ValueError(
                "Custom structural functions "
                "cannot be serialized to JSON"
            )

        return {
            "nodes": [
                {
                    "name": node.name,
                    "description": node.description,
                    "category": node.category,
                    "metadata": node.metadata,
                    "bias": self._biases[node.name],
                    "noise_std": (
                        self._noise_std[node.name]
                    ),
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "description": edge.description,
                    "metadata": edge.metadata,
                }
                for edge in self.edges.values()
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "CausalGraph":
        """Restore a graph from serialized data."""

        graph = cls()

        for node in data.get("nodes", []):
            graph.add_node(
                name=node["name"],
                description=node.get(
                    "description", ""
                ),
                category=node.get(
                    "category", "state"
                ),
                metadata=node.get(
                    "metadata", {}
                ),
            )

            graph.set_structural_equation(
                node=node["name"],
                bias=node.get("bias", 0.0),
                noise_std=node.get(
                    "noise_std", 0.0
                ),
            )

        for edge in data.get("edges", []):
            graph.add_edge(
                source=edge["source"],
                target=edge["target"],
                weight=edge.get("weight", 1.0),
                description=edge.get(
                    "description", ""
                ),
                metadata=edge.get(
                    "metadata", {}
                ),
            )

        return graph

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save causal graph as JSON."""

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "CausalGraph":
        """Load a causal graph from JSON."""

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        return cls.from_dict(data)

    def summary(self) -> Dict[str, Any]:
        """Return graph statistics."""

        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "topological_order": (
                self.topological_sort()
            ),
            "root_nodes": sorted(
                node
                for node in self.nodes
                if not self._parents[node]
            ),
            "leaf_nodes": sorted(
                node
                for node in self.nodes
                if not self._children[node]
            ),
        }


if __name__ == "__main__":

    graph = CausalGraph()

    # Example: industrial machine safety.
    graph.add_node(
        "machine_load",
        category="action",
    )

    graph.add_node(
        "temperature",
        category="sensor",
    )

    graph.add_node(
        "vibration",
        category="sensor",
    )

    graph.add_node(
        "failure_risk",
        category="outcome",
    )

    # Causal relationships.
    graph.add_edge(
        "machine_load",
        "temperature",
        weight=0.8,
    )

    graph.add_edge(
        "machine_load",
        "vibration",
        weight=0.5,
    )

    graph.add_edge(
        "temperature",
        "failure_risk",
        weight=0.6,
    )

    graph.add_edge(
        "vibration",
        "failure_risk",
        weight=0.4,
    )

    graph.set_structural_equation(
        "machine_load",
        bias=50.0,
    )

    graph.set_structural_equation(
        "temperature",
        bias=20.0,
    )

    graph.set_structural_equation(
        "vibration",
        bias=2.0,
    )

    graph.set_structural_equation(
        "failure_risk",
        bias=0.0,
    )

    print("Graph summary:")
    print(graph.summary())

    print("\nTopological order:")
    print(graph.topological_sort())

    print("\nCausal paths:")
    print(
        graph.causal_paths(
            "machine_load",
            "failure_risk",
        )
    )

    print("\nFactual simulation:")

    factual = graph.simulate(
        noise={
            node: 0.0
            for node in graph.nodes
        }
    )

    print(factual.values)

    print("\nIntervention: reduce machine load")

    intervention = graph.intervene(
        {"machine_load": 30.0},
        noise={
            node: 0.0
            for node in graph.nodes
        },
    )

    print(intervention.values)

    print("\nCounterfactual:")

    counterfactual = graph.counterfactual(
        factual_values=factual.values,
        interventions={
            "machine_load": 30.0
        },
    )

    print(
        "Effects:",
        counterfactual.effects,
    )

    print("\nD-separation test:")

    print(
        graph.d_separated(
            {"temperature"},
            {"vibration"},
            {"machine_load"},
        )
    )

    graph.save(
        "artifacts/causal_graph.json"
    )

    restored = CausalGraph.load(
        "artifacts/causal_graph.json"
    )

    print("\nRestored graph:")
    print(restored.summary())

    print("\nGraphviz DOT:")
    print(restored.to_dot())
```
