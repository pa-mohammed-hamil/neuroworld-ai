
```python
#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/build_trajectories.py

Generate navigation trajectory datasets from simulated worlds.

Features:
    - Multiple scenario types
    - Deterministic experiment seeds
    - Multiple robots per scenario
    - Navigation-driven data collection
    - State/action/next-state transitions
    - Rewards and termination labels
    - Collision and goal metadata
    - Train/validation/test splitting
    - Compressed NumPy dataset export
    - JSON metadata and summaries
    - Command-line interface

Output:
    data/trajectories/
        train.npz
        val.npz
        test.npz
        metadata.json
        summary.json

Dependencies:
    numpy

Project dependencies:
    neuroworld.environment.scenario
    neuroworld.environment.simulator

Example:
    python scripts/build_trajectories.py \
        --episodes 100 \
        --scenario random \
        --robots 2 \
        --max-steps 300 \
        --output data/trajectories
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


# Allow execution directly from the scripts directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from neuroworld.environment.scenario import (
    Scenario,
    ScenarioConfig,
)

from neuroworld.environment.simulator import (
    SimulatorConfig,
    SimulatorStatus,
    WorldSimulator,
)


LOGGER = logging.getLogger("neuroworld.trajectories")

STATE_DIM = 8
ACTION_DIM = 2

STATE_FEATURES = [
    "x",
    "y",
    "sin_yaw",
    "cos_yaw",
    "linear_velocity",
    "angular_velocity",
    "goal_dx",
    "goal_dy",
]

ACTION_FEATURES = [
    "linear_velocity_command",
    "angular_velocity_command",
]


@dataclass
class TrajectoryBuilderConfig:
    episodes: int = 100
    scenario_type: str = "random"

    world_width: float = 10.0
    world_height: float = 10.0
    map_resolution: float = 0.1

    num_robots: int = 1
    num_obstacles: int = 10

    max_steps: int = 300
    timestep: float = 0.1

    seed: int = 42

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    goal_reward: float = 10.0
    collision_penalty: float = 5.0
    step_penalty: float = 0.01
    progress_reward_scale: float = 1.0

    stop_on_collision: bool = True

    output_dir: str = "data/trajectories"

    save_episode_json: bool = False
    log_interval: int = 10


@dataclass
class Transition:
    state: list[float]
    action: list[float]
    next_state: list[float]

    reward: float
    terminated: bool
    truncated: bool

    episode_id: int
    robot_id: str
    step: int

    collision: bool
    goal_reached: bool


@dataclass
class EpisodeSummary:
    episode_id: int
    scenario_type: str
    seed: int

    steps: int
    transitions: int

    total_reward: float
    goals_reached: int
    collisions: int

    success: bool
    termination_reason: str

    robot_ids: list[str] = field(
        default_factory=list
    )


class TrajectoryDatasetBuilder:
    """
    Generate world-model training transitions.

    State representation:
        [x, y, sin(yaw), cos(yaw),
         linear_velocity, angular_velocity,
         goal_dx, goal_dy]

    Action representation:
        [linear_velocity_command,
         angular_velocity_command]

    Transitions are recorded separately
    for each robot, but episode IDs are
    kept together during dataset splitting.
    """

    def __init__(
        self,
        config: Optional[
            TrajectoryBuilderConfig
        ] = None,
    ) -> None:
        self.config = (
            config or TrajectoryBuilderConfig()
        )

        self._validate_config()

        self.transitions: list[
            Transition
        ] = []

        self.episode_summaries: list[
            EpisodeSummary
        ] = []

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.episodes <= 0:
            raise ValueError(
                "episodes must be positive"
            )

        if cfg.num_robots <= 0:
            raise ValueError(
                "num_robots must be positive"
            )

        if cfg.num_obstacles < 0:
            raise ValueError(
                "num_obstacles cannot be negative"
            )

        if cfg.max_steps <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

        if cfg.log_interval <= 0:
            raise ValueError(
                "log_interval must be positive"
            )

        for name in (
            "world_width",
            "world_height",
            "map_resolution",
            "timestep",
        ):
            value = getattr(cfg, name)

            if (
                not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be positive "
                    "and finite"
                )

        ratios = (
            cfg.train_ratio,
            cfg.val_ratio,
            cfg.test_ratio,
        )

        if any(
            not math.isfinite(value)
            or value < 0
            for value in ratios
        ):
            raise ValueError(
                "Split ratios must be "
                "nonnegative and finite"
            )

        if not math.isclose(
            sum(ratios),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Train, validation and test "
                "ratios must sum to 1"
            )

        if not cfg.output_dir.strip():
            raise ValueError(
                "output_dir cannot be empty"
            )

    def _create_scenario(
        self,
        episode_seed: int,
    ) -> Scenario:
        cfg = self.config

        scenario = Scenario(
            ScenarioConfig(
                name=(
                    f"NeuroWorld Episode "
                    f"{episode_seed}"
                ),
                scenario_type=(
                    cfg.scenario_type
                ),
                world_width=(
                    cfg.world_width
                ),
                world_height=(
                    cfg.world_height
                ),
                map_resolution=(
                    cfg.map_resolution
                ),
                num_robots=(
                    cfg.num_robots
                ),
                num_obstacles=(
                    cfg.num_obstacles
                ),
                seed=episode_seed,
                simulation_dt=(
                    cfg.timestep
                ),
            )
        )

        scenario.reset()

        return scenario

    def _create_simulator(
        self,
        scenario: Scenario,
        episode_seed: int,
    ) -> WorldSimulator:
        cfg = self.config

        simulator = WorldSimulator(
            scenario=scenario,
            config=SimulatorConfig(
                timestep=cfg.timestep,
                max_steps=cfg.max_steps,
                enable_navigation=True,
                enable_collision_detection=True,
                stop_on_collision=(
                    cfg.stop_on_collision
                ),
                enable_dynamic_obstacles=True,
                enable_rendering=False,
                capture_frames=False,
                seed=episode_seed,
            ),
        )

        simulator.reset(
            regenerate_scenario=False
        )

        return simulator

    @staticmethod
    def _goals_by_robot(
        scenario: Scenario,
    ) -> dict[str, tuple[float, float]]:
        return {
            goal.assigned_robot: (
                goal.x,
                goal.y,
            )
            for goal in scenario.state.goals
            if goal.assigned_robot is not None
        }

    @staticmethod
    def _encode_state(
        robot: Any,
        goal: Optional[
            tuple[float, float]
        ],
    ) -> list[float]:
        if goal is None:
            goal_dx = 0.0
            goal_dy = 0.0
        else:
            goal_dx = goal[0] - robot.x
            goal_dy = goal[1] - robot.y

        return [
            float(robot.x),
            float(robot.y),
            math.sin(float(robot.yaw)),
            math.cos(float(robot.yaw)),
            float(robot.linear_velocity),
            float(robot.angular_velocity),
            float(goal_dx),
            float(goal_dy),
        ]

    @staticmethod
    def _distance_to_goal(
        robot: Any,
        goal: Optional[
            tuple[float, float]
        ],
    ) -> float:
        if goal is None:
            return 0.0

        return math.hypot(
            goal[0] - robot.x,
            goal[1] - robot.y,
        )

    @staticmethod
    def _collision_counts(
        simulator: WorldSimulator,
    ) -> dict[str, int]:
        return {
            robot_id: int(
                state.collision_count
            )
            for robot_id, state
            in simulator.robot_states.items()
        }

    @staticmethod
    def _command_for_robot(
        simulator: WorldSimulator,
        robot_id: str,
    ) -> list[float]:
        """
        Retrieve the velocity actually
        applied during the simulation step.

        This is a velocity-command dataset,
        not a force/torque-action dataset.

        For rejected movements, the simulator
        may set applied velocity to zero.
        """
        robot = simulator.robot_states[robot_id]

        return [
            float(robot.linear_velocity),
            float(robot.angular_velocity),
        ]

    def _reward(
        self,
        previous_distance: float,
        current_distance: float,
        collision: bool,
        newly_reached_goal: bool,
    ) -> float:
        cfg = self.config

        progress = (
            previous_distance
            - current_distance
        )

        reward = (
            cfg.progress_reward_scale
            * progress
            - cfg.step_penalty
        )

        if collision:
            reward -= (
                cfg.collision_penalty
            )

        if newly_reached_goal:
            reward += (
                cfg.goal_reward
            )

        return float(reward)

    def generate_episode(
        self,
        episode_id: int,
    ) -> EpisodeSummary:
        cfg = self.config

        episode_seed = (
            cfg.seed + episode_id
        )

        random.seed(episode_seed)
        np.random.seed(
            episode_seed % (2**32)
        )

        scenario = self._create_scenario(
            episode_seed
        )

        simulator = self._create_simulator(
            scenario,
            episode_seed,
        )

        goals = self._goals_by_robot(
            scenario
        )

        robot_ids = list(
            simulator.robot_states.keys()
        )

        episode_transitions: list[
            Transition
        ] = []

        total_reward = 0.0
        termination_reason = "max_steps"

        try:
            simulator.start()

            for step_index in range(
                cfg.max_steps
            ):
                previous_states = {
                    robot_id: (
                        self._encode_state(
                            state,
                            goals.get(robot_id),
                        )
                    )
                    for robot_id, state
                    in simulator.robot_states.items()
                }

                previous_distances = {
                    robot_id: (
                        self._distance_to_goal(
                            state,
                            goals.get(robot_id),
                        )
                    )
                    for robot_id, state
                    in simulator.robot_states.items()
                }

                previous_collisions = (
                    self._collision_counts(
                        simulator
                    )
                )

                previous_goal_flags = {
                    robot_id: bool(
                        state.goal_reached
                    )
                    for robot_id, state
                    in simulator.robot_states.items()
                }

                step_result = (
                    simulator.step()
                )

                terminal = (
                    step_result.status
                    in (
                        SimulatorStatus.COMPLETED,
                        SimulatorStatus.EMERGENCY_STOP,
                        SimulatorStatus.ERROR,
                    )
                )

                step_limit_reached = (
                    step_index + 1
                    >= cfg.max_steps
                )

                for robot_id in robot_ids:
                    robot = (
                        simulator.robot_states[
                            robot_id
                        ]
                    )

                    current_state = (
                        self._encode_state(
                            robot,
                            goals.get(robot_id),
                        )
                    )

                    current_distance = (
                        self._distance_to_goal(
                            robot,
                            goals.get(robot_id),
                        )
                    )

                    collision = (
                        robot.collision_count
                        > previous_collisions[
                            robot_id
                        ]
                    )

                    newly_reached_goal = (
                        bool(robot.goal_reached)
                        and not previous_goal_flags[
                            robot_id
                        ]
                    )

                    reward = self._reward(
                        previous_distance=(
                            previous_distances[
                                robot_id
                            ]
                        ),
                        current_distance=(
                            current_distance
                        ),
                        collision=collision,
                        newly_reached_goal=(
                            newly_reached_goal
                        ),
                    )

                    total_reward += reward

                    terminated = (
                        terminal
                        or bool(
                            robot.goal_reached
                        )
                    )

                    truncated = (
                        step_limit_reached
                        and not terminated
                    )

                    transition = Transition(
                        state=(
                            previous_states[
                                robot_id
                            ]
                        ),
                        action=(
                            self._command_for_robot(
                                simulator,
                                robot_id,
                            )
                        ),
                        next_state=(
                            current_state
                        ),
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        episode_id=episode_id,
                        robot_id=robot_id,
                        step=step_index,
                        collision=collision,
                        goal_reached=bool(
                            robot.goal_reached
                        ),
                    )

                    episode_transitions.append(
                        transition
                    )

                if terminal:
                    termination_reason = (
                        step_result.status.value
                    )
                    break

            report = simulator.report()

            if (
                simulator.status
                == SimulatorStatus.EMERGENCY_STOP
            ):
                termination_reason = (
                    "emergency_stop"
                )

            summary = EpisodeSummary(
                episode_id=episode_id,
                scenario_type=(
                    cfg.scenario_type
                ),
                seed=episode_seed,
                steps=int(
                    report.total_steps
                ),
                transitions=len(
                    episode_transitions
                ),
                total_reward=(
                    total_reward
                ),
                goals_reached=int(
                    report.goals_reached
                ),
                collisions=int(
                    report.total_collisions
                ),
                success=(
                    report.goals_reached
                    == report.total_robots
                    and report.total_robots > 0
                ),
                termination_reason=(
                    termination_reason
                ),
                robot_ids=robot_ids,
            )

            self.transitions.extend(
                episode_transitions
            )

            self.episode_summaries.append(
                summary
            )

            if cfg.save_episode_json:
                self._save_episode_json(
                    summary,
                    episode_transitions,
                )

            return summary

        finally:
            simulator.close()

    def _output_path(self) -> Path:
        path = Path(
            self.config.output_dir
        )

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def _save_episode_json(
        self,
        summary: EpisodeSummary,
        transitions: Sequence[
            Transition
        ],
    ) -> None:
        directory = (
            self._output_path()
            / "episodes"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "summary": asdict(
                summary
            ),
            "transitions": [
                asdict(item)
                for item in transitions
            ],
        }

        path = directory / (
            f"episode_"
            f"{summary.episode_id:06d}.json"
        )

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    def generate(
        self,
    ) -> list[EpisodeSummary]:
        LOGGER.info(
            "Generating %d episodes",
            self.config.episodes,
        )

        for episode_id in range(
            self.config.episodes
        ):
            try:
                summary = (
                    self.generate_episode(
                        episode_id
                    )
                )

            except Exception:
                LOGGER.exception(
                    "Episode %d failed",
                    episode_id,
                )
                continue

            if (
                (episode_id + 1)
                % self.config.log_interval
                == 0
            ):
                LOGGER.info(
                    "Episodes: %d/%d | "
                    "Transitions: %d | "
                    "Last success: %s",
                    episode_id + 1,
                    self.config.episodes,
                    len(self.transitions),
                    summary.success,
                )

        if not self.episode_summaries:
            raise RuntimeError(
                "No episodes were generated"
            )

        return self.episode_summaries

    def _split_episode_ids(
        self,
    ) -> dict[str, set[int]]:
        """
        Split by episode, not transition,
        to prevent train/test leakage.
        """

        episode_ids = [
            summary.episode_id
            for summary
            in self.episode_summaries
        ]

        rng = np.random.default_rng(
            self.config.seed
        )

        rng.shuffle(
            episode_ids
        )

        count = len(
            episode_ids
        )

        train_count = int(
            count
            * self.config.train_ratio
        )

        val_count = int(
            count
            * self.config.val_ratio
        )

        train_end = train_count
        val_end = (
            train_end + val_count
        )

        return {
            "train": set(
                episode_ids[:train_end]
            ),
            "val": set(
                episode_ids[
                    train_end:val_end
                ]
            ),
            "test": set(
                episode_ids[val_end:]
            ),
        }

    @staticmethod
    def _to_arrays(
        transitions: Sequence[
            Transition
        ],
    ) -> dict[str, np.ndarray]:
        count = len(transitions)

        if count == 0:
            return {
                "states": np.empty(
                    (0, STATE_DIM),
                    dtype=np.float32,
                ),
                "actions": np.empty(
                    (0, ACTION_DIM),
                    dtype=np.float32,
                ),
                "next_states": np.empty(
                    (0, STATE_DIM),
                    dtype=np.float32,
                ),
                "rewards": np.empty(
                    (0,),
                    dtype=np.float32,
                ),
                "terminated": np.empty(
                    (0,),
                    dtype=np.bool_,
                ),
                "truncated": np.empty(
                    (0,),
                    dtype=np.bool_,
                ),
                "collisions": np.empty(
                    (0,),
                    dtype=np.bool_,
                ),
                "goals_reached": np.empty(
                    (0,),
                    dtype=np.bool_,
                ),
                "episode_ids": np.empty(
                    (0,),
                    dtype=np.int64,
                ),
                "robot_ids": np.empty(
                    (0,),
                    dtype="U64",
                ),
                "steps": np.empty(
                    (0,),
                    dtype=np.int32,
                ),
            }

        return {
            "states": np.asarray(
                [
                    item.state
                    for item in transitions
                ],
                dtype=np.float32,
            ),
            "actions": np.asarray(
                [
                    item.action
                    for item in transitions
                ],
                dtype=np.float32,
            ),
            "next_states": np.asarray(
                [
                    item.next_state
                    for item in transitions
                ],
                dtype=np.float32,
            ),
            "rewards": np.asarray(
                [
                    item.reward
                    for item in transitions
                ],
                dtype=np.float32,
            ),
            "terminated": np.asarray(
                [
                    item.terminated
                    for item in transitions
                ],
                dtype=np.bool_,
            ),
            "truncated": np.asarray(
                [
                    item.truncated
                    for item in transitions
                ],
                dtype=np.bool_,
            ),
            "collisions": np.asarray(
                [
                    item.collision
                    for item in transitions
                ],
                dtype=np.bool_,
            ),
            "goals_reached": np.asarray(
                [
                    item.goal_reached
                    for item in transitions
                ],
                dtype=np.bool_,
            ),
            "episode_ids": np.asarray(
                [
                    item.episode_id
                    for item in transitions
                ],
                dtype=np.int64,
            ),
            "robot_ids": np.asarray(
                [
                    item.robot_id
                    for item in transitions
                ],
                dtype="U64",
            ),
            "steps": np.asarray(
                [
                    item.step
                    for item in transitions
                ],
                dtype=np.int32,
            ),
        }

    def export(self) -> dict[str, Any]:
        output = self._output_path()

        splits = self._split_episode_ids()

        split_counts: dict[
            str, int
        ] = {}

        for split_name, episode_ids in (
            splits.items()
        ):
            selected = [
                transition
                for transition
                in self.transitions
                if transition.episode_id
                in episode_ids
            ]

            arrays = self._to_arrays(
                selected
            )

            path = output / (
                f"{split_name}.npz"
            )

            np.savez_compressed(
                path,
                **arrays,
            )

            split_counts[
                split_name
            ] = len(selected)

            LOGGER.info(
                "Saved %s: %d transitions",
                path,
                len(selected),
            )

        metadata = {
            "project": "NeuroWorld AI",
            "dataset_type": (
                "robot_navigation_transitions"
            ),
            "created_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "state_features": (
                STATE_FEATURES
            ),
            "action_features": (
                ACTION_FEATURES
            ),
            "action_semantics": (
                "applied_velocity"
            ),
            "configuration": asdict(
                self.config
            ),
            "split_transition_counts": (
                split_counts
            ),
            "split_episode_counts": {
                name: len(ids)
                for name, ids
                in splits.items()
            },
        }

        (
            output / "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        summary = self.summary()

        (
            output / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return {
            "output_dir": str(output),
            "splits": split_counts,
            "metadata": metadata,
            "summary": summary,
        }

    def summary(self) -> dict[str, Any]:
        episodes = len(
            self.episode_summaries
        )

        successful = sum(
            item.success
            for item
            in self.episode_summaries
        )

        total_collisions = sum(
            item.collisions
            for item
            in self.episode_summaries
        )

        rewards = [
            item.total_reward
            for item
            in self.episode_summaries
        ]

        lengths = [
            item.steps
            for item
            in self.episode_summaries
        ]

        return {
            "episodes_generated": episodes,
            "transitions_generated": len(
                self.transitions
            ),
            "successful_episodes": successful,
            "success_rate": (
                successful / episodes
                if episodes
                else 0.0
            ),
            "total_collisions": (
                total_collisions
            ),
            "mean_episode_reward": (
                float(np.mean(rewards))
                if rewards
                else 0.0
            ),
            "mean_episode_length": (
                float(np.mean(lengths))
                if lengths
                else 0.0
            ),
            "episode_summaries": [
                asdict(item)
                for item
                in self.episode_summaries
            ],
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate NeuroWorld AI "
            "navigation trajectory datasets"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="random",
        choices=[
            "empty",
            "random",
            "maze",
            "corridor",
            "warehouse",
        ],
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--obstacles",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--timestep",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/trajectories",
    )

    parser.add_argument(
        "--save-episodes",
        action="store_true",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    config = TrajectoryBuilderConfig(
        episodes=args.episodes,
        scenario_type=args.scenario,
        num_robots=args.robots,
        num_obstacles=args.obstacles,
        max_steps=args.max_steps,
        timestep=args.timestep,
        seed=args.seed,
        output_dir=args.output,
        save_episode_json=(
            args.save_episodes
        ),
        log_interval=args.log_interval,
    )

    builder = TrajectoryDatasetBuilder(
        config
    )

    builder.generate()

    result = builder.export()

    LOGGER.info(
        "Dataset generation complete."
    )

    LOGGER.info(
        "Output: %s",
        result["output_dir"],
    )

    LOGGER.info(
        "Transitions: %d",
        result["summary"][
            "transitions_generated"
        ],
    )

    LOGGER.info(
        "Success rate: %.2f%%",
        result["summary"][
            "success_rate"
        ] * 100,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
