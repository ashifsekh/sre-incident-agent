"""Gymnasium environment for SRE incident triage."""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium import spaces
from pydantic import BaseModel, Field

from grader import SREReward as GraderReward
from grader import score_decision
from incidents import generate_incident


SEVERITY_LABELS = ["P1", "P2", "P3"]
ROOT_CAUSE_LABELS = [
    "database",
    "deployment",
    "infra",
    "external_dependency",
    "network",
    "memory_leak",
    "unknown",
]
ACTION_LABELS = [
    "rollback",
    "scale_db",
    "page_all_teams",
    "monitor",
    "restart_service",
    "escalate",
    "investigate",
]


class SREObservation(BaseModel):
    """Current environment observation returned to the agent."""

    incident_text: str
    step_number: int = Field(ge=0)
    incidents_resolved: int = Field(ge=0)
    current_score: float = Field(ge=0.0, le=1.0)


class SREAction(BaseModel):
    """Structured agent decision for one incident."""

    severity: str
    root_cause: str
    action: str


class SREReward(BaseModel):
    """Local reward model that mirrors grader output."""

    severity_score: float = Field(ge=0.0, le=1.0)
    root_cause_score: float = Field(ge=0.0, le=1.0)
    action_score: float = Field(ge=-0.2, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)


class SREState(BaseModel):
    """Internal environment state snapshot."""

    step_number: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    difficulty: int = Field(ge=1, le=3)
    seed: int
    incidents_resolved: int = Field(ge=0)
    cumulative_reward: float
    average_score: float = Field(ge=0.0, le=1.0)
    current_incident_text: str
    last_action: Optional[SREAction] = None
    last_reward: Optional[SREReward] = None


class SREIncidentTriageEnv(gym.Env):
    """Environment for SRE incident triage episodes."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        config = config or {}

        self.max_steps = int(config.get("max_steps", 10))
        self.difficulty = int(config.get("difficulty", 1))
        self.seed_value = int(config.get("seed", 42))

        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")
        if self.difficulty not in (1, 2, 3):
            raise ValueError("difficulty must be 1, 2, or 3")

        self.action_space = spaces.Dict(
            {
                "severity": spaces.Discrete(len(SEVERITY_LABELS)),
                "root_cause": spaces.Discrete(len(ROOT_CAUSE_LABELS)),
                "action": spaces.Discrete(len(ACTION_LABELS)),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "incident_text": spaces.Text(max_length=4096),
                "step_number": spaces.Discrete(self.max_steps + 1),
                "incidents_resolved": spaces.Discrete(self.max_steps + 1),
                "current_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=float),
            }
        )

        self._step_number = 0
        self._incidents_resolved = 0
        self._cumulative_reward = 0.0
        self._episode_seed = self.seed_value
        self._current_incident: Dict[str, str] | None = None
        self._last_action: Optional[SREAction] = None
        self._last_reward: Optional[SREReward] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        self._step_number = 0
        self._incidents_resolved = 0
        self._cumulative_reward = 0.0
        self._last_action = None
        self._last_reward = None

        if options and "seed" in options:
            self._episode_seed = int(options["seed"])
        elif seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = self.seed_value

        self._current_incident = generate_incident(seed=self._episode_seed, difficulty=self.difficulty)
        obs = self._build_observation()
        info = {
            "state": self.state().model_dump(),
        }
        return obs, info

    def step(self, action: Dict[str, int]):
        if self._current_incident is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        parsed_action = self._parse_action(action)
        true_answer = self._build_true_answer(self._current_incident)
        grader_reward = score_decision(
            agent_decision=parsed_action.model_dump(),
            true_answer=true_answer,
        )
        reward_model = self._to_reward_model(grader_reward)

        self._last_action = parsed_action
        self._last_reward = reward_model

        reward = max(0.001, min(0.999, reward_model.total_score))
        self._cumulative_reward += reward

        if reward >= 0.6:
            self._incidents_resolved += 1

        self._step_number += 1
        terminated = self._step_number >= self.max_steps
        truncated = False

        if not terminated:
            next_seed = self._episode_seed + self._step_number
            self._current_incident = generate_incident(seed=next_seed, difficulty=self.difficulty)

        obs = self._build_observation()
        info = {
            "reward_breakdown": reward_model.model_dump(),
            "true_answer": true_answer,
            "state": self.state().model_dump(),
        }
        return obs, reward, terminated, truncated, info

    def state(self) -> SREState:
        current_text = ""
        if self._current_incident is not None:
            current_text = self._current_incident["incident_text"]

        avg_score = 0.0
        if self._step_number > 0:
            avg_score = self._cumulative_reward / self._step_number

        return SREState(
            step_number=self._step_number,
            max_steps=self.max_steps,
            difficulty=self.difficulty,
            seed=self._episode_seed,
            incidents_resolved=self._incidents_resolved,
            cumulative_reward=round(self._cumulative_reward, 4),
            average_score=round(min(max(avg_score, 0.0), 1.0), 4),
            current_incident_text=current_text,
            last_action=self._last_action,
            last_reward=self._last_reward,
        )

    def render(self):
        if self._current_incident is None:
            print("No active incident. Call reset() first.")
            return

        print("=== SRE Incident ===")
        print(self._current_incident["incident_text"])
        print("\n=== Last Action ===")
        if self._last_action is None:
            print("None")
        else:
            print(self._last_action.model_dump())

    def _build_observation(self) -> Dict[str, Any]:
        if self._current_incident is None:
            raise RuntimeError("Current incident missing")

        current_score = 0.0
        if self._step_number > 0:
            current_score = self._cumulative_reward / self._step_number

        obs_model = SREObservation(
            incident_text=self._current_incident["incident_text"],
            step_number=self._step_number,
            incidents_resolved=self._incidents_resolved,
            current_score=round(min(max(current_score, 0.0), 1.0), 4),
        )
        return obs_model.model_dump()

    def _parse_action(self, action: Dict[str, int]) -> SREAction:
        try:
            severity_idx = int(action["severity"])
            root_cause_idx = int(action["root_cause"])
            action_idx = int(action["action"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("action must be a dict with severity/root_cause/action integers") from exc

        if not (0 <= severity_idx < len(SEVERITY_LABELS)):
            raise ValueError("severity index out of range")
        if not (0 <= root_cause_idx < len(ROOT_CAUSE_LABELS)):
            raise ValueError("root_cause index out of range")
        if not (0 <= action_idx < len(ACTION_LABELS)):
            raise ValueError("action index out of range")

        return SREAction(
            severity=SEVERITY_LABELS[severity_idx],
            root_cause=ROOT_CAUSE_LABELS[root_cause_idx],
            action=ACTION_LABELS[action_idx],
        )

    def _build_true_answer(self, incident: Dict[str, str]) -> Dict[str, Any]:
        severity = incident.get("true_severity", "P3")
        root_cause = self._infer_root_cause_category(
            true_root_cause=incident.get("true_root_cause", ""),
            incident_text=incident.get("incident_text", ""),
        )
        canonical_action = self._infer_action_label(incident.get("correct_action", ""))

        alternatives = []
        if canonical_action == "escalate":
            alternatives = ["investigate"]
        elif canonical_action == "scale_db":
            alternatives = ["restart_service", "investigate"]
        elif canonical_action == "rollback":
            alternatives = ["escalate"]

        return {
            "severity": severity,
            "root_cause": root_cause,
            "action": canonical_action,
            "acceptable_alternatives": alternatives,
        }

    @staticmethod
    def _infer_root_cause_category(true_root_cause: str, incident_text: str) -> str:
        combined = f"{true_root_cause} {incident_text}".lower()

        keyword_map = {
            "database": ["db", "postgres", "sql", "replica", "transaction", "connection pool"],
            "deployment": ["deploy", "rollout", "manifest", "release", "canary", "version"],
            "infra": ["kubernetes", "node", "volume", "autoscaler", "kernel", "cluster"],
            "external_dependency": ["third-party", "vendor", "upstream", "gateway", "provider"],
            "network": ["dns", "packet", "network", "bgp", "isp", "latency", "switch"],
            "memory_leak": ["memory leak", "heap", "oom", "rss", "retention", "cache growth"],
        }

        for category, keywords in keyword_map.items():
            if any(keyword in combined for keyword in keywords):
                return category

        return "unknown"

    @staticmethod
    def _infer_action_label(correct_action: str) -> str:
        text = correct_action.lower()

        if any(token in text for token in ["rollback", "revert"]):
            return "rollback"
        if any(token in text for token in ["db", "database", "replica", "iops", "query"]):
            return "scale_db"
        if any(token in text for token in ["restart", "reschedule", "drain", "reboot"]):
            return "restart_service"
        if any(token in text for token in ["monitor", "observe", "watch"]):
            return "monitor"
        if any(token in text for token in ["investigate", "validate", "profile", "debug"]):
            return "investigate"
        if any(token in text for token in ["incident commander", "on-call", "escalate", "coordinate"]):
            return "escalate"

        return "escalate"

    @staticmethod
    def _to_reward_model(grader_reward: GraderReward) -> SREReward:
        return SREReward(**grader_reward.model_dump())
