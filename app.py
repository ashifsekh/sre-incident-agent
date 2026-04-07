"""FastAPI app for running SRE baseline evaluations."""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI

from baseline import run_baseline
from env import (
    ACTION_LABELS,
    ROOT_CAUSE_LABELS,
    SEVERITY_LABELS,
    SREAction,
    SREIncidentTriageEnv,
)


app = FastAPI(title="sre-incident-agent", version="0.1.0")

_active_env = None

PROJECT_NAME = "sre-incident-agent"
ENV_DESCRIPTION = (
    "Gymnasium-based SRE incident triage environment with text observations "
    "and multi-part discrete actions (severity, root_cause, action)."
)

TASKS: Dict[str, Dict[str, int]] = {
    "Easy": {"max_steps": 10, "difficulty": 1, "seed": 42},
    "Medium": {"max_steps": 20, "difficulty": 2, "seed": 42},
    "Hard": {"max_steps": 30, "difficulty": 3, "seed": 42},
}


@app.get("/")
def run_demo() -> Dict[str, object]:
    """Run heuristic baseline on all tasks and return aggregate metrics."""

    # Instantiate once to prove environment import and construction works.
    _ = SREIncidentTriageEnv({"max_steps": 1, "difficulty": 1, "seed": 42})

    scores: Dict[str, float] = {}
    for task_name, config in TASKS.items():
        scores[task_name] = run_baseline(
            agent_type="heuristic",
            task_name=task_name,
            config=config,
        )

    average_score = round(sum(scores.values()) / max(len(scores), 1), 4)

    return {
        "project": PROJECT_NAME,
        "status": "ok",
        "scores": scores,
        "average_score": average_score,
        "environment": ENV_DESCRIPTION,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset_env() -> Dict[str, object]:
    """
    OpenEnv validator ping endpoint.
    Instantiates a fresh environment, calls reset(), and returns the
    initial observation to prove the environment works end to end.
    """
    global _active_env
    env_config = {"max_steps": 10, "difficulty": 1, "seed": 42}
    if config:
        for key in ("max_steps", "difficulty", "seed"):
            if key in config:
                env_config[key] = int(config[key])

    _active_env = SREIncidentTriageEnv(env_config)
    obs, info = _active_env.reset()
    return {
        "status": "ok",
        "observation": {
            "incident_text": obs.get("incident_text", ""),
            "step_number": obs.get("step_number", 0),
            "incidents_resolved": obs.get("incidents_resolved", 0),
            "current_score": obs.get("current_score", 0.0),
        },
        "info": str(info.get("state", {})),
    }


@app.post("/step")
def step_env(action: SREAction) -> Dict[str, object]:
    """
    OpenEnv step endpoint. Takes a typed SREAction (severity, root_cause,
    action as strings) and returns the next observation, reward, and done status.
    Uses a module-level env instance so state persists between calls.
    """
    global _active_env
    if _active_env is None:
        return {"error": "Call /reset first to initialize the environment"}

    def _label_index(label: str, choices: list, default: int = 0) -> int:
        norm = label.strip().lower().replace("-", "_").replace(" ", "_")
        for i, c in enumerate(choices):
            if c.lower() == norm:
                return i
        return default

    env_action = {
        "severity": _label_index(action.severity, SEVERITY_LABELS, 1),
        "root_cause": _label_index(action.root_cause, ROOT_CAUSE_LABELS, 6),
        "action": _label_index(action.action, ACTION_LABELS, 3),
    }

    obs, reward, terminated, truncated, info = _active_env.step(env_action)
    return {
        "observation": {
            "incident_text": obs.get("incident_text", ""),
            "step_number": obs.get("step_number", 0),
            "incidents_resolved": obs.get("incidents_resolved", 0),
            "current_score": obs.get("current_score", 0.0),
        },
        "reward": round(reward, 4),
        "terminated": terminated,
        "truncated": truncated,
        "done": terminated or truncated,
        "info": info.get("reward_breakdown", {}),
    }


@app.get("/state")
def get_state() -> Dict[str, object]:
    """
    OpenEnv state endpoint. Returns current environment state snapshot.
    """
    global _active_env
    if _active_env is None:
        return {"error": "Call /reset first to initialize the environment"}

    state = _active_env.state()
    return {"state": state.model_dump()}
