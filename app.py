"""FastAPI app for running SRE baseline evaluations."""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI

from baseline import run_baseline
from env import SREIncidentTriageEnv


app = FastAPI(title="sre-incident-agent", version="0.1.0")

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
