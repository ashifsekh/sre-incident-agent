"""Scoring utilities for SRE incident triage agent decisions."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


SEVERITY_ORDER = {"P1": 1, "P2": 2, "P3": 3}

# Groups categories considered related for partial root-cause credit.
RELATED_CAUSES: Dict[str, List[str]] = {
    "database": ["database", "deployment", "memory_leak"],
    "deployment": ["deployment", "infra"],
    "infra": ["infra", "network", "deployment"],
    "external_dependency": ["external_dependency", "network"],
    "network": ["network", "infra", "external_dependency"],
    "memory_leak": ["memory_leak", "database", "infra"],
}


class SREReward(BaseModel):
    """Detailed reward breakdown for one triage decision."""

    severity_score: float = Field(ge=0.0, le=1.0)
    root_cause_score: float = Field(ge=0.0, le=1.0)
    action_score: float = Field(ge=-0.2, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)


def _normalize_text(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _severity_score(predicted: str, expected: str) -> float:
    p = SEVERITY_ORDER.get(predicted.upper())
    t = SEVERITY_ORDER.get(expected.upper())
    if p is None or t is None:
        return 0.0

    diff = abs(p - t)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0


def _root_cause_score(predicted: str, expected: str) -> float:
    predicted_norm = _normalize_text(predicted)
    expected_norm = _normalize_text(expected)

    if predicted_norm == expected_norm:
        return 1.0

    expected_related = RELATED_CAUSES.get(expected_norm)
    if not expected_related:
        return 0.0

    if predicted_norm in expected_related:
        return 0.4
    return 0.0


def _action_score(predicted_action: str, true_answer: Dict[str, object]) -> float:
    expected_action = str(true_answer.get("action", "")).strip()
    predicted_action = predicted_action.strip()

    if predicted_action == expected_action and expected_action:
        return 1.0

    alternatives = true_answer.get("acceptable_alternatives", [])
    if isinstance(alternatives, list) and predicted_action in alternatives:
        return 0.6

    true_severity = str(true_answer.get("severity", "")).upper()
    if true_severity == "P3" and predicted_action == "page_all_teams":
        return -0.2

    return 0.0


def score_decision(agent_decision: Dict[str, str], true_answer: Dict[str, object]) -> SREReward:
    """Score one agent decision against the true incident answer.

    Expected keys:
    - agent_decision: severity, root_cause, action
    - true_answer: severity, root_cause, action
      Optional: acceptable_alternatives (list[str])
    """

    severity = _severity_score(
        predicted=str(agent_decision.get("severity", "")),
        expected=str(true_answer.get("severity", "")),
    )
    root_cause = _root_cause_score(
        predicted=str(agent_decision.get("root_cause", "")),
        expected=str(true_answer.get("root_cause", "")),
    )
    action = _action_score(
        predicted_action=str(agent_decision.get("action", "")),
        true_answer=true_answer,
    )

    weighted = (severity * 0.3) + (root_cause * 0.4) + (action * 0.3)
    total = max(0.0, min(1.0, round(weighted, 4)))

    return SREReward(
        severity_score=round(severity, 4),
        root_cause_score=round(root_cause, 4),
        action_score=round(action, 4),
        total_score=total,
    )
