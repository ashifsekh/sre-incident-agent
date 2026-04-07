"""Baseline agents for the SRE incident triage environment."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from env import ACTION_LABELS, ROOT_CAUSE_LABELS, SEVERITY_LABELS, SREIncidentTriageEnv


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _severity_to_index(value: str) -> int:
    value = value.strip().upper()
    if value in SEVERITY_LABELS:
        return SEVERITY_LABELS.index(value)
    return SEVERITY_LABELS.index("P2")


def _root_cause_to_index(value: str) -> int:
    normalized = _normalize(value)
    if normalized in ROOT_CAUSE_LABELS:
        return ROOT_CAUSE_LABELS.index(normalized)
    return ROOT_CAUSE_LABELS.index("unknown")


def _action_to_index(value: str) -> int:
    normalized = _normalize(value)
    if normalized in ACTION_LABELS:
        return ACTION_LABELS.index(normalized)
    return ACTION_LABELS.index("monitor")


def _decision_to_env_action(decision: Dict[str, str]) -> Dict[str, int]:
    return {
        "severity": _severity_to_index(decision.get("severity", "P2")),
        "root_cause": _root_cause_to_index(decision.get("root_cause", "unknown")),
        "action": _action_to_index(decision.get("action", "monitor")),
    }


def _extract_percent(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)%", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _severity_from_text(incident_text: str) -> str:
    lower = incident_text.lower()

    # Prefer explicit error-rate percentages if present in text.
    pct = _extract_percent(lower)
    if pct is not None:
        if pct > 10.0:
            return "P1"
        if pct > 2.0:
            return "P2"
        return "P3"

    if "timeout" in lower or "outage" in lower or "sev1" in lower:
        return "P1"
    if "degraded" in lower or "latency" in lower or "error" in lower:
        return "P2"
    return "P3"


def heuristic_agent_decision(incident_text: str) -> Dict[str, str]:
    """Keyword-based baseline with no external API calls."""

    text = incident_text.lower()

    severity = _severity_from_text(text)

    root_cause = "unknown"
    action = "monitor"

    mapping: List[Tuple[List[str], str, str]] = [
        (["deploy", "rollout", "release", "canary", "manifest"], "deployment", "rollback"),
        (["db", "database", "postgres", "replica", "deadlock", "connection pool"], "database", "scale_db"),
        (["memory", "oom", "heap", "rss", "leak"], "memory_leak", "restart_service"),
        (["kubernetes", "node", "volume", "autoscaler", "cluster"], "infra", "escalate"),
        (["dns", "packet", "network", "bgp", "isp", "switch"], "network", "investigate"),
        (["third-party", "vendor", "upstream", "provider", "gateway"], "external_dependency", "escalate"),
    ]

    for keywords, inferred_root, inferred_action in mapping:
        if any(keyword in text for keyword in keywords):
            root_cause = inferred_root
            action = inferred_action
            break

    return {
        "severity": severity,
        "root_cause": root_cause,
        "action": action,
    }


def _extract_with_keywords(text: str) -> Dict[str, str]:
    lower = text.lower()

    severity = "P2"
    if "p1" in lower:
        severity = "P1"
    elif "p3" in lower:
        severity = "P3"

    root_cause = "unknown"
    for cause in ROOT_CAUSE_LABELS:
        if cause == "unknown":
            continue
        if cause in lower:
            root_cause = cause
            break

    action = "monitor"
    for candidate in ACTION_LABELS:
        if candidate in lower:
            action = candidate
            break

    return {
        "severity": severity,
        "root_cause": root_cause,
        "action": action,
    }


def llm_agent_decision(client: OpenAI | None, incident_text: str) -> Dict[str, str]:
    """LLM baseline with strict JSON contract and robust fallback parsing."""

    default_decision = {"severity": "P2", "root_cause": "unknown", "action": "monitor"}

    if client is None:
        LOGGER.warning("OPENAI_API_KEY missing; using default LLM fallback decision.")
        return default_decision

    system_prompt = """You are an expert Site Reliability Engineer (SRE) on an on-call shift.
You will receive an incident alert. Your job is to triage it immediately.

You MUST respond with ONLY a JSON object. No explanation. No markdown.
No extra text. Only raw JSON.

Your response format:
{
  "severity": "<P1 or P2 or P3>",
  "root_cause": "<one of: database, deployment, infra, external_dependency, network, memory_leak, unknown>",
  "action": "<one of: rollback, scale_db, page_all_teams, monitor, restart_service, escalate, investigate>"
}

Severity guide:
- P1: System down or data loss risk. Immediate action. Revenue impacted.
- P2: Degraded performance. Some users affected. Fix within 1 hour.
- P3: Minor issue. No user impact. Fix within 24 hours.

Root cause guide:
- database: DB connection issues, slow queries, pool exhaustion, replication lag
- deployment: Recent code push or config change caused the issue
- infra: CPU, memory, disk, or hardware level problems
- external_dependency: Third-party API, payment gateway, or DNS failure
- network: Packet loss, latency spikes, firewall or routing issues
- memory_leak: Gradual memory growth causing OOM or slowdown
- unknown: Cannot determine from available signals

Action guide:
- rollback: Undo the last deployment immediately
- scale_db: Add read replicas or increase DB connection pool
- page_all_teams: Wake up all engineers (only for true P1 outages)
- monitor: Watch the metrics, no immediate action needed
- restart_service: Restart the affected microservice
- escalate: Escalate to senior engineer or vendor support
- investigate: Gather more data before deciding (for ambiguous P2/P3)

Think carefully about the signals. Some alerts contain misleading information.
Focus on the most likely root cause given ALL signals together."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"INCIDENT ALERT:\n{incident_text}\n\nProvide your triage decision as JSON.",
                },
            ],
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()

        try:
            parsed = json.loads(raw)
            if all(key in parsed for key in ("severity", "root_cause", "action")):
                return {
                    "severity": str(parsed["severity"]).upper(),
                    "root_cause": _normalize(str(parsed["root_cause"])),
                    "action": _normalize(str(parsed["action"])),
                }
        except json.JSONDecodeError:
            pass

        fallback = _extract_with_keywords(raw)
        if fallback != default_decision:
            return fallback

        LOGGER.warning("LLM response parsing failed; using safe default decision.")
        return default_decision

    except Exception as exc:
        LOGGER.warning("LLM call failed (%s); using safe default decision.", exc)
        return default_decision


def run_baseline(agent_type: str, task_name: str, config: Dict[str, Any]) -> float:
    """Run a complete episode and return normalized score in [0.0, 1.0]."""

    env = SREIncidentTriageEnv(config)
    obs, _ = env.reset()

    client = None
    if agent_type == "llm":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)

    total_score = 0.0
    step_count = 0
    terminated = False
    truncated = False

    print(f"\nRunning {agent_type} baseline on task={task_name} with config={config}")

    while not (terminated or truncated):
        incident_text = str(obs.get("incident_text", ""))

        if agent_type == "heuristic":
            decision = heuristic_agent_decision(incident_text)
        elif agent_type == "llm":
            decision = llm_agent_decision(client, incident_text)
        else:
            raise ValueError("agent_type must be 'heuristic' or 'llm'")

        env_action = _decision_to_env_action(decision)
        obs, reward, terminated, truncated, info = env.step(env_action)

        step_count += 1
        total_score += float(reward)

        print(
            f"step={step_count:02d} decision={decision} "
            f"score={reward:.4f} total={total_score:.4f}"
        )

        _ = info

    final_score = total_score / max(step_count, 1)
    print(f"Final normalized score ({agent_type}, {task_name}): {final_score:.4f}")
    return round(final_score, 4)


def _run_all_tasks() -> Dict[str, Dict[str, float]]:
    tasks = {
        "Easy": {"max_steps": 10, "difficulty": 1, "seed": 42},
        "Medium": {"max_steps": 20, "difficulty": 2, "seed": 42},
        "Hard": {"max_steps": 30, "difficulty": 3, "seed": 42},
    }

    summary: Dict[str, Dict[str, float]] = {"heuristic": {}, "llm": {}}

    for task_name, config in tasks.items():
        summary["heuristic"][task_name] = run_baseline("heuristic", task_name, config)

    for task_name, config in tasks.items():
        summary["llm"][task_name] = run_baseline("llm", task_name, config)

    return summary


def _print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    tasks = ["Easy", "Medium", "Hard"]
    print("\nSummary Table (normalized final scores)")
    print("agent      | Easy   | Medium | Hard")
    print("-----------+--------+--------+------")

    for agent_name in ["heuristic", "llm"]:
        scores = summary.get(agent_name, {})
        easy = scores.get("Easy", 0.0)
        medium = scores.get("Medium", 0.0)
        hard = scores.get("Hard", 0.0)
        print(f"{agent_name:<10} | {easy:<6.4f} | {medium:<6.4f} | {hard:<6.4f}")


if __name__ == "__main__":
    final_summary = _run_all_tasks()
    _print_summary_table(final_summary)
