"""OpenEnv inference script for the SRE Incident Triage benchmark."""

from __future__ import annotations

import json
import os
from urllib import error, request
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env when running locally; no-op if not present
except ImportError:
    pass  # dotenv optional — env vars set directly in HF Space secrets

from openai import OpenAI

from env import SREIncidentTriageEnv


API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1")

BENCHMARK = "sre-incident-agent"
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:+.2f} done={done_val} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:+.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) on an on-call shift.
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


def get_llm_decision(client: Optional[OpenAI], incident_text: str) -> Dict[str, str]:
    fallback = {"severity": "P2", "root_cause": "unknown", "action": "monitor"}

    if client is None:
        return fallback

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"INCIDENT ALERT:\n{incident_text}\n\nProvide your triage decision as JSON.",
                },
            ],
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        if all(k in parsed for k in ("severity", "root_cause", "action")):
            return {
                "severity": str(parsed["severity"]),
                "root_cause": str(parsed["root_cause"]),
                "action": str(parsed["action"]),
            }
        return fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Label lists (inlined from env.py so inference.py is fully self-contained)
# ---------------------------------------------------------------------------
SEVERITY_LABELS = ["P1", "P2", "P3"]
ROOT_CAUSE_LABELS = [
    "database", "deployment", "infra", "external_dependency",
    "network", "memory_leak", "unknown",
]
ACTION_LABELS = [
    "rollback", "scale_db", "page_all_teams", "monitor",
    "restart_service", "escalate", "investigate",
]


class RemoteSREIncidentTriageEnv:
    def __init__(self, base_url: str, config: Dict[str, int]) -> None:
        self.base_url = base_url.rstrip("/")
        self.config = dict(config)

    def _request(self, path: str, payload: Dict[str, object]) -> Dict[str, object]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=30) as response:
                response_text = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"failed to call {url}: {exc}") from exc

        if not response_text:
            return {}
        return json.loads(response_text)

    def reset(self):
        payload = dict(self.config)
        response = self._request("/reset", payload)
        observation = response.get("observation", {})
        info = {"state": response.get("info", {})}
        return observation, info

    def step(self, action: Dict[str, int]):
        response = self._request("/step", action)
        observation = response.get("observation", {})
        reward = float(response.get("reward", 0.0))
        terminated = bool(response.get("terminated", False))
        truncated = bool(response.get("truncated", False))
        info = response.get("info", {})
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        return None


def _normalize(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _decision_to_env_action(decision: Dict[str, str]) -> Dict[str, int]:
    """Convert string labels to environment integer indices."""
    sev = decision.get("severity", "P2").strip().upper()
    sev_idx = SEVERITY_LABELS.index(sev) if sev in SEVERITY_LABELS else 1

    rc = _normalize(decision.get("root_cause", "unknown"))
    rc_idx = ROOT_CAUSE_LABELS.index(rc) if rc in ROOT_CAUSE_LABELS else 6

    act = _normalize(decision.get("action", "monitor"))
    act_idx = ACTION_LABELS.index(act) if act in ACTION_LABELS else 3

    return {"severity": sev_idx, "root_cause": rc_idx, "action": act_idx}


def _make_env(config: Dict[str, int]):
    try:
        remote_env = RemoteSREIncidentTriageEnv(ENV_BASE_URL, config)
        remote_env.reset()
        return remote_env
    except Exception:
        return SREIncidentTriageEnv(config)


def run_task(task_name: str, config: Dict[str, int]) -> List[float]:
    env = _make_env(config)
    obs, _ = env.reset()

    client = None
    if API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    max_steps = int(config.get("max_steps", 10))
    max_total_reward = max_steps * 1.0  # each step can yield at most 1.0

    rewards: List[float] = []
    done = False
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_name, BENCHMARK, MODEL_NAME)

    try:
        step = 0
        while not done:
            step += 1
            incident_text = str(obs.get("incident_text", ""))

            error = None
            try:
                decision = get_llm_decision(client, incident_text)
                env_action = _decision_to_env_action(decision)
                obs, reward, terminated, truncated, _ = env.step(env_action)
                done = bool(terminated or truncated)
                rewards.append(float(reward))
                steps_taken = step
                log_step(step, json.dumps(decision, separators=(",", ":")), float(reward), done, error)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                done = True
                steps_taken = step
                log_step(step, json.dumps({"severity": "P2", "root_cause": "unknown", "action": "monitor"}, separators=(",", ":")), 0.0, done, error)

        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return rewards


def main() -> None:
    run_task("easy", {"max_steps": 10, "difficulty": 1, "seed": 42})
    run_task("medium", {"max_steps": 20, "difficulty": 2, "seed": 42})
    run_task("hard", {"max_steps": 30, "difficulty": 3, "seed": 42})


if __name__ == "__main__":
    main()

