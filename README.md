---
title: SRE Incident Triage Agent
emoji: 🚨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# SRE Incident Triage Agent

> A Gymnasium-compatible benchmark environment for training, evaluating, and comparing incident triage agents — from keyword heuristics to frontier LLMs.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Environment Design](#environment-design)
   - [Observation Space](#observation-space)
   - [Action Space](#action-space)
   - [Reward Function](#reward-function)
4. [Task Definitions](#task-definitions)
5. [Incident Corpus](#incident-corpus)
6. [Baseline Agents](#baseline-agents)
7. [Baseline Scores](#baseline-scores)
8. [Project Structure](#project-structure)
9. [Local Setup](#local-setup)
10. [Running Baselines](#running-baselines)
11. [API Reference](#api-reference)
12. [Docker Deployment](#docker-deployment)
13. [OpenEnv Submission](#openenv-submission)
14. [Configuration](#configuration)

---

## Overview

Modern production systems generate a constant stream of noisy, high-volume alerts. During an incident, an on-call engineer must make three critical decisions in seconds:

1. **Severity** — How urgent is this? Is revenue impacted?
2. **Root Cause** — What actually broke?
3. **Immediate Action** — What should be done right now?

This project provides a reproducible, extensible benchmark for building and evaluating automated incident triage agents. It simulates realistic SRE (Site Reliability Engineering) scenarios, injects configurable telemetry noise (misleading signals), and scores agent decisions with partial credit — enabling fair comparison between rule-based heuristics, classical ML, and large language models.

**Key capabilities:**

- 🎮 Standard [Gymnasium](https://gymnasium.farama.org/) interface compatible with any RL or LLM agent
- 📊 Three difficulty levels with increasing noise in telemetry signals
- 🧮 Weighted, partial-credit scoring across severity, root cause, and action
- 🤖 Two included baseline agents: a keyword heuristic and a GPT-4o-mini LLM agent
- 🌐 FastAPI server with OpenEnv-compatible `/reset` endpoint for remote evaluation
- 🐳 Docker-ready for deployment to Hugging Face Spaces

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server (app.py)                  │
│   GET /       → run heuristic demo on all tasks                 │
│   GET /health → liveness probe                                  │
│   POST /reset → OpenEnv validator ping                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │    SREIncidentTriageEnv     │  (env.py)
            │   Gymnasium gym.Env         │
            │                             │
            │  reset() → initial obs      │
            │  step(action) → obs,reward  │
            │  render() → human output    │
            └──────┬───────────┬──────────┘
                   │           │
       ┌───────────▼──┐   ┌────▼──────────────┐
       │  incidents.py │   │    grader.py       │
       │               │   │                    │
       │ generate_     │   │ score_decision()   │
       │ incident()    │   │ Weighted partial   │
       │ 18 templates  │   │ credit scoring     │
       │ + noise       │   │                    │
       └───────────────┘   └────────────────────┘
                   │
       ┌───────────▼───────────────┐
       │       baseline.py         │
       │                           │
       │  heuristic_agent_decision │  ← keyword matching, no API
       │  llm_agent_decision       │  ← GPT-4o-mini via OpenAI API
       │  run_baseline()           │  ← full episode runner
       └───────────────────────────┘
                   │
       ┌───────────▼───────────────┐
       │       inference.py        │
       │                           │
       │  OpenEnv benchmark runner │
       │  HuggingFace router       │
       │  Qwen2.5-72B-Instruct     │
       └───────────────────────────┘
```

---

## Environment Design

### Observation Space

At each step, the agent receives a structured dictionary observation:

| Field                | Type    | Description                                               |
|----------------------|---------|-----------------------------------------------------------|
| `incident_text`      | `str`   | Full alert text with injected telemetry signals and noise |
| `step_number`        | `int`   | Current step index within the episode                     |
| `incidents_resolved` | `int`   | Count of steps where total reward ≥ 0.6                   |
| `current_score`      | `float` | Running average reward score in `[0.0, 1.0]`              |

**Example observation:**
```
ALERT: Write latency spike on primary Postgres cluster; checkout API timing out.
Context: service=payments, region=us-east-1, incident_age=14m, affected_requests=73%.
Additional telemetry (may include noise):
- CPU briefly high on API pods due to retries
- A/B test rollout happened 20 minutes earlier
```

### Action Space

Each action is a multi-part discrete decision with three required fields:

| Field        | Options                                                                                           |
|--------------|---------------------------------------------------------------------------------------------------|
| `severity`   | `P1`, `P2`, `P3`                                                                                  |
| `root_cause` | `database`, `deployment`, `infra`, `external_dependency`, `network`, `memory_leak`, `unknown`     |
| `action`     | `rollback`, `scale_db`, `page_all_teams`, `monitor`, `restart_service`, `escalate`, `investigate` |

Internally, actions are represented as integer indices into these label lists and decoded by the environment.

### Reward Function

The grader (`grader.py`) scores each triage decision with explicit partial credit:

#### Severity Score (weight: **0.3**)

| Prediction vs Ground Truth | Score |
|----------------------------|-------|
| Exact match                | 1.0   |
| One level off (e.g. P1→P2) | 0.5   |
| Two levels off (e.g. P1→P3)| 0.0   |

#### Root Cause Score (weight: **0.4**)

| Prediction vs Ground Truth      | Score |
|---------------------------------|-------|
| Exact match                     | 1.0   |
| Semantically related category   | 0.4   |
| Wrong                           | 0.0   |

Related category groups (provide partial credit):

```python
{
  "database":            ["database", "deployment", "memory_leak"],
  "deployment":          ["deployment", "infra"],
  "infra":               ["infra", "network", "deployment"],
  "external_dependency": ["external_dependency", "network"],
  "network":             ["network", "infra", "external_dependency"],
  "memory_leak":         ["memory_leak", "database", "infra"],
}
```

#### Action Score (weight: **0.3**)

| Prediction                             | Score |
|----------------------------------------|-------|
| Exact match with expected action       | 1.0   |
| Acceptable alternative action          | 0.6   |
| Wrong (no penalty)                     | 0.0   |
| Overreaction: `page_all_teams` on P3   | −0.2  |

#### Total Score Formula

```
total = (severity × 0.3) + (root_cause × 0.4) + (action × 0.3)
total = clamp(total, 0.0, 1.0)
```

An incident is considered **resolved** if `total_score ≥ 0.6`.

---

## Task Definitions

Three tasks of increasing difficulty are defined in `openenv.yaml`:

| Task ID  | Name   | `max_steps` | `difficulty` | Noise Level                         |
|----------|--------|-------------|--------------|-------------------------------------|
| `easy`   | Easy   | 10          | 1            | 1 misleading signal per incident    |
| `medium` | Medium | 20          | 2            | 2 misleading signals per incident   |
| `hard`   | Hard   | 30          | 3            | 3 misleading signals per incident   |

All tasks use `seed=42` for reproducibility. The seed advances deterministically at each step (`seed + step_number`), ensuring every evaluation run produces the same sequence of incidents.

---

## Incident Corpus

The environment ships with **18 hand-crafted incident templates** across 6 root cause categories:

| Category              | Examples                                                         |
|-----------------------|------------------------------------------------------------------|
| `database`            | Postgres write latency, connection pool exhaustion, replica lag  |
| `deployment`          | API contract breakage, ingress misconfiguration, thread pool bug |
| `infra`               | Kubernetes node eviction, volume quota exhaustion, kernel mismatch|
| `external_dependency` | Payment gateway outage, SMS vendor throttling, identity API lag  |
| `network`             | Packet loss from faulty switch, CoreDNS throttling, BGP flap    |
| `memory_leak`         | Unbounded in-process cache growth, library object retention, middleware leak |

Each template includes:
- An **alert text** (the primary signal)
- **True severity**, **true root cause**, and **correct action** (the ground truth)
- **Misleading signals** (injected at difficulty-appropriate counts to challenge agents)

At runtime, `generate_incident()` randomly selects a template, injects a service name, region, incident age, and affected request percentage, then appends `N` misleading signals based on the difficulty setting.

---

## Baseline Agents

Two baseline agents are included in `baseline.py`:

### 1. Heuristic Agent (`heuristic_agent_decision`)

A zero-dependency, keyword-based rule agent. Requires **no API calls**.

- **Severity**: Derived from percentage of affected requests in alert text, or keywords like `timeout`, `outage`.
- **Root cause & action**: Matched against a priority-ordered keyword table covering all 6 categories.

```python
decision = heuristic_agent_decision(incident_text)
# → {"severity": "P1", "root_cause": "database", "action": "scale_db"}
```

### 2. LLM Agent (`llm_agent_decision`)

Uses **GPT-4o-mini** (via OpenAI API) with a strict JSON-only system prompt. Falls back gracefully to keyword extraction or a safe default if the API is unavailable or the response is malformed.

```python
from openai import OpenAI
client = OpenAI(api_key="...")
decision = llm_agent_decision(client, incident_text)
# → {"severity": "P1", "root_cause": "database", "action": "scale_db"}
```

### 3. OpenEnv Inference Agent (`inference.py`)

A benchmark-grade runner that connects to the Hugging Face Inference Router and uses **Qwen/Qwen2.5-72B-Instruct** by default. Produces structured `[START]`, `[STEP]`, and `[END]` log lines consumable by the OpenEnv platform.

Configurable via environment variables:

| Variable       | Default                                    | Description                        |
|----------------|--------------------------------------------|------------------------------------|
| `HF_TOKEN`     | —                                          | Hugging Face API token             |
| `API_KEY`      | —                                          | Alternative API key                |
| `API_BASE_URL` | `https://router.huggingface.co/v1`         | OpenAI-compatible API base URL     |
| `MODEL_NAME`   | `Qwen/Qwen2.5-72B-Instruct`               | Model to use for inference         |

---

## Baseline Scores

Scores are normalized averages across all steps in each task (range: `[0.0, 1.0]`):

| Task        | Heuristic Agent |
|-------------|----------------|
| Easy        | 0.6210         |
| Medium      | 0.6960         |
| Hard        | 0.6903         |
| **Average** | **0.6691**     |

> **Note:** The heuristic agent is a strong baseline due to its keyword-to-action mapping closely mirroring the ground truth label design. LLM agents are expected to score higher on harder tasks with heavier noise by leveraging broader contextual reasoning.

---

## Project Structure

```
sre-incident-agent/
├── app.py                  # FastAPI server — root, /health, /reset endpoints
├── env.py                  # SREIncidentTriageEnv — Gymnasium environment
├── incidents.py            # Deterministic incident generator (18 templates)
├── grader.py               # Scoring logic — partial credit reward function
├── baseline.py             # Heuristic + LLM baseline agents and episode runner
├── inference.py            # OpenEnv inference script (HF router, Qwen2.5)
├── openenv.yaml            # OpenEnv benchmark manifest (tasks, config, grader)
├── pyproject.toml          # Python project metadata and dependencies
├── requirements.txt        # Minimal requirements for Docker builds
├── Dockerfile              # Container image for Hugging Face Spaces (port 7860)
├── validate-submission.sh  # Pre-submission checker (HF ping + docker build + openenv validate)
└── server/
    └── app.py              # Alternate server entry point
```

---

## Local Setup

**Prerequisites:** Python 3.11+, `pip` or [`uv`](https://github.com/astral-sh/uv)

### Install dependencies

```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended — uses uv.lock for reproducible installs)
uv sync
```

### Verify the environment

```bash
python -c "from env import SREIncidentTriageEnv; e = SREIncidentTriageEnv({'max_steps': 1, 'difficulty': 1, 'seed': 42}); print(e.reset())"
```

### (Optional) Set your OpenAI API key for LLM baselines

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Running Baselines

### Run the heuristic baseline directly

```bash
python baseline.py
```

Runs both heuristic and LLM agents across all three tasks and prints a summary table:

```
Summary Table (normalized final scores)
agent      | Easy   | Medium | Hard
-----------+--------+--------+------
heuristic  | 0.6210 | 0.6960 | 0.6903
llm        | 0.xxxx | 0.xxxx | 0.xxxx
```

### Run the OpenEnv inference script

```bash
export HF_TOKEN="hf_..."
python inference.py
```

### Start the FastAPI server locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then visit [`http://localhost:7860/`](http://localhost:7860/) to trigger a live heuristic demo across all tasks, or POST to `/reset` to see an initial environment observation.

---

## API Reference

| Method | Endpoint  | Description                                                                 |
|--------|-----------|-----------------------------------------------------------------------------|
| `GET`  | `/`       | Run heuristic baseline on all tasks and return aggregate scores             |
| `GET`  | `/health` | Liveness check — returns `{"status": "ok"}`                                 |
| `POST` | `/reset`  | Instantiate and reset the environment; returns initial observation and state |

### `GET /` — Response example

```json
{
  "project": "sre-incident-agent",
  "status": "ok",
  "scores": {
    "Easy": 0.621,
    "Medium": 0.696,
    "Hard": 0.6903
  },
  "average_score": 0.6691,
  "environment": "Gymnasium-based SRE incident triage environment with text observations and multi-part discrete actions (severity, root_cause, action)."
}
```

### `POST /reset` — Response example

```json
{
  "status": "ok",
  "observation": {
    "incident_text": "ALERT: Write latency spike on primary Postgres cluster...",
    "step_number": 0,
    "incidents_resolved": 0,
    "current_score": 0.0
  },
  "info": "{'step_number': 0, 'max_steps': 10, ...}"
}
```

---

## Docker Deployment

The included `Dockerfile` builds a minimal Python 3.11-slim image that serves the FastAPI app on port **7860** (the default for Hugging Face Spaces).

```bash
# Build
docker build -t sre-incident-agent .

# Run
docker run -p 7860:7860 \
  -e OPENAI_API_KEY="sk-..." \
  sre-incident-agent
```

The app will be available at `http://localhost:7860`.

---

## OpenEnv Submission

This project is an [OpenEnv](https://openenv.dev/) benchmark environment. The `openenv.yaml` manifest declares the entry point, tasks, and grader.

### Pre-submission checklist

Run the included validator script to confirm your submission is ready:

```bash
bash validate-submission.sh <YOUR_HF_SPACE_URL>
```

The script performs three checks:

1. **HF Space liveness** — `POST /reset` returns HTTP 200
2. **Docker build** — `docker build` succeeds within 600 seconds
3. **OpenEnv validation** — `openenv validate` passes against `openenv.yaml`

All three must pass before submitting.

### Install the OpenEnv CLI

```bash
pip install openenv-core
```

---

## Configuration

The `SREIncidentTriageEnv` accepts a `config` dictionary at construction time:

| Key         | Type  | Default | Description                                  |
|-------------|-------|---------|----------------------------------------------|
| `max_steps` | `int` | `10`    | Maximum number of incidents per episode      |
| `difficulty`| `int` | `1`     | Noise level: 1 (easy), 2 (medium), 3 (hard)  |
| `seed`      | `int` | `42`    | Base seed for deterministic incident generation |

```python
from env import SREIncidentTriageEnv

env = SREIncidentTriageEnv({
    "max_steps": 20,
    "difficulty": 2,
    "seed": 1337,
})
obs, info = env.reset()
```

The seed advances deterministically at each step (`seed + step_number`), so episodes are fully reproducible across runs, languages, and frameworks.