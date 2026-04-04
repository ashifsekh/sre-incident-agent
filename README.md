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

## 1. Project Description and Real-World Motivation

Modern production systems generate noisy, high-volume alerts. During incidents, responders must quickly decide three things:

- How severe the incident is
- What the most likely root cause is
- What immediate action should be taken

This project provides a lightweight benchmark environment for training and evaluating incident triage agents. It simulates realistic SRE scenarios with noisy signals and scores decisions with partial credit, so both rule-based and LLM-based agents can be compared under increasing difficulty.

The environment is useful for:

- Rapid prototyping of incident-response policies
- Evaluating LLM decision quality under ambiguity
- Benchmarking heuristics against model-driven approaches
- Demonstrating SRE triage workflows in a reproducible setup

## 2. Observation Space

The environment returns a structured, text-first observation with:

- `incident_text`: The full alert context and injected telemetry noise
- `step_number`: Current step in the episode
- `incidents_resolved`: Count of incidents solved with strong reward signal
- `current_score`: Running average score during the episode

## 3. Action Space

Each action is multi-part and must include all three fields:

- `severity`: P1, P2, P3
- `root_cause`: database, deployment, infra, external_dependency, network, memory_leak, unknown
- `action`: rollback, scale_db, page_all_teams, monitor, restart_service, escalate, investigate

## 4. Reward Function Breakdown

- Severity score (weight 0.3): exact=1.0, one level off=0.5, two levels off=0.0
- Root cause score (weight 0.4): exact=1.0, related category=0.4, wrong=0.0
- Action score (weight 0.3): exact=1.0, acceptable alternative=0.6, wrong=0.0, overreaction penalty=-0.2

`total = (severity * 0.3) + (root_cause * 0.4) + (action * 0.3)` clamped to [0.0, 1.0]

## 5. Task Definitions

| Task   | max_steps | difficulty |
|--------|-----------|------------|
| Easy   | 10        | 1          |
| Medium | 20        | 2          |
| Hard   | 30        | 3          |

## 6. Setup and Local Run
```bash
pip install -r requirements.txt
python baseline.py
uvicorn app:app --host 0.0.0.0 --port 7860
```

## 7. Baseline Scores

| Task        | Heuristic Agent |
|-------------|----------------|
| Easy        | 0.6210         |
| Medium      | 0.6960         |
| Hard        | 0.6903         |
| **Average** | **0.6691**     |

## 8. Deployment

Dockerfile included. Deploy to Hugging Face Spaces with Docker SDK on port 7860.
Optionally add `OPENAI_API_KEY` in Space Settings → Variables and secrets for LLM baseline.