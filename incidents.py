"""Seed-based incident generator for SRE training and evaluation."""

from __future__ import annotations

import random
from typing import Any, Dict, List


INCIDENT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "root_cause_category": "database",
        "alert_text": "Write latency spike on primary Postgres cluster; checkout API timing out.",
        "true_severity": "P1",
        "true_root_cause": "DB primary underprovisioned after traffic surge",
        "correct_action": "Increase DB capacity and throttle expensive write paths",
        "misleading_signals": [
            "CPU briefly high on API pods due to retries",
            "A/B test rollout happened 20 minutes earlier",
            "Background analytics job log spam increased",
            "CDN cache hit ratio dipped in one region",
        ],
    },
    {
        "root_cause_category": "database",
        "alert_text": "Connection pool exhaustion in payments service; deadlock count rising.",
        "true_severity": "P1",
        "true_root_cause": "Long-running transaction causing lock contention",
        "correct_action": "Terminate blocking transaction and deploy lock-safe query path",
        "misleading_signals": [
            "Node reboot completed around same time",
            "Minor packet loss alarm from edge switch",
            "Feature flag was toggled for 2 percent of users",
            "Cache eviction warnings from unrelated service",
        ],
    },
    {
        "root_cause_category": "database",
        "alert_text": "Replica lag exceeded 12 minutes; reads returning stale order status.",
        "true_severity": "P2",
        "true_root_cause": "Replication I/O throttled by storage burst limit",
        "correct_action": "Move replica to higher IOPS storage and reduce read pressure",
        "misleading_signals": [
            "Search index rebuild started this hour",
            "TLS cert auto-renewal occurred successfully",
            "Slight increase in 429 responses on auth endpoint",
            "One canary pod restarted due to liveness timeout",
        ],
    },
    {
        "root_cause_category": "deployment",
        "alert_text": "Error rate climbed immediately after cart-service deployment v2026.04.1.",
        "true_severity": "P1",
        "true_root_cause": "Backward-incompatible API contract in new deployment",
        "correct_action": "Rollback deployment and gate release behind contract tests",
        "misleading_signals": [
            "Redis memory fragmentation warning appears in dashboard",
            "Spike in sign-ins from promotional campaign",
            "Intermittent DNS warning from internal tooling",
            "Daily batch ETL job started on schedule",
        ],
    },
    {
        "root_cause_category": "deployment",
        "alert_text": "Canary looked healthy, but global rollout caused checkout 502s.",
        "true_severity": "P1",
        "true_root_cause": "Misconfigured ingress path rewrite in deployment manifest",
        "correct_action": "Revert ingress config and add staged traffic validation",
        "misleading_signals": [
            "DB CPU is moderately elevated",
            "Latency jitter observed in one cloud zone",
            "Synthetic tests from APAC briefly flaked",
            "Cache warmup was incomplete after autoscale",
        ],
    },
    {
        "root_cause_category": "deployment",
        "alert_text": "Worker queue backlog exploded after release of image-processor service.",
        "true_severity": "P2",
        "true_root_cause": "Thread pool size reduced accidentally in new config",
        "correct_action": "Restore worker concurrency and redeploy",
        "misleading_signals": [
            "Object storage p95 latency increased slightly",
            "Security scanner ran during the same window",
            "Message broker disk usage warning from another cluster",
            "Unrelated auth service had one restart",
        ],
    },
    {
        "root_cause_category": "infra",
        "alert_text": "Kubernetes node pressure evicting pods in production cluster.",
        "true_severity": "P1",
        "true_root_cause": "Node pool capacity mismatch after autoscaler policy change",
        "correct_action": "Expand node pool and revert autoscaler policy",
        "misleading_signals": [
            "Recent app deployment looked suspicious to on-call",
            "External API had a short-lived 500 spike",
            "Certificate rotation completed this morning",
            "One AZ reported elevated control-plane latency",
        ],
    },
    {
        "root_cause_category": "infra",
        "alert_text": "Persistent volume attach failures causing database pod crash loops.",
        "true_severity": "P1",
        "true_root_cause": "Cloud volume quota exhausted in region",
        "correct_action": "Free or increase volume quota and reschedule impacted pods",
        "misleading_signals": [
            "API gateway 4xx increased due to bot traffic",
            "Build pipeline had intermittent timeout",
            "Tracing backend dropped spans briefly",
            "Background cache rebalancing in progress",
        ],
    },
    {
        "root_cause_category": "infra",
        "alert_text": "Service mesh sidecars failing readiness on newly provisioned nodes.",
        "true_severity": "P2",
        "true_root_cause": "Kernel parameter mismatch in node image",
        "correct_action": "Patch node image and cordon/drain bad nodes",
        "misleading_signals": [
            "Error logs mention expired token but refresh succeeds",
            "Marketing traffic increased during campaign launch",
            "A stale alert from previous incident reopened",
            "Backup job consumed extra network bandwidth",
        ],
    },
    {
        "root_cause_category": "external_dependency",
        "alert_text": "Payment authorization failures from third-party gateway exceeded SLO.",
        "true_severity": "P1",
        "true_root_cause": "Upstream payment provider partial outage",
        "correct_action": "Fail over to secondary provider and enable graceful degradation",
        "misleading_signals": [
            "CPU usage rose in checkout pods from retry load",
            "Recent app release is under investigation",
            "Internal DNS cache miss rate is higher than normal",
            "Redis replica lag alert fired once",
        ],
    },
    {
        "root_cause_category": "external_dependency",
        "alert_text": "SMS notifications delayed by 15 minutes for most users.",
        "true_severity": "P2",
        "true_root_cause": "Vendor messaging API throttling requests",
        "correct_action": "Apply rate-aware backoff and route through backup SMS vendor",
        "misleading_signals": [
            "Queue depth rose after feature launch",
            "One worker node had elevated memory but stable",
            "Auth token rotation happened recently",
            "Internal NTP sync warning appeared once",
        ],
    },
    {
        "root_cause_category": "external_dependency",
        "alert_text": "Identity verification checks timing out intermittently.",
        "true_severity": "P3",
        "true_root_cause": "Third-party verification API latency regression",
        "correct_action": "Use async fallback flow and increase timeout with circuit breaker",
        "misleading_signals": [
            "Application logs show sporadic DB reconnect messages",
            "Cache hit ratio dipped by 3 percent",
            "Canary deploy happened in unrelated service",
            "Node autoscaler scaled down one group",
        ],
    },
    {
        "root_cause_category": "network",
        "alert_text": "Cross-zone traffic experiencing packet loss; API p99 latency doubled.",
        "true_severity": "P1",
        "true_root_cause": "Faulty top-of-rack switch dropping east-west packets",
        "correct_action": "Reroute traffic and isolate faulty network segment",
        "misleading_signals": [
            "Recent code merge touched timeout constants",
            "One DB replica showed moderate lag",
            "CDN provider posted advisory unrelated to region",
            "Log ingestion queue is above baseline",
        ],
    },
    {
        "root_cause_category": "network",
        "alert_text": "Intermittent DNS resolution failures in service-to-service calls.",
        "true_severity": "P2",
        "true_root_cause": "CoreDNS pods throttled by CPU limits",
        "correct_action": "Increase DNS capacity and tune client-side resolver retries",
        "misleading_signals": [
            "Auth service deploy happened earlier",
            "Background data export consumed I/O",
            "Brief clock skew warning on one node",
            "Trace sampling rate changed for observability tests",
        ],
    },
    {
        "root_cause_category": "network",
        "alert_text": "Ingress requests from one ISP failing with connection resets.",
        "true_severity": "P2",
        "true_root_cause": "BGP route flap at upstream edge provider",
        "correct_action": "Shift ingress to unaffected edge and coordinate with provider NOC",
        "misleading_signals": [
            "Slight rise in app GC pauses",
            "A non-critical migration ran in background",
            "Old alert about cert expiry resurfaced",
            "Deployment dashboard shows benign warning",
        ],
    },
    {
        "root_cause_category": "memory_leak",
        "alert_text": "Recommendation service memory usage climbs until OOM kills pods.",
        "true_severity": "P1",
        "true_root_cause": "Unbounded in-process cache growth after feature toggle",
        "correct_action": "Disable feature toggle and cap cache with eviction policy",
        "misleading_signals": [
            "CPU spikes align with traffic peak",
            "Search index compaction started recently",
            "A/B testing service had one timeout",
            "Network retransmits increased in one zone",
        ],
    },
    {
        "root_cause_category": "memory_leak",
        "alert_text": "Batch worker RSS increases each run; node memory pressure alerts firing.",
        "true_severity": "P2",
        "true_root_cause": "Library upgrade introduced object retention bug",
        "correct_action": "Rollback library version and apply memory profiling fix",
        "misleading_signals": [
            "Disk throughput on queue broker is elevated",
            "Ingress 4xx rose due to invalid client requests",
            "Deployment pipeline queued for longer than usual",
            "DB maintenance vacuum running concurrently",
        ],
    },
    {
        "root_cause_category": "memory_leak",
        "alert_text": "API gateway containers restart every 40 minutes due to heap exhaustion.",
        "true_severity": "P2",
        "true_root_cause": "Connection metadata retained by custom middleware",
        "correct_action": "Patch middleware leak and temporarily reduce keep-alive duration",
        "misleading_signals": [
            "Edge firewall logged transient deny events",
            "Slight packet loss seen in one subnet",
            "Canary rollout for another service completed",
            "Metrics scraper missed one interval",
        ],
    },
]


_SERVICES = [
    "checkout",
    "payments",
    "orders",
    "catalog",
    "auth",
    "notifications",
    "search",
    "gateway",
]

_REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"]


def _misleading_count_for_difficulty(difficulty: int) -> int:
    if difficulty == 1:
        return 1
    if difficulty == 2:
        return 2
    if difficulty == 3:
        return 3
    raise ValueError("difficulty must be 1 (easy), 2 (medium), or 3 (hard)")


def generate_incident(seed: int, difficulty: int) -> Dict[str, str]:
    """Generate a deterministic but slightly randomized SRE incident.

    Args:
        seed: Seed controlling deterministic selection and randomization.
        difficulty: 1 (easy), 2 (medium), 3 (hard). Higher values inject more
            misleading signals into the incident text.

    Returns:
        Dict with incident_text, true_severity, true_root_cause, correct_action.
    """

    rng = random.Random(seed)
    template = rng.choice(INCIDENT_TEMPLATES)

    misleading_count = min(
        _misleading_count_for_difficulty(difficulty),
        len(template["misleading_signals"]),
    )
    selected_misleading = rng.sample(template["misleading_signals"], k=misleading_count)

    service = rng.choice(_SERVICES)
    region = rng.choice(_REGIONS)
    minutes = rng.randint(4, 27)
    impact = rng.randint(18, 92)

    incident_text_lines = [
        f"ALERT: {template['alert_text']}",
        f"Context: service={service}, region={region}, incident_age={minutes}m, affected_requests={impact}%.",
    ]

    if selected_misleading:
        incident_text_lines.append("Additional telemetry (may include noise):")
        for signal in selected_misleading:
            incident_text_lines.append(f"- {signal}")

    return {
        "incident_text": "\n".join(incident_text_lines),
        "true_severity": template["true_severity"],
        "true_root_cause": template["true_root_cause"],
        "correct_action": template["correct_action"],
    }
