"""Seed-based incident generator for SRE training and evaluation."""

from __future__ import annotations

import random
from typing import Any, Dict, List


INCIDENT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "root_cause_category": "database",
        "alert_text": "DB write latency spike on primary Postgres cluster; checkout API timing out.",
        "true_severity": "P1",
        "true_root_cause": "DB primary underprovisioned after traffic surge",
        "correct_action": "Increase DB capacity and throttle expensive write paths",
        "misleading_signals": [
            "Deploy pipeline promoted a new checkout rollout 15 minutes earlier",
            "Kubernetes manifest for cart-service was hotfixed before alert onset",
            "Blue-green deploy cutover happened in one region",
            "Canary release gate reported flaky rollout health checks",
        ],
        "contradictory_signals": [
            "Deployment logs confirm all rollouts completed successfully with zero errors — this appears to be a DB-layer issue masked by deploy timing",
            "Correlation analysis shows latency spike started 8 minutes BEFORE the deploy completed, ruling out deployment as root cause",
        ],
    },
    {
        "root_cause_category": "database",
        "alert_text": "Database connection pool exhaustion in payments DB; deadlock count rising.",
        "true_severity": "P1",
        "true_root_cause": "Long-running transaction causing lock contention",
        "correct_action": "Terminate blocking transaction and deploy lock-safe query path",
        "misleading_signals": [
            "New deployment manifest added sidecar env vars to payments pods",
            "Rollout controller retried a failed canary promotion twice",
            "Helm chart deploy introduced a rewritten readiness probe",
            "Feature branch release train merged during the same window",
        ],
        "contradictory_signals": [
            "All deployment diffs reviewed — no schema or query changes in this release cycle, pointing to organic DB contention",
            "Identical deadlock pattern observed last month with no deploys in flight, confirming this is a recurring DB concurrency issue",
        ],
    },
    {
        "root_cause_category": "database",
        "alert_text": "Postgres replica lag exceeded 12 minutes; DB reads returning stale order status.",
        "true_severity": "P2",
        "true_root_cause": "Replication I/O throttled by storage burst limit",
        "correct_action": "Move replica to higher IOPS storage and reduce read pressure",
        "misleading_signals": [
            "Ingress rollout applied a new manifest with header rewrite",
            "Deploy bot posted successful progressive rollout events",
            "Rollback marker appeared for a recently shipped release",
            "Argo CD sync wave reordered deployment hooks",
        ],
        "contradictory_signals": [
            "Storage team confirms burst IOPS limit was hit — replica starvation is a known issue on this volume class, unrelated to any deploy",
            "Network team verified zero packet loss between primary and replica — this is purely a storage throughput bottleneck",
        ],
    },
    {
        "root_cause_category": "deployment",
        "alert_text": "Error rate climbed immediately after cart-service deployment v2026.04.1.",
        "true_severity": "P1",
        "true_root_cause": "Backward-incompatible API contract in new deployment",
        "correct_action": "Rollback deployment and gate release behind contract tests",
        "misleading_signals": [
            "Replica lag on orders DB jumped to 90 seconds",
            "Connection pool saturation warnings appeared in checkout",
            "Deadlock counter increased on primary Postgres",
            "Read replica reported stale rows during peak",
        ],
        "contradictory_signals": [
            "DB metrics show connection pool and replica lag spikes started exactly when new pods began serving traffic — classic deployment-induced cascade",
            "Git blame confirms the breaking change: checkout-service v2026.04.1 removed a required header that downstream services depend on",
        ],
    },
    {
        "root_cause_category": "deployment",
        "alert_text": "Canary looked healthy, but global rollout caused checkout 502s.",
        "true_severity": "P1",
        "true_root_cause": "Misconfigured ingress path rewrite in deployment manifest",
        "correct_action": "Revert ingress config and add staged traffic validation",
        "misleading_signals": [
            "Primary database connection pool hit max clients",
            "Replica lag alert fired for the catalog cluster",
            "Query planner switched to sequential scan on hot table",
            "Transaction lock wait time rose in payments DB",
        ],
        "contradictory_signals": [
            "DB load is a symptom, not the cause — connection surge correlates exactly with the ingress rewrite sending malformed requests that trigger retry storms",
            "Canary passed because it only received 1% of traffic; the ingress path rewrite bug only manifests under full production routing rules",
        ],
    },
    {
        "root_cause_category": "deployment",
        "alert_text": "Worker queue backlog exploded after release of image-processor service.",
        "true_severity": "P2",
        "true_root_cause": "Thread pool size reduced accidentally in new config",
        "correct_action": "Restore worker concurrency and redeploy",
        "misleading_signals": [
            "Database pool timeout errors increased on worker startup",
            "Replica read consistency lag crossed warning threshold",
            "Postgres autovacuum on queue table ran unusually long",
            "Connection churn exhausted DB max_connections briefly",
        ],
        "contradictory_signals": [
            "Config diff shows worker_threads changed from 16 to 4 in the new release — this is not a DB issue, it is a deployment config regression",
            "DB metrics returned to normal after reverting the image-processor config, confirming the deploy caused the queue backlog",
        ],
    },
    {
        "root_cause_category": "infra",
        "alert_text": "Kubernetes node pressure evicting pods in production cluster.",
        "true_severity": "P1",
        "true_root_cause": "Node pool capacity mismatch after autoscaler policy change",
        "correct_action": "Expand node pool and revert autoscaler policy",
        "misleading_signals": [
            "CoreDNS reported packet drops on east-west traffic",
            "BGP flap advisory posted by edge network provider",
            "Cross-zone TCP retransmits increased sharply",
            "ISP path instability caused intermittent connection resets",
        ],
    },
    {
        "root_cause_category": "infra",
        "alert_text": "Persistent volume attach failures causing database pod crash loops.",
        "true_severity": "P1",
        "true_root_cause": "Cloud volume quota exhausted in region",
        "correct_action": "Free or increase volume quota and reschedule impacted pods",
        "misleading_signals": [
            "Service-to-service DNS lookup latency spiked",
            "Top-of-rack switch port showed burst packet loss",
            "Inter-region route convergence delayed new flows",
            "NAT gateway reset count rose in one subnet",
        ],
    },
    {
        "root_cause_category": "infra",
        "alert_text": "Service mesh sidecars failing readiness on newly provisioned nodes.",
        "true_severity": "P2",
        "true_root_cause": "Kernel parameter mismatch in node image",
        "correct_action": "Patch node image and cordon/drain bad nodes",
        "misleading_signals": [
            "Ingress endpoints saw periodic SYN retransmit spikes",
            "DNS resolver cache miss burst triggered lookup failures",
            "Edge provider reported route dampening event",
            "Cross-AZ packet loss alarm cleared after 2 minutes",
        ],
    },
    {
        "root_cause_category": "external_dependency",
        "alert_text": "Payment authorization failures from third-party gateway exceeded SLO.",
        "true_severity": "P1",
        "true_root_cause": "Upstream payment provider partial outage",
        "correct_action": "Fail over to secondary provider and enable graceful degradation",
        "misleading_signals": [
            "Checkout deploy rollout introduced a new payment adapter manifest",
            "Canary release for billing service failed health gate",
            "Rollback event triggered after staged deploy verification",
            "Helm release promotion happened minutes before incident",
        ],
    },
    {
        "root_cause_category": "external_dependency",
        "alert_text": "SMS notifications delayed by 15 minutes for most users.",
        "true_severity": "P2",
        "true_root_cause": "Vendor messaging API throttling requests",
        "correct_action": "Apply rate-aware backoff and route through backup SMS vendor",
        "misleading_signals": [
            "Deploy controller applied a manifest patch to notifier service",
            "Blue-green rollout switched traffic for messaging workers",
            "Release pipeline retried a failed canary for notifications",
            "Kubernetes rollout history shows emergency hotfix deploy",
        ],
    },
    {
        "root_cause_category": "external_dependency",
        "alert_text": "Identity verification checks timing out intermittently.",
        "true_severity": "P3",
        "true_root_cause": "Third-party verification API latency regression",
        "correct_action": "Use async fallback flow and increase timeout with circuit breaker",
        "misleading_signals": [
            "Gateway deployment manifest changed timeout defaults",
            "Progressive rollout paused due to failing synthetic checks",
            "Release manager initiated rollback of identity-service deploy",
            "Argo rollout analysis job flagged new pod errors",
        ],
    },
    {
        "root_cause_category": "network",
        "alert_text": "Cross-zone traffic experiencing packet loss; API p99 latency doubled.",
        "true_severity": "P1",
        "true_root_cause": "Faulty top-of-rack switch dropping east-west packets",
        "correct_action": "Reroute traffic and isolate faulty network segment",
        "misleading_signals": [
            "Kubernetes node pressure eviction warnings appeared in prod",
            "Control-plane reported node NotReady due to disk pressure",
            "Cluster autoscaler reduced node pool below steady-state",
            "Pod scheduling stalled from insufficient allocatable CPU",
        ],
    },
    {
        "root_cause_category": "network",
        "alert_text": "Intermittent DNS resolution failures in service-to-service calls.",
        "true_severity": "P2",
        "true_root_cause": "CoreDNS pods throttled by CPU limits",
        "correct_action": "Increase DNS capacity and tune client-side resolver retries",
        "misleading_signals": [
            "Kubernetes node drain left workloads pending on tainted nodes",
            "Container runtime errors increased on one node pool",
            "Cluster reported sustained memory pressure in gateway nodes",
            "DaemonSet rollout failed on newly provisioned worker nodes",
        ],
    },
    {
        "root_cause_category": "network",
        "alert_text": "Ingress requests from one ISP failing with connection resets.",
        "true_severity": "P2",
        "true_root_cause": "BGP route flap at upstream edge provider",
        "correct_action": "Shift ingress to unaffected edge and coordinate with provider NOC",
        "misleading_signals": [
            "Scheduler logged node pressure preemption events",
            "Kubernetes API showed repeated pod eviction notices",
            "Node image rollout introduced cgroup misconfiguration",
            "Autoscaler policy reduced infra headroom during peak",
        ],
    },
    {
        "root_cause_category": "memory_leak",
        "alert_text": "Recommendation service memory usage climbs until OOM kills pods.",
        "true_severity": "P1",
        "true_root_cause": "Unbounded in-process cache growth after feature toggle",
        "correct_action": "Disable feature toggle and cap cache with eviction policy",
        "misleading_signals": [
            "Primary Postgres reported replica lag and stale reads",
            "Connection pool queue length exceeded max wait threshold",
            "Deadlock detector logged cyclic lock contention",
            "Database failover simulation caused temporary writer switch",
        ],
    },
    {
        "root_cause_category": "memory_leak",
        "alert_text": "Batch worker RSS increases each run; node memory pressure alerts firing.",
        "true_severity": "P2",
        "true_root_cause": "Library upgrade introduced object retention bug",
        "correct_action": "Rollback library version and apply memory profiling fix",
        "misleading_signals": [
            "Replica lag alarm triggered on jobs database",
            "Connection pool reached saturation under worker load",
            "Slow query log shows lock waits on batch table",
            "Read-after-write inconsistency reported from replica endpoint",
        ],
    },
    {
        "root_cause_category": "memory_leak",
        "alert_text": "API gateway containers restart every 40 minutes due to heap exhaustion.",
        "true_severity": "P2",
        "true_root_cause": "Connection metadata retained by custom middleware",
        "correct_action": "Patch middleware leak and temporarily reduce keep-alive duration",
        "misleading_signals": [
            "Write transactions queued behind DB connection pool exhaustion",
            "Replica replay delay crossed incident warning threshold",
            "Database lock timeout errors rose after traffic spike",
            "Primary node failover drill increased query latency briefly",
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

    rng = random.Random((seed * 13 + 97) % 999983)
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

    # At hard difficulty, inject contradictory signals that actively argue
    # for the wrong root cause with plausible-sounding explanations.
    if difficulty == 3 and template.get("contradictory_signals"):
        contradictory = rng.choice(template["contradictory_signals"])
        incident_text_lines.append(f"Analyst note: {contradictory}")

    return {
        "incident_text": "\n".join(incident_text_lines),
        "true_severity": template["true_severity"],
        "true_root_cause": template["true_root_cause"],
        "correct_action": template["correct_action"],
    }


if __name__ == "__main__":
    sample = generate_incident(seed=42, difficulty=2)
    print("Sample incident:")
    print(sample["incident_text"])
    print("\nAnswer key:")
    print(
        {
            "true_severity": sample["true_severity"],
            "true_root_cause": sample["true_root_cause"],
            "correct_action": sample["correct_action"],
        }
    )
