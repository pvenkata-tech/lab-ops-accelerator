from prometheus_client import Counter, Gauge, Histogram

EXCEPTIONS_PROCESSED = Counter(
    "exceptions_processed_total",
    "Total specimen exceptions processed by the agent",
    ["exception_type", "disposition"],
)

HITL_RATE = Gauge(
    "hitl_rate",
    "Fraction of cases routed to human review (rolling)",
)

SUPERVISOR_OVERRIDES = Counter(
    "supervisor_overrides_total",
    "Cases where supervisor decision differed from agent recommendation",
)

EXCEPTION_RESOLUTION_SECONDS = Histogram(
    "exception_resolution_seconds",
    "Time from specimen event to disposition decision",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

BEDROCK_TOKENS = Counter(
    "bedrock_tokens_total",
    "Total tokens consumed from Bedrock",
    ["model", "token_type"],
)
