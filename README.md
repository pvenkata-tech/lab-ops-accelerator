# Lab Ops Accelerator

**A Forward Deployed AI Engine for genetic testing lab workflows. Accelerates specimen exception resolution by 10-100x using LangGraph and standardized MCP (Model Context Protocol) integrations.**

---

## The Problem It Solves

A high-throughput genetic testing lab processes thousands of specimens daily. When a specimen fails quality control (wrong tube type, insufficient volume, hemolysis), a lab technician manually looks up the rejection protocol, decides on disposition, and triggers downstream notifications. At volume, this exception-handling loop is the single largest source of avoidable turnaround-time (TAT) delay.

**Lab Ops Accelerator collapses that loop to seconds.** By leveraging **Model Context Protocol (MCP)** to standardize connections to the LIMS and EHR, the agent seamlessly classifies the exception, retrieves protocols via RAG, and executes the disposition. Standard cases are auto-routed; ambiguous cases are held for a lab supervisor's one-click review.

---

## Architecture: MCP-Driven Integrations

To handle the "messy last-mile of enterprise systems" (auth, schema drift, rate limits), this architecture decouples the AI reasoning from the business systems using **MCP Servers**. Adding a new upstream data source is simply configuring a new MCP server, requiring zero changes to the LangGraph orchestrator.

```text
Specimen Event (LIMS webhook via MCP)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                           │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │   Intake     │──▶│  QC          │──▶│  Exception         │  │
│  │  Classifier  │   │  Evaluator   │   │  Router            │  │
│  └──────────────┘   └──────────────┘   └────────┬───────────┘  │
│                                                  │              │
│                           ┌──────────────────────┤              │
│                     confidence ≥ threshold   confidence < threshold
│                           │                      │              │
│                           ▼                      ▼              │
│                  ┌────────────────┐    ┌──────────────────┐    │
│                  │  MCP Client    │    │  HITL interrupt() │    │
│                  │  (Dispatcher)  │    │  supervisor review│    │
│                  └────────┬───────┘    └──────────┬────────┘   │
│                           │                       │             │
│                           │            supervisor decision      │
│                           │                       │             │
│                           ▼                       ▼             │
│                 [ LIMS / EHR MCP Servers ]                      │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
   Prometheus metrics → Grafana KPI dashboard
   Structured logs → CloudWatch / LangSmith tracing
```

### Data Flow

| Stage | What Happens | Systems Touched |
|-------|-------------|-----------------|
| **Intake Classification** | Agent reads specimen event payload, classifies exception type | LIMS MCP server |
| **QC Evaluation** | Retrieves applicable QC thresholds and rejection criteria via RAG | pgvector knowledge base |
| **Exception Router** | Determines disposition (auto-route vs. HITL) based on exception type, protocol confidence, and specimen criticality | Protocol knowledge base |
| **HITL (when triggered)** | Checkpoints state to Postgres; returns HTTP 202; supervisor notified | PostgreSQL checkpoint store |
| **MCP Client (Dispatcher)** | Alerts ordering physician (EHR), triggers retest order in LIMS if indicated, updates specimen status | LIMS MCP server, EHR MCP server |

---

## Key Technologies

- **Runtime:** Python 3.12, FastAPI, Uvicorn
- **Orchestration:** LangGraph with `interrupt()` for HITL state management; PostgreSQL checkpointing preserves state across human review delays
- **Model Layer:** AWS Bedrock — Claude 3.5 Sonnet for exception reasoning; Amazon Titan Embeddings v2 for protocol retrieval
- **Knowledge Base:** PostgreSQL 16 with pgvector; stores specimen handling protocols, QC threshold tables, rejection criteria, and retest decision trees
- **MCP Servers:** LIMS integration and EHR notification are exposed to the agent as standardized Model Context Protocol servers — the orchestrator calls tools, not bespoke SDKs. Adding a new upstream source is a new MCP server, not an orchestrator code change
- **Infrastructure:** Docker Compose (local), Terraform/AWS Fargate (production) — 100% AWS stack aligned with enterprise security boundaries
- **Observability:** Prometheus metrics, Grafana dashboards, LangSmith tracing; structured per-call logs including prompt, model, confidence, disposition, and override flag

---

## Production KPIs Tracked

The agent is instrumented for the KPIs that matter to Lab Ops leadership, not just model accuracy:

| KPI | Definition | Instrumented Via |
|-----|-----------|-----------------|
| **Throughput** | Exception cases resolved per hour vs. manual baseline | Prometheus counter `exceptions_processed_total` |
| **TAT impact** | Minutes from specimen event to disposition decision | Histogram `exception_resolution_seconds` |
| **Auto-resolution rate** | % of exceptions routed without human review | Gauge `hitl_rate` (inverse) |
| **Human intervention rate** | % of cases reaching HITL; leading indicator of model drift | Gauge `hitl_rate` |
| **Override rate** | % of auto-dispositions where supervisor later corrected | Counter `supervisor_overrides_total` |
| **Cost per case** | Bedrock inference cost + human review time per exception | Logged per call, aggregated in dashboard |
| **Protocol retrieval quality** | Top-1 retrieved protocol relevance, measured on golden eval set | Eval runner output |

Override rate spikes before business KPIs degrade — it is the canary.

---

## HITL Workflow

```
Agent evaluates exception
        │
 confidence < threshold?
        │ YES
        ▼
interrupt() called
→ State checkpointed to Postgres
→ HTTP 202 returned to caller
→ Supervisor notified via EHR MCP server

[supervisor reviews in approval surface]
        │
POST /v1/resume {thread_id, decision, rationale}
        │
        ▼
Graph resumes from checkpoint
→ MCP Client (Dispatcher) executes supervisor's decision
→ LIMS status updated
→ Override logged if decision differs from agent recommendation
```

The approval surface is designed for lab supervisors, not engineers. It shows:
- Exception type and specimen metadata in plain language
- The agent's recommended disposition with the retrieved protocol
- The confidence score and why it triggered HITL
- **Approve / Override / Escalate** — three actions, no code

A supervisor can review and act without understanding the model behind it. That is the design intent.

---

## Evals

Every material change to the model, prompt, protocol knowledge base, or routing logic triggers a full eval run before deployment.

**Offline eval:** Golden dataset of 500+ known-correct exception dispositions spanning all major exception categories. The eval runner measures accuracy, protocol retrieval precision, and HITL trigger rate against ground truth.

**Shadow mode:** Before any traffic shift, the agent runs in parallel with the manual exception process — no live LIMS writes, full output comparison. Shadow mode is the gate, not the model accuracy score.

```bash
# Run offline eval
python -m lab_ops_accelerator.evals.eval_runner --dataset samples/golden_dataset.json

# Output: accuracy, top-1 retrieval precision, HITL trigger rate, cost per case
```

Material changes that require a mandatory eval re-run before deployment:
- Model version change (Bedrock model ID)
- Any prompt modification
- Protocol knowledge base update (new protocols, revised QC thresholds)
- Routing logic or confidence threshold adjustment
- Upstream schema change in LIMS or EHR event payloads (i.e. an MCP server contract change)

---

## API Surface

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe — no DB dependency |
| `/v1/ready` | GET | Readiness check: DB connectivity, Bedrock reachability, knowledge base seed status |
| `/v1/process` | POST | Submit specimen exception event; returns 200 (auto-resolved) or 202 (HITL triggered) |
| `/v1/resume` | POST | Supervisor submits decision for a paused thread; resumes graph from checkpoint |
| `/metrics` | GET | Prometheus text format — all production KPIs |

### Example: Submit exception event

```bash
curl -X POST http://localhost:8000/v1/process \
  -H "Content-Type: application/json" \
  -d @samples/specimen_event_sample.json
```

**Auto-resolved (200):**
```json
{
  "thread_id": "spec-20240315-04821",
  "status": "resolved",
  "disposition": "retest_required",
  "protocol_applied": "SOP-LAB-047 Insufficient Volume — EDTA Tube",
  "notification_sent": true,
  "resolution_seconds": 4.2,
  "confidence": 0.94
}
```

**HITL triggered (202):**
```json
{
  "thread_id": "spec-20240315-04891",
  "status": "pending_review",
  "agent_recommendation": "reject",
  "confidence": 0.61,
  "protocol_retrieved": "SOP-LAB-012 Hemolysis — Ambiguous Grade",
  "review_url": "/v1/review/spec-20240315-04891"
}
```

---

## Configuration

Copy `.env.example` to `.env` and populate:

```env
# AWS Bedrock — required
AWS_REGION=us-east-1
BEDROCK_CLAUDE_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0

# PostgreSQL — required
DATABASE_URL=postgresql+asyncpg://labops:labops@localhost:5432/labops
CHECKPOINT_DATABASE_URL=postgresql://labops:labops@localhost:5432/labops

# LIMS integration (MCP server) — required
LIMS_API_BASE_URL=http://lims-mock:8001
LIMS_API_KEY=dev-lims-key

# EHR / Notification (MCP server) — required
EHR_WEBHOOK_URL=http://ehr-mock:8002/notify
EHR_API_KEY=dev-ehr-key

# Observability — optional (recommended for production)
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=lab-ops-accelerator

# Agent tuning
HITL_CONFIDENCE_THRESHOLD=0.80
EMBEDDING_DIMENSIONS=1024
```

Startup validation enforces all required fields — the service fails loudly at boot rather than serving wrong results silently.

---

## Deployment

### Local Development

```bash
cp .env.example .env
# populate AWS credentials and API keys

docker compose up --build
```

Compose brings up: PostgreSQL with pgvector, LIMS mock, EHR mock, Prometheus, Grafana, and the Accelerator service. The knowledge base seeds automatically from `samples/protocols/` on first start.

### Production (AWS)

Terraform provisions:
- ECS Fargate tasks in private subnets
- RDS PostgreSQL (managed, encrypted at rest)
- Bedrock access via IAM task role — no hardcoded credentials
- ALB with TLS termination
- S3 for audit log archival
- Secrets Manager for all API keys and credentials
- CloudWatch log groups with structured log retention

```bash
cd terraform
terraform init
terraform plan -var-file=prod.tfvars
terraform apply
```

IAM task roles are scoped to the specific Bedrock model IDs and RDS instance — least-privilege by construction.

---

## Security

- All Bedrock calls stay inside the AWS boundary — no PHI or specimen data crosses to a public API endpoint
- LIMS and EHR credentials loaded from Secrets Manager; never in environment literals in production
- Every agent action is auditable: input payload, model, protocol retrieved, confidence score, disposition, human override flag — logged per call
- Audit log archived to S3 with object-level access logging — immutable trail for GxP and HIPAA audit purposes
- Rollback path: every deployment is tagged; rollback is a Terraform apply of the prior tag
- HITL approval surface requires an authenticated session — reviewer identity is captured in the audit record alongside their decision

---

## Testing

```bash
# Unit + graph tests (no external dependencies)
pytest tests/ -q

# Integration tests (requires Docker Compose stack running)
INTEGRATION=1 pytest tests/test_workflow_integration.py -q
```

| Test File | Coverage |
|-----------|----------|
| `test_exception_routing.py` | All exception categories; confidence threshold boundary cases; HITL trigger logic |
| `test_workflow_integration.py` | End-to-end graph with patched Bedrock and pgvector; HITL interrupt/resume cycle |
| `test_eval_runner.py` | Eval runner against golden dataset fixture; asserts accuracy floor before any CI merge |

---

## The Forward Deployed Motion

This prototype was built to demonstrate a specific operating model, not just a technology stack:

1. **Find the leverage** — Exception routing is the highest-friction step in Lab Ops not because the decisions are hard, but because every decision requires system-hopping. High volume, consistent judgment pattern, system-of-record lookup as the bottleneck. That is the shape of a 10–100x workflow candidate.

2. **Design the future-state workflow** — Mapped all exception types and their protocols. Defined the agent/human handoff at the confidence boundary that eval data validates, not the boundary that feels safest. Designed the supervisor approval surface for lab staff vocabulary, not engineering vocabulary.

3. **Build and connect the systems** — MCP servers decouple "what tools exist" from "how the agent is wired." Adding a new upstream source (e.g., cold-chain temperature logger) is a new MCP server, not an orchestrator code change. The Golden Record pattern ensures agents never act directly against a flaky upstream system.

4. **Enable the business to run the workflows** — The HITL surface lets a lab supervisor approve, override, or escalate without touching code. Confidence thresholds are adjustable through configuration, not a pull request.

5. **Run in production and own the KPIs** — Override rate is the leading indicator — it surfaces before TAT degrades. The dashboard tracks it alongside throughput, cost per case, and adoption: the numbers that matter to Lab Ops leadership, not just model metrics.

---

## Related Work

- **[RCM Guardian](https://github.com/pvenkata-tech/the-rcm-guardian)** — The same orchestration pattern applied downstream: billing document extraction, payer policy matching, and claims adjudication support. Lab Ops Accelerator operates upstream (specimen handling); RCM Guardian operates downstream (revenue cycle). Together they cover the full specimen-to-payment lifecycle.

---

*Built as a Forward Deployed AI prototype targeting genetic testing lab operations. Stack: Python 3.12 · FastAPI · LangGraph · AWS Bedrock (Claude 3.5 Sonnet + Titan Embeddings v2) · PostgreSQL + pgvector · MCP servers · Prometheus · Grafana · Terraform/AWS Fargate.*
