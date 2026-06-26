# ML Infrastructure

CortexGuard uses AWS SageMaker for step classification inference, model retraining, and automated model lifecycle management. All resources are provisioned via Terraform in `terraform/endpoint.tf` and `terraform/retraining.tf`.

---

## Architecture

```
Edge StepExecutor
    │  snapshot + outcome via TelemetryClient
    ▼
cloud-api  ──► telemetry table (RDS)
                         │
SageMaker Pipeline (weekly, Monday 06:00 UTC)
    │  ProcessingJob: sagemaker-scikit-learn container
    │  reads telemetry from RDS via CLOUD_DB_URL
    │  calls training_script.py stub
    │  registers artifact in Model Registry
    ▼
SageMaker Model Registry (cortexguard-edge-models)
    │  EventBridge on PendingManualApproval
    ▼
deploy-candidate Lambda
    │  deploys new model as challenger variant at 10%
    ▼
SageMaker Endpoint (ml.t2.medium, champion 90% / challenger 10%)
    │  ┌─────────────────────────────────────┐
    │  │ promote-challenger Lambda           │
    │  │ (every 15 min): checks challenger   │
    │  │ metrics → promotes to champion      │
    │  │ if healthy after 30 min window      │
    │  └─────────────────────────────────────┘
    ▲
    │  POST /api/v1/classify (proxied from edge StepClassifierClient)
cloud-api

Model Monitor (hourly data drift check)
    │  drift detected? → SNS → rollback Lambda
    ▼
Reverts endpoint to previous champion
```

---

## Step Classifier Endpoint

The SageMaker endpoint runs a `sagemaker-scikit-learn` container with two production variants:

| Variant | Weight | Purpose |
|---------|--------|---------|
| champion | 90% | Current approved model |
| challenger | 10% | Canary for new model versions |

The endpoint is deployed with data capture enabled (100% sampling) for Model Monitor analysis.

### Model artifacts

| S3 path | Contents |
|---------|----------|
| `s3://{bucket}/artifacts/model.tar.gz` | `training_result.json` with model metadata |
| `s3://{bucket}/artifacts/sourcedir.tar.gz` | `inference.py` + `setup.py` for SageMaker Inference Toolkit |

The `SAGEMAKER_SUBMIT_DIRECTORY` env var points the container at `sourcedir.tar.gz`, which the Inference Toolkit downloads, extracts, pip-installs, and imports as the `inference` module.

### Bootstrap artifact

Before the first pipeline run, a `null_resource` bootstrap creates a minimal `model.tar.gz` and `sourcedir.tar.gz` so the endpoint can deploy. The bootstrap inference script returns a canned `predicted_outcome: "completed"` with 0.95 confidence.

---

## Retraining Pipeline

The pipeline (`cortexguard-retraining`) runs weekly via EventBridge Scheduler (Monday 06:00 UTC).

### Pipeline steps

| Step | What it does |
|------|-------------|
| `ProcessingJob` | Runs `run.py` in a `sagemaker-scikit-learn` container. Reads telemetry from RDS via `CLOUD_DB_URL`, calls `training_script.py` stub, writes artifact to S3 |
| `RegisterModel` | Creates a new model package version in the Model Registry with `PendingManualApproval` status |

### Environment

The processing job receives:
- `CLOUD_DB_URL` (from Secrets Manager)
- `FetchDays` pipeline parameter (default: 7)

### Training script stub

Currently a stub at `src/cortexguard/cloud/retraining/training_script.py`:

```python
def train(telemetry_rows: list[TelemetryRow]) -> dict:
    return {
        "model_id": "step_classifier_v1",
        "version": 0,
        "params": {"baseline_fpr": 0.0, "baseline_fnr": 0.0},
        "metrics": {"num_records": len(telemetry_rows)},
    }
```

Real training logic is deferred. The plumbing (pipeline, registry, endpoint, monitoring) is in place so a real training script slots in without re-architecting.

---

## Champion/Challenger Promotion

### Phase 1: Deploy as challenger (candidate)

When the retraining pipeline registers a new model package, EventBridge triggers the `deploy-candidate` Lambda:

1. Saves the current champion to SSM Parameter Store (rollback target)
2. Approves the new model package (sets status to `Approved`)
3. Creates a SageMaker Model for the candidate
4. Creates a new endpoint config: **champion** (current model, 90%) + **challenger** (candidate, 10%)
5. Writes candidate state to SSM (`{version, model_name, deployed_at}`)

### Phase 2: Promote challenger → champion

A separate `promote-challenger` Lambda runs on a schedule (every 15 minutes) and:

1. Reads SSM for a pending candidate
2. Checks the monitoring window has elapsed (default: 30 minutes)
3. Checks Challenger variant CloudWatch metrics:
   - 4XX error sum < threshold (default: 5)
   - 5XX errors = 0
   - p99 latency < threshold (default: 2000ms)
4. If healthy: creates champion model from the candidate's artifact, updates endpoint config (champion 90%, challenger 10%), clears SSM candidate state
5. If unhealthy: logs the failure, leaves the challenger at 10% — no promotion. The next retraining cycle deploys a new candidate.

The previous champion is tracked in SSM Parameter Store for rollback. If the challenger is never promoted (e.g. perpetual health failures), the champion remains unchanged and continues serving 90% of traffic.

---

## Rollback

The `rollback` Lambda is triggered by SNS when any of these CloudWatch alarms fire:

| Alarm | Metric | Threshold |
|-------|--------|-----------|
| `data_drift_high` | `DriftViolationCount` | Configurable (default: 5 violations in 1 hour) |
| `endpoint_4xx_rate` | `ModelInvocation4XXErrors` | 10 errors in 5 minutes |
| `endpoint_latency` | `ModelLatency` p99 | 2000 ms |

The rollback Lambda reads the previous champion version from SSM Parameter Store and reverts the endpoint.

---

## Data Drift Monitoring

Model Monitor runs hourly on the endpoint's captured data. It compares inference requests against a baseline computed by the retraining pipeline and emits `DriftViolationCount` metrics. Drift alerts are sent to an SNS topic that:
- Triggers the rollback Lambda
- Optionally sends email (if `drift_alarm_email` is set in `terraform.tfvars`)

---

## Configuration

### Terraform variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `sagemaker_endpoint_instance_type` | `ml.t2.medium` | Instance type for the step classifier endpoint |
| `data_drift_violation_threshold` | `5` | Drift violations before alarm fires |
| `drift_alarm_email` | `""` | Email for drift/health alerts (empty = no email) |

### Environment variables (cloud-api)

| Variable | Purpose |
|----------|---------|
| `SAGEMAKER_ENDPOINT_NAME` | SageMaker endpoint name, injected by Terraform into ECS task definition |
