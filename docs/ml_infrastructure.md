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
promote-champion Lambda
    │  updates endpoint to new model version
    ▼
SageMaker Endpoint (ml.t2.medium, champion 90% / challenger 10%)
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

When the retraining pipeline registers a new model package, EventBridge triggers the `promote-champion` Lambda:

1. Approves the new model package (sets status to `Approved`)
2. Builds a new SageMaker Model from the approved artifact
3. Creates a new endpoint configuration
4. Updates the endpoint to point at the new model

The previous champion is tracked in SSM Parameter Store for rollback.

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
