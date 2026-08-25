# Model serving boundary

## Current service

The `inference` container is a separately deployed, internally authenticated
model service. It exposes:

- `GET /health`: model readiness and whether any live GPU model is loaded;
- `GET /v1/models`: model ID, version, backend, input contract, measurements,
  readiness, and artifact SHA-256;
- `POST /v1/infer`: versioned prediction contract.

The checked-in adapter replays persisted predictions. It explicitly returns
`backend=persisted_prediction_replay` and `live_image_inference=false`.

## Worker behavior

Workers call the service with a private internal token, a three-second timeout,
two request retries with exponential backoff, and a per-process circuit breaker.
Remote model metadata and agreement with the downstream workflow are written to
the terminal result.

An outage integration test produced exactly three job attempts and:

```text
submitted → claimed → retry_scheduled
          → claimed → retry_scheduled
          → claimed → failed
```

The workflow did not silently fall back or invent a prediction.

## Live adapter contract

A future TorchScript or ONNX adapter must implement `InferenceAdapter`:

- expose immutable `ModelInfo`, including a weight hash and live capability;
- validate and decode the S3 artifact input;
- return HVA/IMA measurements through `InferenceOutput`;
- fail readiness if weights, device, or preprocessing assets are unavailable.

No live adapter is claimed until redistributable weights and an end-to-end image
test are available.
## Live HVA/IMA adapter

`hvangle-resnet50` performs actual image inference from JPEG/PNG bytes. The
inference container retrieves content-addressed objects from MinIO, decodes and
normalizes them to ImageNet statistics at 256×256, then runs the hash-locked
two-output ResNet50 checkpoint. It defaults to CPU because the current host has
no working NVIDIA driver; the backend is reported in every result.

The 120-image unilateral test split was rerun through the production adapter:
HVA MAE 3.1199° (79.2% within 5°) and IMA MAE 1.5034° (98.3% within 5°).
These are internal retrospective results, not a clinical-performance claim.

All uploaded-image predictions retain a mandatory human-review gate. Invalid
or unsupported image bytes fail explicitly; the service never substitutes a
fabricated prediction.
