# Symptom Index

Use this file as a fast router from symptom to likely cause.

## `Missing MINT_API_KEY`

Likely causes:
- `.env` missing
- env var not exported
- running from a different shell or directory than expected

Check:
- `../../../docs/troubleshooting.md`
- `../../../README.md`

## `401 Unauthorized` or `403 Forbidden`

Likely causes:
- bad API key
- expired or revoked API key
- wrong key variable

Check:
- `../../../docs/troubleshooting.md`

## `ConnectionError`, `Connection refused`, or obvious timeout before work starts

Likely causes:
- bad `MINT_BASE_URL`
- wrong region endpoint
- real network outage
- server unreachable

Check:
- `../../../docs/troubleshooting.md`
- `../../../advanced/queue_status.py` if the user suspects queue or wait behavior

## Script is slow

Likely causes:
- remote queue wait, not local code
- many serial sample requests in a throughput-oriented workflow
- very large batch or generation length
- user expects throughput from a teaching baseline

Check:
- `concurrency_patterns.md`
- `../../../advanced/queue_status.py`
- `../../../demos/rl/rl_core.py`

## `accuracy=0.0%`, rewards all zero, or `datums=0`

Likely causes:
- task too hard for the base model
- `max_tokens` too low
- reward extraction logic too strict
- advantages all zero after reward normalization
- RL datum alignment bug

Check:
- `../../../docs/troubleshooting.md`
- `../../../demos/rl/README.md`
- `../../../demos/rl/rl_core.py`

## Loss is NaN or training diverges

Likely causes:
- learning rate too high
- unstable reward scale or advantages
- bad custom loss implementation

Check:
- `../../../docs/troubleshooting.md`
- `../../../quickstart/custom_loss.py`
- `../../../demos/rl/README.md`

## Checkpoint path fails

Likely causes:
- confusing `weights` with `sampler_weights`
- expecting optimizer resume from a weights-only load
- legacy path shape not normalized
- metadata lookup fallback not handled

Check:
- `../../../advanced/checkpoint.py`
- `../../../advanced/README.md`
- `../../../tests/test_checkpoint_paths.py`

## Sample or train call seems to do nothing

Likely causes:
- missing `.result()`
- response future stored but never waited on
- async code copied into a sync script without the event loop pieces

Check:
- `debug_checklist.md`
- `../../../quickstart/README.md`
- `../../../mint-skill/mint_api_reference.txt`

## Looks like a backend bug

Slow down first. It may still be one of these:
- env or auth problem
- wrong client code shape
- unsupported local expectation copied from another repo

Before calling it backend-owned, confirm:
- minimal repro exists
- repo code shape is correct
- the failure is not already explained by `../../../docs/troubleshooting.md`
- the issue still reproduces after client-side fixes
