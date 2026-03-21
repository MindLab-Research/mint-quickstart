# MIS Rollout Correction Validation

Validate session-level Seq-MIS rollout correction on a remote MinT server with one script:

```bash
python advanced/validate_mis_rollout_correction.py --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
```

> This is a targeted validation flow, not a full RL tutorial. It checks that session-level `rollout_correction_config` survives `create_model` and is honored later by `forward_backward(..., loss_fn="importance_sampling")` without resending per-step rollout config.

## What this script checks

- `create_model` accepts a session-level Seq-MIS `rollout_correction_config`
- `forward_backward` succeeds with `loss_fn="importance_sampling"`
- the training request omits per-step rollout correction config
- the response contains a valid non-empty `loss_fn_outputs`
- the temporary model is deleted by default after the check

## What it does not check

- your full RL recipe or reward design
- throughput or production-scale stability
- every model/account combination on the server
- a higher-level MinT SDK helper for MIS wiring

## Prerequisites

- Python >= 3.11
- `requests` installed
- a working MinT API key in `MINT_API_KEY` or `TINKER_API_KEY`
- optional `.env` file in the repo root
- a supported MoE / Megatron model; the current server rejects dense models for this validation

Use the MinT endpoint that matches your region:

- Mainland China: `https://mint-cn.macaron.xin/`
- Outside Mainland China: `https://mint.macaron.xin/`

## Supported environment variables

MinT-style names win over Tinker-compatible aliases when both are set.

- `MINT_BASE_URL` / `TINKER_BASE_URL`
- `MINT_API_KEY` / `TINKER_API_KEY`
- `MINT_BASE_MODEL` / `TINKER_MODEL`
- `MINT_LORA_RANK` / `TINKER_LORA_RANK`
- `MINT_MIS_THRESHOLD` / `TINKER_MIS_THRESHOLD`
- `MINT_CREATE_MODEL_TIMEOUT_S` / `TINKER_CREATE_MODEL_TIMEOUT_S`
- `MINT_FORWARD_BACKWARD_TIMEOUT_S` / `TINKER_FORWARD_BACKWARD_TIMEOUT_S`
- `MINT_POLL_INTERVAL_S` / `TINKER_POLL_INTERVAL_S`

## Example commands

Use MinT-style variables:

```bash
export MINT_API_KEY=sk-your-api-key-here
export MINT_BASE_URL=<your-region-endpoint>
python advanced/validate_mis_rollout_correction.py --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
```

Use Tinker-compatible aliases:

```bash
export TINKER_API_KEY=sk-your-api-key-here
export TINKER_BASE_URL=<your-region-endpoint>
export TINKER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
python advanced/validate_mis_rollout_correction.py
```

Skip cleanup if you need the created model for manual inspection:

```bash
python advanced/validate_mis_rollout_correction.py --skip-cleanup
```

## Expected output

```text
[config] base_url=<your-region-endpoint> base_model=Qwen/Qwen3-30B-A3B-Instruct-2507 lora_rank=8 mis_threshold=1.1
[create_model] submitted session_id=validate-mis-1234abcd
[create_model] resolved model_id=model_...
[forward_backward] submitted model_id=model_... loss_fn=importance_sampling
[forward_backward] resolved outputs=1
PASS: MIS rollout_correction request succeeded and response was valid
[cleanup] deleted model_id=model_...
```

## Common failure cases

- `FAIL [config]` — API key is missing; set `MINT_API_KEY` or `TINKER_API_KEY`
- `FAIL [create_model]` — model creation failed, timed out, or the account/model is unavailable
- `FAIL [forward_backward]` — the training request failed or timed out after model creation
- `FAIL [malformed_response]` — the server returned a response without `loss_fn_outputs`
- `[cleanup] warning` — validation finished, but best-effort deletion failed

## Positioning

- recommended for advanced RL or Tinker-migration users
- remote-only against an already deployed MinT server
- honest about server-side blockers: if the backend rejects the flow, document the exact stage and error text

See `advanced/README.md` for the full advanced workflow index.
