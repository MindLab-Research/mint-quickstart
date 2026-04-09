# Embodied Demos

This track now has one primary SDK demo and one low-level HTTP reference for MinT.

> **Status:** Embodied-1 is available now as a `mintx` / `mint.mint` SDK example. The HTTP variant stays as a low-level protocol reference.

## Available Demo

### Embodied-1: OpenPI FAST SDK

- **Script:** `openpi_vla_sdk.py`
- **Scope:** minimal end-to-end VLA training round-trip through the MinT-only `mintx` namespace
- **Demonstrates:** `mint.ServiceClient()` + `mint.mint.create_openpi_training_client(...)` + `train_step(...)` + `save_weights_for_sampler(...)`
- **Why this is the main path:** it keeps top-level `mint` Tinker-compatible while moving OpenPI / VLA-specific APIs into `mint.mint` / `mintx`
- **Current scope:** the helper is pinned to `openpi/pi0-fast-libero-low-mem-finetune` with LoRA rank `16`

## Reference Demo

### OpenPI FAST HTTP

- **Script:** `openpi_vla_http.py`
- **Scope:** raw HTTP version of the same OpenPI VLA flow
- **Demonstrates:** `create_session` -> `create_model` -> `train_step` -> `save_weights_for_sampler` -> `delete model`
- **Why keep it:** this example shows the wire protocol directly for debugging, backend comparison, and request-shape inspection

## Core Shape

```text
model_input.chunks
  1. image: base_0_rgb
  2. image: left_wrist_0_rgb
  3. image: right_wrist_0_rgb
  4. encoded_text: prefix_tokens

loss_fn_inputs
  - state
  - target_tokens
  - weights
  - token_ar_mask
  - optional: logprobs + advantages
```

The example keeps itself self-contained by sending three 1x1 PNG placeholders. For a real robot or simulator rollout, replace those bytes plus the `state`, `target_tokens`, `weights`, and `token_ar_mask` tensors with your real trajectory data.

## Prerequisites

```bash
pip install httpx python-dotenv
```

Set credentials via `.env` or shell env vars:

```bash
MINT_API_KEY=sk-...
```

Use the MinT endpoint that matches your region:
- Mainland China: `https://mint-cn.macaron.xin/`
- Outside Mainland China: `https://mint.macaron.xin/`

## Run

```bash
python demos/embodied/openpi_vla_sdk.py
python demos/embodied/openpi_vla_http.py
```

## Useful Env Vars

- `MINT_API_KEY` / `TINKER_API_KEY`: auth
- `MINT_BASE_URL` / `TINKER_BASE_URL`: server endpoint
- `MINT_OPENPI_HTTP_BASE_MODEL`: default `openpi/pi0-fast-libero-low-mem-finetune`
- `MINT_OPENPI_HTTP_LORA_RANK`: default `16`
- `MINT_OPENPI_HTTP_LR`: default `0.003`
- `MINT_OPENPI_HTTP_SAMPLER_PATH`: default `mint-openpi-vla-http-example`
- `MINT_OPENPI_HTTP_CLIENT_TIMEOUT_SECONDS`: default `120`
- `MINT_OPENPI_HTTP_FUTURE_TIMEOUT_SECONDS`: default `1200`
- `MINT_OPENPI_SDK_BASE_MODEL`: default `openpi/pi0-fast-libero-low-mem-finetune`
- `MINT_OPENPI_SDK_LORA_RANK`: default `16`
- `MINT_OPENPI_SDK_LR`: default `0.003`
- `MINT_OPENPI_SDK_SAMPLER_NAME`: default `mint-openpi-sdk-example`
- `MINT_OPENPI_SDK_CREATE_TIMEOUT_SECONDS`: default `1200`
- `MINT_OPENPI_SDK_STEP_TIMEOUT_SECONDS`: default `1200`
- `MINT_OPENPI_SDK_SAVE_TIMEOUT_SECONDS`: default `1200`
- `MINT_OPENPI_SDK_TTL_SECONDS`: default `3600`

## Expected Output Shape

`openpi_vla_sdk.py` prints a smaller dict focused on the SDK-facing result:

```text
model_id
model_name
lora_rank
train_step_metrics
sampler_path
```

`openpi_vla_http.py` prints a Python dict with these keys:

```text
session_id
model_id
model_info
train_step
sampler
delete_model
models
```

## What This Unlocks Next

The SDK demo is now the main user-facing path, and the HTTP demo remains the ground-truth wire reference. A later simulator or rollout demo can build on both shapes without re-inventing either the namespace boundary or the request contract.
