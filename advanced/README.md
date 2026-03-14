# Advanced Workflows

Remote-only MinT workflows that sit beyond the basic quickstart and demo scripts.

> All operations run against a remote MinT server. This repo does not start backend services locally.

Use the MinT endpoint that matches your region:

- Mainland China: `https://mint-cn.macaron.xin/`
- Outside Mainland China: `https://mint.macaron.xin/`

## Checkpoint lifecycle (`checkpoint.py`)

Single script with four subcommands. All require `MINT_API_KEY` in the environment or `.env` file.

### save

Run 1 SFT step, save both a full state checkpoint (weights + optimizer) and a sampler-only checkpoint (weights only).

```bash
MINT_API_KEY=... python advanced/checkpoint.py save --name my-ckpt
```

Options:
- `--model` — base model (default: `$MINT_BASE_MODEL` or `Qwen/Qwen3-0.6B`)
- `--rank` — LoRA rank (default: `$MINT_LORA_RANK` or `16`)
- `--lr` — learning rate (default: `$MINT_RL_LR` or `5e-5`)

### download

Download a checkpoint archive from a `mint://` path. Retries on 409 (archive being created).

```bash
MINT_API_KEY=... python advanced/checkpoint.py download mint://run-id/weights/step-100 -o ./ckpts
```

Options:
- `-o` / `--output` — output directory (default: `./checkpoints`)
- `--checkpoint-type` — `training`, `sampler`, or `auto` (default: `auto`)
- `--no-extract` — skip tar extraction
- `--max-wait` — max wait for 409 retry in seconds (default: 600)

### upload

Upload a local `.tar.gz` checkpoint archive to the server.

```bash
MINT_API_KEY=... python advanced/checkpoint.py upload ./ckpts/step-100.tar.gz
```

Options:
- `--timeout` — upload timeout in seconds (default: 300)

### resume

Resume training from a previously saved or uploaded checkpoint. Two modes:

```bash
# Weights only (optimizer resets, auto-detects model/rank):
MINT_API_KEY=... python advanced/checkpoint.py resume ckpt_abc123

# With optimizer state (requires MINT_BASE_MODEL + MINT_LORA_RANK):
MINT_API_KEY=... python advanced/checkpoint.py resume ckpt_abc123 --with-optimizer --steps 3
```

Options:
- `--with-optimizer` — preserve optimizer momentum (requires `MINT_BASE_MODEL`, `MINT_LORA_RANK`)
- `--steps` — SFT steps to run after resume (default: 3)
- `--lr` — learning rate (default: `$MINT_RL_LR` or `5e-5`)
- `--save-name` — name for checkpoint saved after training (default: `resumed-checkpoint`)

## MIS rollout correction validation (`validate_mis_rollout_correction.py`)

Use this script to validate that a session-level Seq-MIS `rollout_correction_config` is accepted during `create_model` and later honored by `forward_backward(..., loss_fn="importance_sampling")` without resending rollout config per step.

```bash
MINT_API_KEY=... python advanced/validate_mis_rollout_correction.py --base-model Qwen/Qwen3-0.6B
```

Supported CLI flags:
- `--base-url`
- `--api-key`
- `--base-model`
- `--lora-rank`
- `--mis-threshold`
- `--create-timeout-s`
- `--forward-backward-timeout-s`
- `--poll-interval-s`
- `--skip-cleanup`

The script prefers `MINT_*` environment variables and falls back to Tinker-compatible aliases such as `TINKER_BASE_URL`, `TINKER_API_KEY`, and `TINKER_MODEL`.

See [`docs/mis_rollout_correction.md`](../docs/mis_rollout_correction.md) for usage notes, expected output, and failure modes.
