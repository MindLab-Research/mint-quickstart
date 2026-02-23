# Advanced Demos

Advanced usage patterns for MinT: checkpoint upload, training resumption, and weight management.

> **Note:** All operations run against a remote MinT server. This repo does not start MinT backend services locally.

## Scripts

| Script | Description |
|--------|-------------|
| `resume.py` | RL training with save/resume across interruptions |
| `upload_weights.py` | Upload a local checkpoint archive to the MinT server |
| `resume_from_upload.py` | Resume training from a previously uploaded checkpoint |

## Resume Training (`resume.py`)

Demonstrates fault-tolerant RL training that survives interruptions:

```bash
MINT_API_KEY=... python advanced/resume.py
```

Key env vars:
- `MINT_TOTAL_STEPS` — total training steps (default: 100)
- `MINT_CHECKPOINT_EVERY_STEPS` — save checkpoint every N steps (default: 20)
- `MINT_RESUME_PATH` — checkpoint path to resume from
- `MINT_BASE_MODEL` — model to train (default: `Qwen/Qwen3-0.6B`)
- `MINT_GROUP_SIZE` — sampled candidates per prompt in each RL step (default: 4)
- `MINT_MAX_TOKENS` — max generation tokens per sample (default: 256)

If `MINT_RESUME_PATH` is invalid, `resume.py` fails fast instead of silently restarting from step 0.
Runtime logs use explicit tags (`[run]`, `[resume]`, `[train]`, `[checkpoint]`, `[interrupt]`, `[done]`) to make checkpoint/resume state easy to audit.

## Upload Weights (`upload_weights.py`)

Upload a `.tar.gz` checkpoint archive to the server:

```bash
MINT_API_KEY=... MINT_UPLOAD_ARCHIVE=/path/to/ckpt.tar.gz python advanced/upload_weights.py
```

## Resume from Upload (`resume_from_upload.py`)

Load an uploaded checkpoint and continue training:

```bash
MINT_API_KEY=... MINT_RESUME_PATH=ckpt_... python advanced/resume_from_upload.py
```

Set `MINT_RESUME_WITH_OPTIMIZER=1` if your archive includes optimizer state.
