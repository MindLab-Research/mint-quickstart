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
- `MINT_INTERRUPT_COUNT` — simulated interruptions (default: 5)
- `MINT_RESUME_PATH` — checkpoint path to resume from

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
