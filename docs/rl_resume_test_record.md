# RL Resume Test Record

- Date: 2026-02-23
- Script: `advanced/resume.py`
- Environment: remote MinT server (`https://mint.macaron.im/` from local `.env`)
- Validation policy: quickstart-first, explicit run -> interrupt -> resume checks

## Matrix Summary

| Model | 1) First training | 2) Manual interrupt | 3) Resume with `MINT_RESUME_PATH` | Result |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` | Success | Success | Success | Pass |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Success | Success | Success | Pass |

## Detailed Runs

### Model: `Qwen/Qwen3-0.6B`

1. First training run

Command:

```bash
MINT_BASE_MODEL='Qwen/Qwen3-0.6B' \
MINT_TOTAL_STEPS=2 \
MINT_CHECKPOINT_EVERY_STEPS=1 \
MINT_GROUP_SIZE=1 \
MINT_MAX_TOKENS=16 \
python advanced/resume.py
```

Evidence (`.tmp/rl_resume_logs/qwen06b_first_20260223.log`):

- `[run] ... start_step=0`
- `[train] step=1/2 ...`
- `[checkpoint] step=1 reason=periodic path=mint://.../rl-resume-periodic-step-000001`
- `[train] step=2/2 ...`
- `[checkpoint] step=2 reason=periodic path=mint://.../rl-resume-periodic-step-000002`
- `[checkpoint] step=2 reason=final path=mint://.../rl-resume-final-step-000002`
- `[done] final_step=2 ...`

2. Interrupt run (SIGINT)

Command (timeboxed):

```bash
timeout -s INT 90s env \
MINT_BASE_MODEL='Qwen/Qwen3-0.6B' \
MINT_TOTAL_STEPS=100 \
MINT_CHECKPOINT_EVERY_STEPS=1 \
MINT_GROUP_SIZE=1 \
MINT_MAX_TOKENS=16 \
python advanced/resume.py
```

Evidence (`.tmp/rl_resume_logs/qwen06b_interrupt_20260223_long.log`):

- `[train] step=4/100 ...`
- `[checkpoint] step=4 reason=periodic path=mint://.../rl-resume-periodic-step-000004`
- `[interrupt] received keyboard interrupt; saving recovery checkpoint.`
- `[checkpoint] step=4 reason=interrupt path=mint://.../rl-resume-interrupt-step-000004`

3. Resume run with explicit checkpoint

Command:

```bash
MINT_BASE_MODEL='Qwen/Qwen3-0.6B' \
MINT_TOTAL_STEPS=6 \
MINT_CHECKPOINT_EVERY_STEPS=1 \
MINT_GROUP_SIZE=1 \
MINT_MAX_TOKENS=16 \
MINT_RESUME_PATH='mint://2632b2c6-5f57-4969-b646-22ebce6df95a_0/rl-resume-interrupt-step-000004' \
python advanced/resume.py
```

Evidence (`.tmp/rl_resume_logs/qwen06b_resume_20260223.log`):

- `[resume] loaded successfully; inferred_start_step=4`
- `[run] ... start_step=4`
- `[train] step=5/6 ...`
- `[train] step=6/6 ...`
- `[done] final_step=6 ...`

`global_step` monotonic check: resumed from inferred step `4`, then advanced to `5 -> 6`, no reset to `0`.

### Model: `Qwen/Qwen3-235B-A22B-Instruct-2507`

1. First training run

Command:

```bash
timeout 420s env \
PYTHONUNBUFFERED=1 \
MINT_BASE_MODEL='Qwen/Qwen3-235B-A22B-Instruct-2507' \
MINT_TOTAL_STEPS=2 \
MINT_CHECKPOINT_EVERY_STEPS=1 \
MINT_GROUP_SIZE=1 \
MINT_MAX_TOKENS=16 \
python advanced/resume.py
```

Result: success.

Evidence (`.tmp/rl_resume_logs/qwen235b_first_20260223_032005_retry.log`):

- `[run] ... start_step=0`
- `[train] step=1/2 ...`
- `[checkpoint] step=1 reason=periodic path=mint://.../rl-resume-periodic-step-000001`
- `[train] step=2/2 ...`
- `[checkpoint] step=2 reason=periodic path=mint://.../rl-resume-periodic-step-000002`
- `[checkpoint] step=2 reason=final path=mint://.../rl-resume-final-step-000002`
- `[done] final_step=2 ...`

2. Interrupt run (SIGINT)

Command (timeboxed):

```bash
timeout -s INT 120s env \
PYTHONUNBUFFERED=1 \
MINT_BASE_MODEL='Qwen/Qwen3-235B-A22B-Instruct-2507' \
MINT_TOTAL_STEPS=100 \
MINT_CHECKPOINT_EVERY_STEPS=1 \
MINT_GROUP_SIZE=1 \
MINT_MAX_TOKENS=16 \
python advanced/resume.py
```

Evidence (`.tmp/rl_resume_logs/qwen235b_interrupt_20260223_032005.log`):

- `[train] step=1/100 ...`
- `[interrupt] received keyboard interrupt; saving recovery checkpoint.`
- `[checkpoint] step=1 reason=interrupt path=mint://.../rl-resume-interrupt-step-000001`
- `[interrupt] resume_source=none latest_checkpoint=mint://.../rl-resume-interrupt-step-000001`

3. Resume run with explicit checkpoint

Command:

```bash
PYTHONUNBUFFERED=1 \
MINT_BASE_MODEL='Qwen/Qwen3-235B-A22B-Instruct-2507' \
MINT_TOTAL_STEPS=6 \
MINT_CHECKPOINT_EVERY_STEPS=1 \
MINT_GROUP_SIZE=1 \
MINT_MAX_TOKENS=16 \
MINT_RESUME_PATH='mint://586f7680-3577-42a1-acaf-132feea2b624_0/rl-resume-interrupt-step-000001' \
python advanced/resume.py
```

Evidence (`.tmp/rl_resume_logs/qwen235b_resume_20260223_032005.log`):

- `[resume] loaded successfully; inferred_start_step=1`
- `[run] ... start_step=1`
- `[train] step=2/6 ...`
- `[train] step=3/6 ...`
- `[train] step=4/6 ...`
- `[train] step=5/6 ...`
- `[train] step=6/6 ...`
- `[done] final_step=6 ...`

`global_step` monotonic check: resumed from inferred step `1`, then advanced to `2 -> 3 -> 4 -> 5 -> 6`, no reset to `0`.

## Additional fail-fast check (`MINT_RESUME_PATH`)

Command:

```bash
MINT_BASE_MODEL='Qwen/Qwen3-0.6B' \
MINT_TOTAL_STEPS=2 \
MINT_CHECKPOINT_EVERY_STEPS=1 \
MINT_GROUP_SIZE=1 \
MINT_MAX_TOKENS=8 \
MINT_RESUME_PATH='mint://invalid/resume-path' \
python advanced/resume.py
```

Result (`.tmp/rl_resume_logs/qwen06b_failfast_20260223.log`):

- immediate load attempt: `[resume] source=mint://invalid/resume-path`
- startup fails fast with `RuntimeError: Failed to load MINT_RESUME_PATH=...`
- no `[train]` step logs emitted
