# RL Demos

Three orthogonal RL demos showing different reward sources on MinT.

All three share `rl_core.py` — the generic GRPO loop. Each adapter only defines task-specific logic: dataset, prompt formatting, and reward function.

> **Note:** All training runs against a remote MinT server. These scripts do NOT start any backend services locally. You only need a valid server endpoint + API key.

## Architecture

```
demos/rl/
  rl_core.py              # Shared GRPO loop (sample → reward → advantage → train)
  adapters/
    verifiable_math.py     # RL-1: deterministic verifier reward
    preference_chat.py     # RL-2: helpfulness preference proxy
    environment_tooluse.py # RL-3: code execution environment reward
```

Each adapter implements:
- `build_dataset()` — task items for one step
- `make_prompt(sample, tokenizer)` — convert sample to token ids
- `compute_reward(response, sample)` — score a generation
- `evaluate(step, rewards, num_datums)` — log metrics

## Overview

| Demo | Script | Reward Source | Represents |
|------|--------|---------------|------------|
| **RL-1 Verifiable Math** | `adapters/verifiable_math.py` | Deterministic programmatic verifier | Objective RL |
| **RL-2 Preference Chat** | `adapters/preference_chat.py` | Pairwise/judge preference proxy | Alignment RL |
| **RL-3 Environment Tool Use** | `adapters/environment_tooluse.py` | Delayed reward from code execution | Agentic RL |

## Prerequisites

```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git
```

Create `.env` in the repo root:
```
MINT_API_KEY=sk-mint-your-api-key-here
```

---

## RL-1: Verifiable Math

**What:** Train a model to solve addition problems using RL with exact-match grading.

**Reward:** `1.0` if extracted integer matches correct answer, `0.0` otherwise.

**Run:**
```bash
python demos/rl/adapters/verifiable_math.py
```

**Expected output:**
```
Model: Qwen/Qwen3-0.6B, Vocab: 151,936
Step 1: accuracy=xx.x%, datums=XX
...
Step 5: accuracy=xx.x%, datums=XX
Saved: ckpt_rl-final_...
```

**Common failures:**
- `accuracy=0.0%` on every step → increase `MINT_RL_MAX_TOKENS` (default 8 may be too short)
- `ConnectionError` → verify `MINT_API_KEY` is set and server is reachable

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MINT_RL_STEPS` | `5` | Training steps |
| `MINT_RL_BATCH` | `10` | Problems per step |
| `MINT_RL_GROUP` | `4` | Samples per problem (GRPO group) |
| `MINT_RL_LR` | `1e-4` | Learning rate |
| `MINT_RL_MAX_TOKENS` | `8` | Max generation length |

---

## RL-2: Preference Chat

**What:** Train a model to produce helpful chat responses using a proxy reward function.

**Reward:** Composite score (0.0–1.0) based on response length, sentence structure, and vocabulary diversity — a stand-in for human preference.

**Run:**
```bash
python demos/rl/adapters/preference_chat.py
```

**Expected output:**
```
Model: Qwen/Qwen3-0.6B, Vocab: 151,936
Step 1: avg_reward=0.xxx, datums=XX
...
Step 10: avg_reward=0.xxx, datums=XX
Saved: ckpt_rl-final_...
```

**Common failures:**
- All rewards identical → model always generates same length responses; try higher temperature
- Training diverges → lower `MINT_RL_LR` to `1e-5`

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MINT_RL_STEPS` | `10` | Training steps |
| `MINT_RL_BATCH` | `8` | Prompts per step |
| `MINT_RL_GROUP` | `4` | Samples per prompt |
| `MINT_RL_LR` | `1e-4` | Learning rate |
| `MINT_RL_MAX_TOKENS` | `128` | Max generation length |

---

## RL-3: Environment Tool Use

**What:** Train a model to write Python functions that pass test cases.

**Reward:** `1.0` if generated code passes all test assertions, `0.0` otherwise. Code is extracted from the response and executed in a sandboxed namespace.

**Run:**
```bash
python demos/rl/adapters/environment_tooluse.py
```

**Expected output:**
```
Model: Qwen/Qwen3-0.6B, Vocab: 151,936
Step 1: accuracy=xx.x%, datums=XX
...
Step 10: accuracy=xx.x%, datums=XX
Saved: ckpt_rl-final_...
```

**Common failures:**
- `accuracy=0.0%` throughout → 0.6B model may be too small for code gen; try `MINT_BASE_MODEL=Qwen/Qwen3-1.7B`
- Truncated code → increase `MINT_RL_MAX_TOKENS` beyond 256

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MINT_RL_STEPS` | `10` | Training steps |
| `MINT_RL_BATCH` | `8` | Problems per step |
| `MINT_RL_GROUP` | `4` | Samples per problem |
| `MINT_RL_LR` | `1e-4` | Learning rate |
| `MINT_RL_MAX_TOKENS` | `256` | Max generation length |

---

## Common Configuration

These environment variables apply to all RL demos:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINT_API_KEY` | *(required)* | MinT API key |
| `MINT_BASE_MODEL` | `Qwen/Qwen3-0.6B` | Base model |
| `MINT_LORA_RANK` | `16` | LoRA rank |
| `MINT_RL_TEMPERATURE` | `1.0` | Sampling temperature |
