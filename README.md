<p align="center">
  <img src="docs/assets/mint-icon.jpg" alt="MinT" width="120" height="120">
</p>

# MinT Quickstart

[English](./README.md) | [中文](./README_zh.md)

The single entry repo for learning [MinT](https://github.com/MindLab-Research/mindlab-toolkit) (Mind Lab Toolkit) — from first API call to advanced RL training.

Visit the [MinT website](https://macaron.im/mindlab/mint).

> **Important:** All experiments run against an already deployed MinT server. This repo does **not** start MinT backend services locally. You only need valid server endpoint + API key credentials.

## Demo Portfolio

### Available Now

| # | Demo | Track | Reward Source / Shape | Script |
|---|------|-------|------------------------|--------|
| 1 | **RL-1 Verifiable Math** | RL | Deterministic verifier | [`demos/rl/adapters/verifiable_math.py`](demos/rl/adapters/verifiable_math.py) |
| 2 | **RL-2 Preference Chat** | RL | Pairwise/judge preference | [`demos/rl/adapters/preference_chat.py`](demos/rl/adapters/preference_chat.py) |
| 3 | **RL-3 Environment Tool Use** | RL | Code execution feedback | [`demos/rl/adapters/environment_tooluse.py`](demos/rl/adapters/environment_tooluse.py) |
| 4 | **Sampling Log** | Sampling | Train then inspect model responses | [`quickstart/sampling_log.py`](quickstart/sampling_log.py) |
| 5 | **Embodied-1 OpenPI FAST SDK** | Embodied | MinT-only `mintx` OpenPI client over 3 camera images + state + action-token supervision | [`demos/embodied/openpi_vla_sdk.py`](demos/embodied/openpi_vla_sdk.py) |
| 6 | **OpenAI-Compatible Inference** | Inference | Call a deployed model with the official OpenAI SDK (no `mint` import) | [`quickstart/openai_compat.py`](quickstart/openai_compat.py) |

### Reference

| Demo | Track | Why it exists | Script |
|------|-------|---------------|--------|
| **OpenPI FAST HTTP** | Embodied | Shows the raw wire protocol directly for debugging and request-shape reference | [`demos/embodied/openpi_vla_http.py`](demos/embodied/openpi_vla_http.py) |

### Coming Soon

| # | Demo | Track | Description | Status |
|---|------|-------|-------------|--------|
| 6 | **VLM-1 Vision QA** | VLM | Image + question -> grounded answer | Planned (M2) |
| 7 | **VLM-2 Vision Instruction** | VLM | Image + task -> action/decision | Planned (M2) |

## Quick Start

**Requirements:** Python >= 3.11, a MinT API key

```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git python-dotenv matplotlib numpy
```

Create `.env` in the repo root:
```
MINT_API_KEY=sk-your-api-key-here
```

Use the MinT endpoint that matches your region:
- Mainland China: `https://mint-cn.macaron.xin/`
- Outside Mainland China: `https://mint.macaron.xin/`

## OpenAI-Compatible Inference

If you only want to run inference against an already-deployed model, you do not
need the `mint` SDK at all. MinT serves an OpenAI-compatible endpoint at
`/oai/api/v1`, so the official `openai` SDK works by overriding `base_url`:

```bash
pip install openai

MINT_BASE_URL=https://mint.macaron.xin \
MINT_API_KEY=sk-your-api-key-here \
python quickstart/openai_compat.py chat \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --user-message "Reply with exactly: pong"
```

Subcommands: `completions`, `chat`, `tool`, `smoke`. Supported surface includes
`models.list/retrieve`, completions, chat, tool-call roundtrips, and
`AsyncOpenAI`. Not yet supported: `stream=True`, `n > 1`, `responses.create`,
`embeddings.create`. See the
[OpenAI-Compatible API docs](https://macaron.im/mindlab/mint) for the full
guide.

## Common First Questions

### Should I use SFT or RL?

- Use **SFT** when you already know what the model should say or do and you have labeled target outputs.
- Use **RL** when you do not have one fixed target answer but you can score the model's behavior with a reward, verifier, test suite, or environment feedback.
- If you have both, you can combine them. The common pattern is SFT for the basic behavior, then RL for optimization, but that is not a required order for every task.

### Does MinT support SFT?

Yes. MinT supports SFT directly.

The standard SFT path is:
- `forward_backward(..., loss_fn="cross_entropy")`
- `optim_step(...)`

### Which domain should I use?

Choose by your network path:
- Mainland China -> `https://mint-cn.macaron.xin/`
- Outside Mainland China -> `https://mint.macaron.xin/`

If you are unsure, try the one that matches your region first. The practical goal is lower latency and stable connectivity.

### Where do I get `MINT_API_KEY`?

`MINT_API_KEY` is currently issued by the Mind Lab team.

To request access:
- go to `https://macaron.im/mindlab`
- use **Schedule a Demo**
- or email `contact@mindlab.ltd`

Run the quickstart (SFT then RL in one script):
```bash
python quickstart/quickstart.py
```

Or open the interactive notebook:
```bash
jupyter notebook quickstart/mint_quickstart.ipynb
```

Or run a focused quickstart recipe:
```bash
python quickstart/custom_reward.py
python quickstart/custom_loss.py
```

## Run a Demo

```bash
python demos/rl/adapters/verifiable_math.py      # RL-1: math with exact-match reward
python demos/rl/adapters/preference_chat.py      # RL-2: chat with helpfulness proxy
python demos/rl/adapters/environment_tooluse.py  # RL-3: code gen with execution reward
python demos/embodied/openpi_vla_sdk.py          # Embodied-1: OpenPI via mintx / mint.mint
python demos/embodied/openpi_vla_http.py         # Reference: raw OpenPI FAST HTTP wire shape
```

All demos are configurable via environment variables. See [`demos/rl/README.md`](demos/rl/README.md) for details.

## Advanced Workflows

### Checkpoint Loop (Save -> Download -> Upload -> Resume)

If you want a full checkpoint lifecycle:

```bash
python advanced/checkpoint.py save     --name my-ckpt
python advanced/checkpoint.py download tinker://<run-id>/weights/<ckpt-name> -o ./ckpts
python advanced/checkpoint.py upload   ./ckpts/<archive>.tar.gz
python advanced/checkpoint.py resume   tinker://<run-id>/weights/<ckpt-name> --with-optimizer --steps 3
```

See [`advanced/README.md`](advanced/README.md) for the full command matrix, the optimizer-preserving resume shape (`create_lora_training_client(...)` + `load_state_with_optimizer(...)`), and guardrails (`sampler_weights` vs `weights`).

### MIS Rollout Correction Validation

If you want a focused end-to-end check for session-level Seq-MIS wiring:

```bash
python advanced/validate_mis_rollout_correction.py --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
```

See [`docs/mis_rollout_correction.md`](docs/mis_rollout_correction.md) for prerequisites, env vars, expected output, and failure modes.

### Queue Status Polling

Monitor queue position and estimated wait time for pending sample requests:

```bash
python advanced/queue_status.py
```

Uses the low-level `AsyncTinker` client with backpressure headers to read queue fields from 408 responses.

## Repo Structure

```
mint-quickstart/
  .env.example              # Template for API key configuration
  quickstart/
    quickstart.py           # SFT -> RL in one script
    custom_reward.py        # Client-side reward shaping + importance_sampling
    custom_loss.py          # Pairwise preference training via forward_backward_custom
    sampling_log.py         # Train then inspect model responses
    mint_quickstart.ipynb   # Interactive notebook version
  demos/
    rl/                     # 3 RL demos (available)
      rl_core.py            # Shared GRPO training loop
      adapters/
        verifiable_math.py
        preference_chat.py
        environment_tooluse.py
    vlm/                    # 2 VLM demos (coming soon)
    embodied/               # primary SDK demo + low-level HTTP reference
  advanced/                 # Checkpoint workflows, MIS validation, queue status
  docs/
    roadmap.md              # 6-demo roadmap with status tags
    troubleshooting.md      # Common issues and fixes
    migration-from-minT-demo.md
    experiments/            # Validation reports for quickstart flows
  .pi/
    skills/                 # Project-local pi skills for API, debugging, and issue reporting
  mint-skill/               # AI coding agent migration skill
```

## Tinker SDK Compatibility

If you have existing code using `import tinker`, the lowest-friction MinT migration is:

```python
import mint as tinker
```

Then point the Tinker-style client surface at MinT:

```bash
TINKER_BASE_URL=<your-region-endpoint>
TINKER_API_KEY=<your-mint-api-key>
```

Use the MinT endpoint that matches your region:
- Mainland China: `https://mint-cn.macaron.xin/`
- Outside Mainland China: `https://mint.macaron.xin/`

Why this is the recommended path:
- raw upstream `import tinker` still validates API keys with the `tml-` prefix
- MinT API keys start with `sk-`
- `import mint as tinker` keeps the Tinker-style code shape while enabling MinT compatibility patches

If you must keep the exact `import tinker` statement, import `mint` earlier in the same process before constructing Tinker clients.

## Docs

- [Roadmap](docs/roadmap.md) — all 6 demos with availability status
- [Troubleshooting](docs/troubleshooting.md) — common issues and solutions
- [Migration Guide](docs/migration-from-minT-demo.md) — moving from old MinT-demo repo
- [Quickstart Guide](quickstart/README.md) — first run plus focused custom reward / custom loss recipes
- [RL Demos](demos/rl/README.md) — detailed docs for the 3 available RL demos
- [Embodied Demos](demos/embodied/README.md) — primary OpenPI SDK example plus low-level HTTP reference
- [Advanced](advanced/README.md) — checkpoint workflows and MIS validation entry points
- [MIS Rollout Correction](docs/mis_rollout_correction.md) — targeted Seq-MIS validation flow and troubleshooting
- [Experiment Report](docs/experiments/quickstart-upload-download-resume-report.md) — quickstart upload-download-resume validation template/results
- [Pi Skills](.pi/skills/README.md) — project-local pi skills for API, debugging, and issue reporting
- [Migration Skill](mint-skill/SKILL.md) — AI agent skill for migrating from verl/TRL/OpenRLHF
- [中文 README](README_zh.md) — Chinese version of this document
