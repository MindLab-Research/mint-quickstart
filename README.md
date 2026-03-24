# MinT Quickstart

[English](./README.md) | [中文](./README_zh.md)

The single entry repo for learning [MinT](https://github.com/MindLab-Research/mindlab-toolkit) (Mind Lab Toolkit) — from first API call to advanced RL training.

> **Important:** All experiments run against an already deployed MinT server. This repo does **not** start MinT backend services locally. You only need valid server endpoint + API key credentials.

## Demo Portfolio

### Available Now

| # | Demo | Track | Reward Source | Script |
|---|------|-------|---------------|--------|
| 1 | **RL-1 Verifiable Math** | RL | Deterministic verifier | [`demos/rl/adapters/verifiable_math.py`](demos/rl/adapters/verifiable_math.py) |
| 2 | **RL-2 Preference Chat** | RL | Pairwise/judge preference | [`demos/rl/adapters/preference_chat.py`](demos/rl/adapters/preference_chat.py) |
| 3 | **RL-3 Environment Tool Use** | RL | Code execution feedback | [`demos/rl/adapters/environment_tooluse.py`](demos/rl/adapters/environment_tooluse.py) |

### Coming Soon

| # | Demo | Track | Description | Status |
|---|------|-------|-------------|--------|
| 4 | **VLM-1 Vision QA** | VLM | Image + question -> grounded answer | Planned (M2) |
| 5 | **VLM-2 Vision Instruction** | VLM | Image + task -> action/decision | Planned (M2) |
| 6 | **Embodied-1 Simulator Agent** | Embodied | Simplified env -> action sequences | Planned (M3) |

## Quick Start

**Requirements:** Python >= 3.11, a MinT API key

```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git python-dotenv matplotlib numpy
```

Create `.env` in the repo root:
```
MINT_API_KEY=sk-mint-your-api-key-here
```

If the default base URL is slow, use one of these endpoints in `MINT_BASE_URL`:
- Mainland China: `https://mint.macaron.xin/`
- Outside Mainland China: `https://i18n.mint.macaron.xin/`

Run the quickstart (SFT then RL in one script):
```bash
python quickstart/quickstart.py
```

Or open the interactive notebook:
```bash
jupyter notebook quickstart/mint_quickstart.ipynb
```

## Run a Demo

```bash
python demos/rl/adapters/verifiable_math.py      # RL-1: math with exact-match reward
python demos/rl/adapters/preference_chat.py      # RL-2: chat with helpfulness proxy
python demos/rl/adapters/environment_tooluse.py  # RL-3: code gen with execution reward
```

All demos are configurable via environment variables. See [`demos/rl/README.md`](demos/rl/README.md) for details.

## Advanced Workflows

### Checkpoint Loop (Save -> Download -> Upload -> Resume)

If you want a full checkpoint lifecycle:

```bash
python advanced/checkpoint.py save     --name my-ckpt
python advanced/checkpoint.py download mint://<run-id>/weights/<ckpt-name> -o ./ckpts
python advanced/checkpoint.py upload   ./ckpts/<archive>.tar.gz
python advanced/checkpoint.py resume   ckpt_<id> --with-optimizer --steps 3
```

See [`advanced/README.md`](advanced/README.md) for the full command matrix and guardrails (`sampler_weights` vs `weights`).

### MIS Rollout Correction Validation

If you want a focused end-to-end check for session-level Seq-MIS wiring:

```bash
python advanced/validate_mis_rollout_correction.py --base-model Qwen/Qwen3-0.6B
```

See [`docs/mis_rollout_correction.md`](docs/mis_rollout_correction.md) for prerequisites, env vars, expected output, and failure modes.

## Repo Structure

```
mint-quickstart/
  .env.example              # Template for API key configuration
  quickstart/
    quickstart.py           # SFT -> RL in one script
    mint_quickstart.ipynb   # Interactive notebook version
  demos/
    rl/                     # 3 RL demos (available)
      rl_core.py            # Shared GRPO training loop
      adapters/
        verifiable_math.py
        preference_chat.py
        environment_tooluse.py
    vlm/                    # 2 VLM demos (coming soon)
    embodied/               # 1 embodied demo (coming soon)
  advanced/                 # Checkpoint workflows and MIS validation
  docs/
    roadmap.md              # 6-demo roadmap with status tags
    troubleshooting.md      # Common issues and fixes
    migration-from-minT-demo.md
    experiments/            # Validation reports for quickstart flows
  mint-skill/               # AI coding agent migration skill
```

## Tinker SDK Compatibility

If you have existing code using `import tinker`:

```bash
pip install tinker
```

```
TINKER_BASE_URL=https://mint.macaron.im/
TINKER_API_KEY=<your-mint-api-key>
```

If `https://mint.macaron.im/` is slow, try `https://mint.macaron.xin/` (Mainland China) or `https://i18n.mint.macaron.xin/` (outside Mainland China).

All code works identically with `import tinker` instead of `import mint`.

## Docs

- [Roadmap](docs/roadmap.md) — all 6 demos with availability status
- [Troubleshooting](docs/troubleshooting.md) — common issues and solutions
- [Migration Guide](docs/migration-from-minT-demo.md) — moving from old MinT-demo repo
- [RL Demos](demos/rl/README.md) — detailed docs for the 3 available RL demos
- [Advanced](advanced/README.md) — checkpoint workflows and MIS validation entry points
- [MIS Rollout Correction](docs/mis_rollout_correction.md) — targeted Seq-MIS validation flow and troubleshooting
- [Experiment Report](docs/experiments/quickstart-upload-download-resume-report.md) — quickstart upload-download-resume validation template/results
- [Migration Skill](mint-skill/SKILL.md) — AI agent skill for migrating from verl/TRL/OpenRLHF
- [中文 README](README_zh.md) — Chinese version of this document
