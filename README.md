# MinT Quickstart

Tutorial for training language models with [MinT](https://github.com/MindLab-Research/mindlab-toolkit) (Mind Lab Toolkit) using SFT and RL.

## What's Included

- `mint_quickstart.ipynb` - Complete tutorial: train a model to solve multiplication using SFT, then refine with RL
- `mint-skill/` - Migration skill for converting code from verl/TRL/OpenRLHF to MinT

## Using the Migration Skill

The `mint-skill/` directory contains a skill that helps AI coding agents migrate your existing training code to MinT.

**Claude Code:**
```bash
cp -r mint-skill/ /path/to/your/project/.claude/skills/
```

Then ask Claude Code to migrate your code:
```
Help me migrate my verl PPO training loop to MinT
```

**Other coding agents:** Copy `mint-skill/` into your agent's skills directory (consult your agent's documentation). The skill reads `SKILL.md` for instructions and `mint_api_reference.txt` for API details.

**Supported frameworks:** verl, TRL, OpenRLHF, custom PyTorch training loops.

## Quick Start

**Requirements:** Python >= 3.11

```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git python-dotenv matplotlib numpy
```

Create `.env`:
```
MINT_API_KEY=sk-mint-your-api-key-here
```

Open `mint_quickstart.ipynb` and run the cells.

## Tinker Compatibility

MinT is fully API-compatible with [Tinker](https://tinker.thinkingmachines.ai). If you prefer, you can use the `tinker` package with MinT by configuring environment variables to point to the MinT server:

```
TINKER_BASE_URL=https://mint.macaron.im/
TINKER_API_KEY=<your-mint-api-key>
```

## Tutorial Overview

| Stage | Method | Loss Function | Goal |
|-------|--------|---------------|------|
| 1 | SFT | `cross_entropy` | Learn multiplication from labeled examples |
| 2 | RL | `importance_sampling` | Refine with reward signals |

Key API:
```python
import mint

service_client = mint.ServiceClient()
training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=16)

# Train
training_client.forward_backward(data, loss_fn="cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()

# Checkpoint
checkpoint = training_client.save_state(name="my-model").result()

# Inference
sampling_client = training_client.save_weights_and_get_sampling_client(name="my-model")
sampling_client.sample(prompt, num_samples=1, sampling_params=params).result()
```
