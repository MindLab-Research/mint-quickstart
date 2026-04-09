# Quickstart

Get from zero to a trained model in under 30 minutes.

> **Note:** All training runs against a remote MinT server. This repo does NOT start any backend services locally. You only need a valid server endpoint + API key.

## Prerequisites

- Python >= 3.11
- A MinT API key (starts with `sk-`)

## Setup

```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git python-dotenv matplotlib numpy
```

Create a `.env` file in the repo root:
```
MINT_API_KEY=sk-your-api-key-here
```

Use the MinT endpoint that matches your region:
- Mainland China: `https://mint-cn.macaron.xin/`
- Outside Mainland China: `https://mint.macaron.xin/`

## Run

**Script (recommended for first run):**
```bash
python quickstart/quickstart.py
```

**Notebook (interactive exploration):**
```bash
jupyter notebook quickstart/mint_quickstart.ipynb
```

**Focused recipes:**
```bash
python quickstart/custom_reward.py
python quickstart/custom_loss.py
```

## What You'll Learn

The quickstart demonstrates a two-stage training pipeline:

| Stage | Method | Loss Function | Goal |
|-------|--------|---------------|------|
| 1 | SFT | `cross_entropy` | Teach multiplication from labeled examples |
| 2 | RL | `importance_sampling` | Refine with reward signals |

### Key API Methods

```python
import mint

# Connect
service_client = mint.ServiceClient()
training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=16)

# Train
training_client.forward_backward(data, loss_fn="cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()

# Save
checkpoint = training_client.save_state(name="my-model").result()

# Sample
sampling_client = training_client.save_weights_and_get_sampling_client(name="my-model")
sampling_client.sample(prompt, num_samples=1, sampling_params=params).result()
```

## Expected Output

```
Connecting to MinT server...
Endpoint: https://mint.macaron.xin/
Auth preflight: OK (N supported models)

Model: Qwen/Qwen3-0.6B, Vocab: 151,936

==================================================
STAGE 1: Supervised Fine-Tuning (SFT)
==================================================
Prepared 100 training examples

  Step  1/10: loss = 8.xxxx
  Step  2/10: loss = 6.xxxx
  ...
  Step 10/10: loss = 2.xxxx

SFT checkpoint: ckpt_quickstart-sft_...

==================================================
STAGE 2: Reinforcement Learning (RL)
==================================================
  Step  1/10: accuracy =  x.x%
  ...
  Step 10/10: accuracy = xx.x%

RL checkpoint: ckpt_quickstart-rl-final_...

Done! See demos/ and the focused quickstart recipes for more advanced examples.
```

## Configuration

All parameters are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINT_BASE_MODEL` | `Qwen/Qwen3-0.6B` | Base model name |
| `MINT_LORA_RANK` | `16` | LoRA rank |
| `MINT_SFT_STEPS` | `10` | Number of SFT training steps |
| `MINT_SFT_LR` | `5e-5` | SFT learning rate |
| `MINT_RL_STEPS` | `10` | Number of RL training steps |
| `MINT_RL_LR` | `2e-5` | RL learning rate |
| `MINT_RL_BATCH` | `8` | Problems per RL step |
| `MINT_RL_GROUP` | `8` | Samples per problem (GRPO group size) |

## Next Steps

After the quickstart, try the focused recipes:
- `quickstart/custom_reward.py` — client-side reward shaping with partial credit and `importance_sampling`
- `quickstart/custom_loss.py` — pairwise preference training with `forward_backward_custom`

Then explore the [RL demos](../demos/rl/) for broader RL patterns:
- **Verifiable Math**: deterministic reward with a programmatic verifier
- **Preference Chat**: alignment RL with a helpfulness proxy
- **Environment Tool Use**: code generation with execution-based reward

If you want the current SDK path for VLA, see [Embodied Demos](../demos/embodied/) and `demos/embodied/openpi_vla_sdk.py`. It uses `import mint` for the Tinker-compatible client surface and `import mint.mint as mintx` for the MinT-only OpenPI extension.

If you want the raw wire protocol for debugging or backend comparison, see `demos/embodied/openpi_vla_http.py`. That file keeps the HTTP request shape visible without going through the SDK helper layer.
