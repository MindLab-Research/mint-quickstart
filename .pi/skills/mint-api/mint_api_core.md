# MinT API Core

This file keeps the stable code shapes that show up across `mint-quickstart`.
Use it for writing or reviewing repo-local MinT client code.

## 1. Client Creation

Most repo scripts use a simple sync shape:

```python
import mint
from mint import types

service_client = mint.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-0.6B",
    rank=16,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
)
```

Prefer the repo's copy-friendly pattern before introducing wrappers.

## 2. SFT Datum Shape

Use next-token prediction alignment exactly:

```python
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
completion_tokens.append(tokenizer.eos_token_id)

all_tokens = prompt_tokens + completion_tokens
all_weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

input_tokens = all_tokens[:-1]
target_tokens = all_tokens[1:]
weights = all_weights[1:]

datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=input_tokens),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "weights": weights,
    },
)
```

If `weights` still has the same length as `all_tokens`, the shape is usually wrong.

## 3. SFT Training Loop Shape

The repo mostly uses a readable sync-first loop:

```python
fb = training_client.forward_backward(data, loss_fn="cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=lr)).result()
```

For this repo, sync is not automatically a bug. Many examples are teaching baselines.

## 4. RL Datum Shape

The shared GRPO loop in `demos/rl/rl_core.py` follows this shape:

```python
full_tokens = prompt_tokens + response_tokens
prefix_len = len(prompt_tokens) - 1
response_len = len(response_tokens)

datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": full_tokens[1:],
        "weights": [0.0] * prefix_len + [1.0] * response_len,
        "logprobs": [0.0] * prefix_len + sampling_logprobs,
        "advantages": [0.0] * prefix_len + [advantage] * response_len,
    },
)
```

Checks that catch common mistakes:
- `target_tokens`, `weights`, `logprobs`, and `advantages` must align in length
- prompt positions are usually masked with `0.0`
- empty responses should usually be skipped

## 5. Reward Loop Shape

Repo RL flow is:

```python
sampling_client = training_client.save_weights_and_get_sampling_client(name="rl-step-1")
result = sampling_client.sample(
    prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
    num_samples=group_size,
    sampling_params=types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop_token_ids=[tokenizer.eos_token_id],
    ),
).result()

# score each sequence on the client side
# turn scores into advantages
training_client.forward_backward(datums, loss_fn="importance_sampling").result()
training_client.optim_step(types.AdamParams(learning_rate=lr)).result()
```

## 6. Custom Loss Shape

Use `forward_backward_custom(...)` when built-in losses do not fit.
In this repo, `quickstart/custom_loss.py` is the main example.

```python
result = training_client.forward_backward_custom(data, pairwise_preference_loss).result()
training_client.optim_step(types.AdamParams(learning_rate=loss_lr)).result()
```

Prefer this path for pairwise preference or DPO-style losses.

## 7. Sampling Shape

Sampling still returns a future in the sync examples:

```python
result = sampling_client.sample(
    prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
    num_samples=4,
    sampling_params=types.SamplingParams(max_tokens=32),
).result()
```

Do not forget `.result()`.

## 8. Checkpoint Shape

Sampling checkpoint:

```python
sampler_ckpt = training_client.save_weights_for_sampler(name="demo-sampler").result()
```

Full training checkpoint:

```python
state_ckpt = training_client.save_state(name="demo-state").result()
```

True optimizer-preserving resume shape used in this repo:

```python
training_client = service_client.create_lora_training_client(
    base_model=model,
    rank=rank,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
)
training_client.load_state_with_optimizer(checkpoint_path).result()
```

Weights-only load shape:

```python
training_client.load_state(checkpoint_path).result()
```

Do not present `load_state(...)` as full optimizer resume.

## 9. When To Reach For More Detail

- Need file-specific patterns: read `mint_api_examples.md`
- Need defaults or current repo behavior: read `mint_api_runtime_facts.md`
- Need broader SDK details: read `../../../mint-skill/mint_api_reference.txt`
