---
description: Help migrate LLM training code from verl, TRL, or similar frameworks to MinT
argument-hint: [framework_name or migration_question]
---

# MinT Migration Assistant

Help users migrate their LLM training code to MinT from verl, TRL, OpenRLHF, or similar frameworks.

## Instructions

When invoked:

1. Identify the source framework (verl, TRL, OpenRLHF, custom PyTorch)
2. Map concepts from source framework to MinT equivalents
3. Provide before/after code examples for the specific migration pattern
4. Highlight key differences (distributed training, checkpointing, loss functions)
5. Warn about common pitfalls (token alignment, learning rate scaling, async patterns)

## Reference

Read `mint_api_reference.txt` in this directory for complete MinT API documentation.

## Concept Mapping

### verl to MinT

| verl | MinT |
|------|------|
| `RolloutWorker` | `SamplingClient` |
| `ActorRolloutRefWorker` | `TrainingClient` + `SamplingClient` |
| `RewardManager` | User-defined reward function |
| `PPOTrainer` | `forward_backward(loss_fn="ppo")` + `optim_step()` |
| `DataProto` | `types.Datum` |
| `vllm` backend | MinT handles inference internally |
| `fsdp`/`megatron` sharding | MinT handles distributed training internally |

### TRL to MinT

| TRL | MinT |
|-----|------|
| `SFTTrainer` | `forward_backward(loss_fn="cross_entropy")` loop |
| `PPOTrainer` | `forward_backward(loss_fn="ppo")` loop |
| `DPOTrainer` | `forward_backward_custom()` with DPO loss |
| `AutoModelForCausalLM` | `service_client.create_lora_training_client()` |
| HuggingFace dataset | Convert to `list[types.Datum]` |

## Key Differences

1. **Distributed training**: MinT handles sharding server-side. No FSDP/Megatron config needed.

2. **Model loading**: Specify model by name, not local path.
   ```python
   training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B")
   ```

3. **Inference**: Built-in `SamplingClient` instead of vLLM workers.
   ```python
   sampling_client = training_client.save_weights_and_get_sampling_client(name="...")
   ```

4. **Checkpointing**: Named checkpoints with `save_state()` / `save_weights_for_sampler()`.

5. **Loss functions**: String selector for built-in losses.
   - `"cross_entropy"` - SFT
   - `"importance_sampling"` - Basic policy gradient
   - `"ppo"` - Clipped policy gradient
   - `"cispo"` - Clipped importance sampling
   - `"dro"` - Direct reward optimization

## Common Pitfalls

1. **Async pattern**: Always call `.result()` on futures.
   ```python
   training_client.forward_backward(data, loss_fn).result()
   training_client.optim_step(params).result()
   ```

2. **Token alignment**: Next-token prediction format.
   ```python
   input_tokens = all_tokens[:-1]
   target_tokens = all_tokens[1:]
   weights = weights[1:]  # Aligned with targets
   ```

3. **LoRA learning rate**: 20-100x higher than full fine-tuning.

4. **Gradient accumulation**: Multiple `forward_backward` calls before single `optim_step`.
