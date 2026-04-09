# Debug Checklist

Use this list in order. Most client-side MinT problems in this repo fall into one of these buckets.

## 1. Environment And Connectivity

Check these first:
- `MINT_API_KEY` exists and looks valid
- `MINT_BASE_URL` points to the right region when set
- the server is reachable
- tokenizer or Hugging Face setup is not failing first

Repo references:
- `../../../docs/troubleshooting.md`
- `../../../README.md`

Typical clues:
- `Missing MINT_API_KEY`
- `401 Unauthorized`
- `403 Forbidden`
- `ConnectionError`
- tokenizer download timeout before training starts

## 2. Sync Or Async Call Shape

Common client mistake: forgetting how MinT calls finish.

Check for these patterns:
- sync request methods missing `.result()`
- mixed async style copied from docs without checking the actual method shape
- code assuming `sample(...)` already returns the final response

Repo references:
- `../../../quickstart/README.md`
- `../../../demos/rl/rl_core.py`
- `../../../advanced/checkpoint.py`
- `../../../mint-skill/mint_api_reference.txt`

Typical wrong shapes:

```python
training_client.forward_backward(data, loss_fn="cross_entropy")
training_client.optim_step(types.AdamParams(learning_rate=lr))
```

Typical right shapes for this repo:

```python
training_client.forward_backward(data, loss_fn="cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=lr)).result()
```

## 3. Datum And Token Alignment

Most silent training bugs come from bad shapes here.

Check:
- `input_tokens` uses `all_tokens[:-1]`
- `target_tokens` uses `all_tokens[1:]`
- `weights` aligns with `target_tokens`, not `all_tokens`
- RL `logprobs` and `advantages` match the target length
- prompt positions are masked when expected

Repo references:
- `../../../quickstart/quickstart.py`
- `../../../quickstart/custom_reward.py`
- `../../../demos/rl/rl_core.py`

Typical symptoms:
- `datums=0`
- no learning signal
- shape mismatch or length mismatch errors
- reward loop runs but training does not improve

## 4. Checkpoint And Resume Path

Split the problem into the right subtype.

Check:
- is the path for training state or sampler weights?
- does the code need optimizer state or only weights?
- is the path `mint://`, `tinker://`, or legacy `ckpt_...`?
- is the code using `load_state_with_optimizer(...)` when true resume is needed?

Repo references:
- `../../../advanced/checkpoint.py`
- `../../../advanced/README.md`
- `../../../tests/test_checkpoint_paths.py`

Fast rule:
- sampling / inference -> `save_weights_for_sampler()`
- true resume -> `save_state()` + `load_state_with_optimizer()`
- weights-only reload -> `load_state()`

## 5. Performance And Concurrency

Only evaluate this after the basic call shape is correct.

Check:
- is the script supposed to be a simple teaching baseline?
- is it doing many independent sample requests serially?
- is it actually blocked on remote queue wait, not local Python?
- is someone proposing unbounded `gather(...)`?

Repo references:
- `concurrency_patterns.md`
- `../../../advanced/queue_status.py`
- `../../../mint-skill/mint_api_reference.txt`

Do not mark sync code as wrong just because it is not async.

## 6. Reward Signal And Optimization

Check:
- all rewards zero
- advantages all zero
- group size too small for stable comparison
- `max_tokens` too low and responses are truncated
- learning rate too high causing NaN or divergence

Repo references:
- `../../../docs/troubleshooting.md`
- `../../../demos/rl/README.md`
- `../../../demos/rl/rl_core.py`

Common first fixes:
- raise `MINT_RL_MAX_TOKENS`
- lower `MINT_RL_LR`
- increase `MINT_RL_GROUP`
- use a stronger base model for harder tasks

## Final Classification

Before answering, classify the root cause as one of:
- setup mistake
- client code bug
- repo docs gap
- likely upstream SDK or backend gap

Only the third and fourth cases should usually move toward issue reporting.
