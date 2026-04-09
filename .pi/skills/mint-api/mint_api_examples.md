# MinT API Examples

Use these repo files as copy sources. Prefer copying the smallest useful shape.

## `../../../quickstart/quickstart.py`

Use when the user wants the shortest end-to-end example.

What it gives you:
- one script that runs SFT and then RL
- sync-first, readable `forward_backward(...).result()` and `optim_step(...).result()` usage
- simple checkpoint save flow

Good for:
- first MinT script
- teaching or explanation
- lightweight modifications

## `../../../quickstart/custom_reward.py`

Use when the task is about client-side reward shaping.

What it gives you:
- reward computed on the client
- `importance_sampling` training loop
- practical reward engineering example

Good for:
- custom scalar reward functions
- debugging reward signal logic
- adapting RL examples to a new task

## `../../../quickstart/custom_loss.py`

Use when the user needs a custom differentiable loss.

What it gives you:
- `forward_backward_custom(...)`
- pairwise preference training shape
- clear separation between data prep and custom loss code

Good for:
- preference training
- DPO-like experiments
- custom loss debugging

## `../../../quickstart/sampling_log.py`

Use when the task is about sampling after training and inspecting responses.

What it gives you:
- save checkpoint
- create sampler
- inspect outputs after training

Good for:
- qualitative checks
- demo scripts that need sample logs

## `../../../demos/rl/rl_core.py`

Use when the task is about a reusable RL loop.

What it gives you:
- shared GRPO loop
- reward -> advantage -> datum construction shape
- clean adapter boundary: dataset, prompt, reward, evaluation

Good for:
- adding a new RL adapter
- reviewing RL datum construction
- understanding prompt-level sampling and training flow

## `../../../advanced/checkpoint.py`

Use when the task is about checkpoint lifecycle or path handling.

What it gives you:
- `save`, `download`, `upload`, `resume`
- `mint://`, `tinker://`, and `ckpt_...` path handling
- weights-only vs optimizer-preserving resume split

Good for:
- checkpoint bug fixes
- docs updates around resume behavior
- copying a robust CLI shape

## `../../../advanced/queue_status.py`

Use when the task is about queue polling or server wait visibility.

What it gives you:
- low-level async status polling
- backpressure-oriented monitoring logic

Good for:
- diagnosing queue delays
- explaining why a request looks stuck even when client code is fine

Do not use it as the default template for training scripts.

## `../../../docs/troubleshooting.md`

Use when the task is mainly diagnostic or docs-oriented.

What it gives you:
- common env problems
- common reward and divergence symptoms
- repo-approved first checks

Good for:
- debugging answers
- docs edits
- separating client mistakes from setup mistakes

## Copy Strategy

- Copy the smallest working shape, not whole files by default.
- Keep beginner examples readable.
- Only introduce async or extra helpers when the user really needs throughput, overlap, or queue-aware behavior.
