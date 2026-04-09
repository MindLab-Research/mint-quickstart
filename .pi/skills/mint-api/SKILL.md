---
name: mint-api
description: Write, review, or explain MinT client code in this repo. Use for `import mint` code in `quickstart/`, `demos/`, `advanced/`, `tests/`, or docs tied to those scripts. Covers repo-approved training, sampling, checkpoint, and RL patterns.
---

# MinT API

Use this skill when the task is about repo-local MinT client code.

Do not use this skill for generic backend ownership, public issue drafting, or framework migration from `verl` / `TRL` / `OpenRLHF`.
For migration tasks, use the existing `../../../mint-skill/SKILL.md` skill instead.

## First Pass

1. Classify the task:
   - quickstart or beginner example
   - RL demo or reward loop
   - checkpoint workflow
   - troubleshooting or explanation
2. Read only the smallest relevant references:
   - `mint_api_core.md` for stable code shapes
   - `mint_api_examples.md` for repo copy patterns
   - `mint_api_runtime_facts.md` for defaults and volatile facts
3. Verify any current runtime claim against repo source before presenting it as fact.

## Source Of Truth Files

Start from these repo files before inventing a new pattern:

- `../../../README.md`
- `../../../quickstart/README.md`
- `../../../demos/rl/README.md`
- `../../../demos/rl/rl_core.py`
- `../../../advanced/README.md`
- `../../../advanced/checkpoint.py`
- `../../../docs/troubleshooting.md`
- `../../../mint-skill/mint_api_reference.txt`

## Hard Rules

- For sync request methods such as `forward_backward(...)`, `optim_step(...)`, `sample(...)`, `save_state(...)`, and `load_state(...)`, finish the call with `.result()`.
- Keep next-token alignment correct: `input=all_tokens[:-1]`, `target=all_tokens[1:]`, `weights=all_weights[1:]`.
- Distinguish checkpoint types clearly:
  - `save_state()` / `load_state_with_optimizer()` for training resume
  - `save_weights_for_sampler()` for sampling or inference
- Prefer repo-established sync-first teaching shapes unless the user explicitly asks for throughput tuning.
- Treat endpoint defaults, checkpoint URI behavior, and SDK quirks as runtime facts. Label them as current repo behavior and cite where they were checked.
- Reuse repo patterns before inventing helpers. In this repo, `quickstart/` and `demos/` are intentionally copy-friendly.

## Do Not Use For

- deciding whether a problem is backend-owned
- drafting GitHub issues
- pure framework migration without repo-local code changes

Hand off confirmed issue drafting to `../mint-issue-reporter/SKILL.md`.

## Decision Guide

- New SFT or RL script: read `mint_api_core.md`, then `mint_api_examples.md`
- RL datum shape or reward loop: read `mint_api_core.md`, then `../../../demos/rl/rl_core.py`
- Checkpoint save, download, upload, or resume: read `mint_api_runtime_facts.md`, then `../../../advanced/checkpoint.py`
- Repo defaults, env vars, or endpoint notes: read `mint_api_runtime_facts.md`, then `../../../README.md` and `../../../docs/troubleshooting.md`
- Full SDK details not covered here: read `../../../mint-skill/mint_api_reference.txt`
