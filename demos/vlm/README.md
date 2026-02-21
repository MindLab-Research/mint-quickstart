# VLM Demos

> **Status: Coming Soon** (Milestone M2)

Two vision-language model demos are planned for this track.

## Planned Demos

### VLM-1: Vision QA

- **Scope:** image + text question → grounded answer with simple eval
- **Planned script:** `vision_qa.py`
- **Planned deliverables:** runnable adapter using `rl_core.py`, sample images, eval function

### VLM-2: Vision Instruction / Decision

- **Scope:** image + task instruction → action/decision text (or structured output)
- **Planned script:** `vision_instruction.py`
- **Planned deliverables:** runnable adapter, task prompts, structured output parser

## Unblock Conditions

- MinT SDK VLM support reaches stable
- VLM-capable base model (e.g., `Qwen/Qwen3-VL-*`) available on server
- Image input format finalized in `mint.types`

## Prerequisites (Expected)

- MinT SDK with VLM support
- A VLM-capable base model hosted on the MinT server
- `MINT_API_KEY` with VLM access

## Timeline

Check the [roadmap](../../docs/roadmap.md) for current status.
