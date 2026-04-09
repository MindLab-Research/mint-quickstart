# Roadmap

`mint-quickstart` provides a unified learning path with 6 demos across 3 tracks.

## Demo Portfolio

| # | Demo | Track | Status | Script |
|---|------|-------|--------|--------|
| 1 | **RL-1 Verifiable Math** | RL | **Available Now** | `demos/rl/adapters/verifiable_math.py` |
| 2 | **RL-2 Preference Chat** | RL | **Available Now** | `demos/rl/adapters/preference_chat.py` |
| 3 | **RL-3 Environment Tool Use** | RL | **Available Now** | `demos/rl/adapters/environment_tooluse.py` |
| 4 | **VLM-1 Vision QA** | VLM | Coming Soon | `demos/vlm/vision_qa.py` |
| 5 | **VLM-2 Vision Instruction** | VLM | Coming Soon | `demos/vlm/vision_instruction.py` |
| 6 | **Embodied-1 OpenPI FAST SDK** | Embodied | **Available Now** | `demos/embodied/openpi_vla_sdk.py` |

## Track Descriptions

### Track A: Reinforcement Learning (Available Now)

Three orthogonal RL demos sharing `rl_core.py` with different reward sources:

- **Verifiable**: deterministic programmatic verifier (exact-match math grading)
- **Preference**: pairwise/judge preference signal (helpfulness proxy)
- **Environment**: delayed trajectory reward from environment feedback (code execution)

### Track B: Vision-Language Models (Coming Soon)

Two VLM demos demonstrating multimodal understanding:

- **Vision QA**: image + question → grounded answer
- **Vision Instruction**: image + task → action/decision output

**Unblock conditions:** MinT SDK VLM support reaches stable; VLM-capable base model available on server.

### Track C: Embodied Agent (Available Now)

One embodied demo now pins down the current MinT SDK path for VLA:

- **OpenPI FAST SDK**: top-level `mint` for the Tinker-compatible client surface plus `mint.mint` / `mintx` for OpenPI-specific training

**Reference path:** keep `openpi_vla_http.py` as the low-level wire-protocol version for debugging and backend comparison.

## Milestones

| Milestone | Contents | Status |
|-----------|----------|--------|
| **M1** | Repo restructure + rl_core.py + 3 RL demos | **Done** |
| **M2** | 2 VLM demos + VLM docs | Planned |
| **M3** | 1 embodied VLA SDK demo + low-level protocol reference | In Progress |

## Capability Ladder

```
Track A: RL (Reasoning & Alignment)        ← Available Now
  └── Verifiable → Preference → Environment

Track B: VLM (Perception & Multimodal)     ← Coming Soon
  └── Vision QA → Vision Instruction

Track C: Embodied (Decision & Action)      ← Available Now
  └── OpenPI FAST SDK
```

The full path demonstrates MinT's range: **reasoning → perception → action**.
