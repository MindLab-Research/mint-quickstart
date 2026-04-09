---
name: mint-client-debugger
description: Diagnose MinT client-side problems in this repo. Use for slow scripts, missing `.result()`, wrong checkpoint paths, bad token alignment, all-zero rewards, NaN loss, queue confusion, or suspected client misuse in `quickstart/`, `demos/`, `advanced/`, or user scripts.
---

# MinT Client Debugger

Use this skill when the user has a MinT client-side symptom and wants a concrete diagnosis.

## Output Contract

Default to this structure:

```text
Symptom
Likely causes
What to check now
Most likely fix
What to verify after fix
```

Be direct. Point to exact files or code shapes whenever possible.

## First Pass

1. Classify the script before suggesting changes:
   - teaching or simple baseline
   - throughput-oriented experiment
   - eval or queue-monitoring workflow
2. Read `debug_checklist.md`.
3. Pull in `symptom_index.md` or `concurrency_patterns.md` only if they match the symptom.
4. Confirm whether the problem is:
   - env or setup mistake
   - client code bug
   - repo bug or docs gap
   - likely upstream MinT SDK or backend issue

## Source Of Truth Files

- `debug_checklist.md`
- `symptom_index.md`
- `concurrency_patterns.md`
- `../../../docs/troubleshooting.md`
- `../../../demos/rl/rl_core.py`
- `../../../advanced/checkpoint.py`
- `../../../advanced/queue_status.py`
- `../../../mint-skill/mint_api_reference.txt`

## Hard Rules

- Check env and connectivity before blaming the training loop.
- Do not immediately blame missing concurrency.
- In this repo, sync-first teaching code is normal and often correct.
- Separate checkpoint-path bugs from optimizer-resume bugs.
- Separate client-side mistakes from backend ownership.
- If the issue becomes a confirmed repo bug or docs gap, hand off to `../mint-issue-reporter/SKILL.md`.

## Do Not Use For

- writing new feature code from scratch without a symptom
- framework migration from `verl` or `TRL`
- filing a GitHub issue before ownership is clear

## Quick Routing

- API call shape, `.result()`, bad datum fields: read `debug_checklist.md`
- slow or queue-related behavior: read `concurrency_patterns.md` and `../../../advanced/queue_status.py`
- all-zero rewards, NaN loss, datums empty: read `symptom_index.md` and `../../../docs/troubleshooting.md`
- checkpoint save, download, or resume failures: read `debug_checklist.md` and `../../../advanced/checkpoint.py`
