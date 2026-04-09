# Concurrency Patterns

Use this file only when the symptom is speed, queue delay, or async confusion.

## Step 1: Classify The Script

Before recommending concurrency changes, classify the script.

### Teaching Or Simple Baseline

Examples:
- `../../../quickstart/quickstart.py`
- `../../../quickstart/custom_reward.py`
- `../../../quickstart/custom_loss.py`
- `../../../demos/rl/rl_core.py`

Guidance:
- sync-first code is acceptable here
- readability is part of the design
- do not call this wrong just because it is not async

### Throughput-Oriented Experiment

Examples:
- user scripts doing many independent sample calls
- large eval jobs
- custom batch samplers

Guidance:
- recommend bounded concurrency only when it fits the workload
- preserve request order when rewards or metrics depend on input order
- measure before and after

### Eval Or Queue Monitoring Workflow

Examples:
- scripts focused on monitoring waits or batch sampling
- `../../../advanced/queue_status.py`

Guidance:
- queue wait can dominate wall time
- local Python changes may not fix a remote queue bottleneck

## Step 2: Recommend The Right Pattern

### Keep It Sync

Recommend this when:
- the script is a tutorial or baseline
- the user needs clarity more than throughput
- there are only a few RPC calls per step

### Overlap Or Async Sampling

Recommend this when:
- the script is clearly throughput-sensitive
- there are many independent sample requests
- the user explicitly asks for speedup

Use bounded concurrency, not unbounded fan-out.

### Queue Monitoring Instead Of Rewrite

Recommend this when:
- the script is waiting on server capacity
- users think the code is hung but the real issue is queue delay

Point to `../../../advanced/queue_status.py`.

## Red Flags

Avoid these recommendations:
- "make everything async" without checking the script type
- unbounded `asyncio.gather(...)`
- changing reward or logging order accidentally
- blaming sync code when the remote queue is the bottleneck

## Safe Review Language

Good:
- "This script is a teaching baseline, so sync is acceptable here."
- "If you want higher sampling throughput, add bounded concurrency around independent sample requests."
- "Check queue wait before rewriting the client code."

Bad:
- "This is wrong because it is not async."
- "Just add concurrency everywhere."
