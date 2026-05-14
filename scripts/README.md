# scripts/

Operational scripts for `mint-quickstart-alpha`.

## `run_verification.sh`

Verification driver for the [`customize-tutorials-verification`](https://github.com/MindLab-Research/agents-cd5e3759fa/blob/main/openspec/changes/customize-tutorials-verification) OpenSpec change.

For each script listed in `verification_targets.txt`, the driver:

1. Loads `MINT_API_KEY` and `MINT_BASE_URL` from the repo's `.env`.
2. Runs the target Python script via `python <path>`.
3. Pipes captured stdout+stderr through a masking sed (`sk-[A-Za-z0-9_-]+` → `sk-***MASKED***`).
4. Writes `<script>.run.log` next to the script.
5. Appends a dated entry to `<script>.verified.md` recording exit code, runtime, endpoint, and pass/fail status.

### Usage

```bash
# Re-run every target whose verified.md is older than $MINT_VERIFY_FRESH_DAYS (default 7).
scripts/run_verification.sh

# Force re-run regardless of recency.
scripts/run_verification.sh --force

# Run only specific targets (path relative to repo root).
scripts/run_verification.sh quickstart/custom_loss.py demos/rl/adapters/verifiable_math.py
```

### Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `MINT_VERIFY_RETRIES` | `3` | Max attempts for a target before marking failed. RL scripts can be stochastic. |
| `MINT_VERIFY_FRESH_DAYS` | `7` | Skip targets whose `verified.md` is newer than this many days, unless `--force`. |

### Convergence criteria

The driver only records exit code and runtime — it does **not** enforce algorithm-specific loss thresholds. Per OpenSpec design.md Decision 2, those thresholds are documented per-script in the corresponding `<script>.expected.txt` file. Reviewers compare `run.log` to `expected.txt` to judge convergence:

- **SFT**: final loss ≤ 0.7 × initial loss across recorded checkpoints.
- **DPO / preference**: chosen-token logprob > rejected-token logprob at last checkpoint.
- **RL (GRPO)**: reward_mean monotonically non-decreasing across ≥60% of checkpoints, OR final reward_mean ≥ 1.5 × initial reward_mean.
- **Concept demos** (rendering, completers, weights, evaluations, async-patterns): non-zero exit and non-empty expected artifact.

### Adding a target

1. Create the Python script (must read `.env` and use `mint.macaron.xin` / `mint-cn.macaron.xin` endpoint).
2. Create a corresponding `tests/test_<basename>.py` that calls into the script's importable functions.
3. Append the script's path to `verification_targets.txt` under the appropriate wave.
4. Run `scripts/run_verification.sh path/to/your_script.py` to produce the first `verified.md`.

## `verification_targets.txt`

Plain-text list of target Python script paths (relative to repo root). Comments start with `#`. The list is grouped by wave (Wave 1–4 per the OpenSpec change). Disabled (commented-out) entries are skipped by the driver.
