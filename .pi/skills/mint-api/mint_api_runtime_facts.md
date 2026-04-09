# MinT API Runtime Facts

These facts are useful, but they are more volatile than the core code shapes.
Before stating them strongly, verify them against the current repo files.

## Remote-Only Repo

Current repo position:
- this repo talks to an already deployed MinT server
- it does not start backend services locally

Verify with:
- `../../../README.md`
- `../../../quickstart/README.md`
- `../../../docs/troubleshooting.md`

## Endpoint And Region Defaults

Current docs point users to these endpoints:
- Mainland China: `https://mint-cn.macaron.xin/`
- Outside Mainland China: `https://mint.macaron.xin/`

Verify with:
- `../../../README.md`
- `../../../quickstart/README.md`
- `../../../advanced/README.md`
- `../../../docs/troubleshooting.md`

## Beginner Model Defaults In This Repo

Current quickstart and RL examples usually default to:
- `MINT_BASE_MODEL=Qwen/Qwen3-0.6B`
- `MINT_LORA_RANK=16`

The MIS validation flow is different and currently expects a larger supported model such as `Qwen/Qwen3-30B-A3B-Instruct-2507`.

Verify with:
- `../../../quickstart/README.md`
- `../../../demos/rl/rl_core.py`
- `../../../advanced/README.md`

## Checkpoint URI Behavior In This Repo

Current repo behavior:
- `advanced/checkpoint.py` accepts `mint://` and `tinker://` for download
- resume also accepts legacy `ckpt_...` identifiers
- download normalizes to canonical `mint://` candidates internally
- docs note that current SDKs often print `tinker://...` paths from save flows

Verify with:
- `../../../advanced/checkpoint.py`
- `../../../advanced/README.md`
- `../../../tests/test_checkpoint_paths.py`

## Resume Behavior Used Here

Current repo guidance:
- optimizer-preserving resume uses a fresh `create_lora_training_client(...)` and then `load_state_with_optimizer(...)`
- weights-only resume tries `create_training_client_from_state(...)` first, then falls back to explicit model/rank loading if metadata lookup returns `404`

Verify with:
- `../../../advanced/checkpoint.py`
- `../../../advanced/README.md`

## Sync-First Teaching Style

Current repo code style:
- many examples prefer sync `.result()` calls for readability
- lack of async is not automatically a bug in this repo

Verify with:
- `../../../quickstart/quickstart.py`
- `../../../quickstart/custom_reward.py`
- `../../../quickstart/custom_loss.py`
- `../../../demos/rl/rl_core.py`

## Common First-Line Troubleshooting Facts

Current docs say to check these first:
- `MINT_API_KEY`
- `MINT_BASE_URL`
- connectivity to `/api/v1/healthz`
- batch size, max tokens, learning rate, group size

Verify with:
- `../../../docs/troubleshooting.md`

## Usage Note

When answering, separate stable guidance from volatile repo facts. Good wording:
- stable: "the repo uses this code shape"
- volatile: "the current repo docs say..." or "the current checkpoint helper does..."
