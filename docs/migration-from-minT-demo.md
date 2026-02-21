# Migration from MinT-Demo

If you previously used the standalone `MinT-demo` repository, this guide helps you migrate to `mint-quickstart`.

## Why Migrate?

`mint-quickstart` is now the **single official entry repo** for MinT demos. The old `MinT-demo` repository is deprecated.

Key improvements:
- Unified structure with quickstart + 6-demo roadmap
- Shared `rl_core.py` + adapter pattern for RL demos
- Consistent environment variable configuration
- Comprehensive docs (troubleshooting, roadmap)

## What Changed

### Directory Mapping

| Old (`MinT-demo` / `demo/`) | New (`mint-quickstart`) |
|------------------------------|-------------------------|
| `demo/math_rl/math_rl_demo.py` | `demos/rl/adapters/verifiable_math.py` |
| `demo/chat_rl/chat_rl_demo.py` | `demos/rl/adapters/preference_chat.py` |
| `demo/code_rl/code_rl_demo.py` | `demos/rl/adapters/environment_tooluse.py` |
| `demo/rft_resume/rft_resume_demo.py` | `advanced/resume.py` |
| `demo/upload_weights/upload_weights_demo.py` | `advanced/upload_weights.py` |
| `demo/upload_weights/resume_from_upload_weights.py` | `advanced/resume_from_upload.py` |
| `mint_quickstart.ipynb` (root) | `quickstart/mint_quickstart.ipynb` |

### Run Command Changes

| Old | New |
|-----|-----|
| `python demo/math_rl/math_rl_demo.py` | `python demos/rl/adapters/verifiable_math.py` |
| `python demo/chat_rl/chat_rl_demo.py` | `python demos/rl/adapters/preference_chat.py` |
| `python demo/code_rl/code_rl_demo.py` | `python demos/rl/adapters/environment_tooluse.py` |

### Configuration

Environment variables are unchanged. The same `.env` file works:
```
MINT_API_KEY=sk-mint-your-api-key-here
```

Place `.env` in the repo root. All scripts load from there.

## Step-by-Step Migration

1. **Clone the new repo** (or pull latest if you already have it):
   ```bash
   git clone https://github.com/MindLab-Research/mint-quickstart.git
   cd mint-quickstart
   ```

2. **Copy your `.env` file** from the old repo:
   ```bash
   cp /path/to/old/MinT-demo/.env .env
   ```

3. **Update your run commands** using the mapping table above.

4. **Existing checkpoints** are compatible — no retraining needed. Resume paths still work:
   ```bash
   MINT_RESUME_PATH=ckpt_... python advanced/resume.py
   ```

## API Compatibility

The MinT Python SDK API has not changed. All existing code using `mint.ServiceClient()`, `training_client.forward_backward()`, etc. works as-is.

If you use Tinker SDK (`import tinker`), it also continues to work with:
```
TINKER_BASE_URL=https://mint.macaron.im/
TINKER_API_KEY=<your-mint-api-key>
```
