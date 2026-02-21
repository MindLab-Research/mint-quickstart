# Troubleshooting

Common issues and solutions when running mint-quickstart demos.

> **Reminder:** All experiments in this repo run against an already deployed MinT server. This repo does not start MinT backend services locally. You only need valid server endpoint + API key credentials.

## Connection Issues

### `MINT_API_KEY not found`

**Symptom:** Script exits with "Missing MINT_API_KEY" or similar error.

**Fix:** Create a `.env` file in the repo root:
```
MINT_API_KEY=sk-mint-your-api-key-here
```

Or export directly:
```bash
export MINT_API_KEY=sk-mint-your-api-key-here
```

### `ConnectionError` / `Connection refused`

**Symptom:** Cannot reach the MinT server.

**Possible causes:**
1. Network connectivity issue — check your internet connection
2. Server is down — check with the MinT team
3. Wrong base URL — if using a custom endpoint, verify `MINT_BASE_URL`

**Fix:**
```bash
curl -s https://mint.macaron.im/health
```
If that returns OK, your network is fine and the server is up.

### `401 Unauthorized` / `403 Forbidden`

**Symptom:** Server rejects your API key.

**Fix:**
- Verify your key starts with `sk-mint-`
- Check the key hasn't expired
- Contact the MinT team for a new key

## Installation Issues

### `ModuleNotFoundError: No module named 'mint'`

**Fix:**
```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git
```

### HuggingFace download failures

**Symptom:** Timeout or connection error when downloading tokenizer/model configs.

**Fix:** Set the HuggingFace mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

This must be set **before** importing `mint`.

## Training Issues

### `forward_backward` hangs or times out

**Possible causes:**
1. Very large batch — reduce `MINT_RL_BATCH` or number of training examples
2. Server under heavy load — wait and retry

**Fix:** Start with small batches:
```bash
MINT_RL_BATCH=4 MINT_RL_GROUP=2 python demos/rl/adapters/verifiable_math.py
```

### All rewards are zero / no gradient signal

**Symptom:** Every RL step shows `accuracy=0.0%` and `datums=0`.

**Possible causes:**
1. Task is too hard for the base model — the model can't produce any correct answers
2. `MAX_TOKENS` too low — model's response gets truncated before the answer

**Fix:**
- Use a larger model: `MINT_BASE_MODEL=Qwen/Qwen3-1.7B`
- Increase max tokens: `MINT_RL_MAX_TOKENS=32`
- Lower temperature for more focused sampling: `MINT_RL_TEMPERATURE=0.7`

### Loss is NaN or training diverges

**Possible causes:**
1. Learning rate too high
2. Advantage values too large

**Fix:**
- Lower learning rate: `MINT_RL_LR=1e-5`
- Increase group size for better advantage estimates: `MINT_RL_GROUP=8`

## Tinker SDK Compatibility

If you have existing code using `import tinker`:

```bash
pip install tinker
```

```
TINKER_BASE_URL=https://mint.macaron.im/
TINKER_API_KEY=<your-mint-api-key>
```

All MinT API calls work identically with `import tinker` instead of `import mint`.

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MINT_API_KEY` | Yes | — | API key for authentication |
| `MINT_BASE_URL` | No | `https://mint.macaron.im/` | Server endpoint |
| `MINT_BASE_MODEL` | No | `Qwen/Qwen3-0.6B` | Base model name |
| `MINT_LORA_RANK` | No | `16` | LoRA adapter rank |
| `HF_ENDPOINT` | No | — | HuggingFace mirror URL |

## Getting Help

1. Check this document first
2. Review the demo README for specific demo issues: `demos/rl/README.md`
3. Open an issue on the repository
