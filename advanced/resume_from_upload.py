#!/usr/bin/env python3
"""Resume training from an uploaded checkpoint.

Run:
  MINT_API_KEY=... MINT_RESUME_PATH=ckpt_... \
    python demo/upload_weights/resume_from_upload_weights.py

Optional:
  MINT_BASE_URL=...            # default https://mint.macaron.im
  MINT_BASE_MODEL=...          # base model if loading weights only
  MINT_LORA_RANK=16            # LoRA rank if loading weights only
  MINT_RL_LR=5e-5              # learning rate
  MINT_STEPS=3                 # training steps to run
  MINT_RESUME_WITH_OPTIMIZER=1 # use create_training_client_from_state_with_optimizer

Notes:
- If your uploaded archive lacks optimizer state, set MINT_RESUME_WITH_OPTIMIZER=0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    print(f"Missing {name}. Set {name}=... and retry.")
    sys.exit(1)


demo_dir = Path(__file__).resolve().parents[1]
load_env_file(demo_dir / ".env")

# Allow running against a local checkout without installing `mindlab-toolkit`.
repo_root = Path(__file__).resolve().parents[2]
mint_src = repo_root / "mindlab-toolkit" / "src"
if mint_src.exists() and str(mint_src) not in sys.path:
    sys.path.insert(0, str(mint_src))

import mint
from mint import types


require_env("MINT_API_KEY")
BASE_URL = os.environ.get("MINT_BASE_URL")
MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
LR = float(os.environ.get("MINT_RL_LR", "5e-5"))
STEPS = int(os.environ.get("MINT_STEPS", "3"))
RESUME_PATH = os.environ.get("MINT_RESUME_PATH")
RESUME_WITH_OPTIMIZER = os.environ.get("MINT_RESUME_WITH_OPTIMIZER", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

if not RESUME_PATH:
    print("Missing MINT_RESUME_PATH (e.g. ckpt_...).")
    sys.exit(1)

PROMPT = "Bug: add() returns wrong sum.\nFix:\n"
ANSWER = "def add(a, b):\n    return a + b\n"


def build_datum(training_client: mint.TrainingClient) -> types.Datum:
    tokenizer = training_client.get_tokenizer()
    prompt_tokens = tokenizer.encode(PROMPT)
    full_tokens = tokenizer.encode(PROMPT + ANSWER)
    if len(full_tokens) < 2:
        raise ValueError("Prompt+answer must tokenize to at least 2 tokens.")
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    prefix_len = max(0, len(prompt_tokens) - 1)
    weights = [0.0] * prefix_len + [1.0] * (len(target_tokens) - prefix_len)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def train_once(training_client: mint.TrainingClient) -> None:
    datum = build_datum(training_client)
    training_client.forward_backward([datum], loss_fn="cross_entropy").result()
    training_client.optim_step(types.AdamParams(learning_rate=LR)).result()


service_client = mint.ServiceClient(base_url=BASE_URL) if BASE_URL else mint.ServiceClient()

if RESUME_WITH_OPTIMIZER:
    training_client = service_client.create_training_client_from_state_with_optimizer(RESUME_PATH)
else:
    training_client = service_client.create_lora_training_client(
        base_model=MODEL,
        rank=RANK,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    training_client.load_state(RESUME_PATH).result()

print(f"Resumed from {RESUME_PATH}")

for _ in range(STEPS):
    train_once(training_client)

ckpt = training_client.save_state(name="resume-from-upload").result()
print(f"Saved: {ckpt.path}")
