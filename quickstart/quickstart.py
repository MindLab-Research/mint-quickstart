#!/usr/bin/env python3
"""MinT Quickstart - SFT then RL on arithmetic.

Demonstrates the full MinT workflow in a single script:
  1. SFT: teach multiplication with labeled examples
  2. RL: refine with reward signals (exploration + advantage)

Prerequisites:
  - Python >= 3.11
  - pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git
  - MINT_API_KEY set in environment or .env file

Run:
  python quickstart/quickstart.py

All training runs against a remote MinT server.
This script does NOT start any backend services locally.
"""

from __future__ import annotations

import os
import random
import re
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
            stripped = stripped[len("export "):].lstrip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


REPO_ROOT = Path(__file__).resolve().parents[1]
load_env_file(REPO_ROOT / ".env")

MINT_SRC = REPO_ROOT / "mindlab-toolkit" / "src"
if MINT_SRC.exists() and str(MINT_SRC) not in sys.path:
    sys.path.insert(0, str(MINT_SRC))

import mint
from mint import types

MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
SFT_STEPS = int(os.environ.get("MINT_SFT_STEPS", "10"))
SFT_LR = float(os.environ.get("MINT_SFT_LR", "5e-5"))
RL_STEPS = int(os.environ.get("MINT_RL_STEPS", "10"))
RL_LR = float(os.environ.get("MINT_RL_LR", "2e-5"))
RL_BATCH = int(os.environ.get("MINT_RL_BATCH", "8"))
RL_GROUP = int(os.environ.get("MINT_RL_GROUP", "8"))
MAX_TOK = int(os.environ.get("MINT_MAX_TOKENS", "16"))
TEMPERATURE = float(os.environ.get("MINT_TEMPERATURE", "0.7"))

random.seed(42)


def extract_answer(response: str) -> str | None:
    nums = re.findall(r"\d+", response)
    return nums[0] if nums else None


def generate_sft_examples(n: int = 100) -> list[dict]:
    return [
        {"question": f"What is {random.randint(10, 99)} * {random.randint(10, 99)}?"}
        for _ in range(n)
    ]


def process_sft_example(ex: dict, tokenizer) -> types.Datum:
    a, b = map(int, re.findall(r"\d+", ex["question"]))
    answer = str(a * b)
    prompt = f"Question: {ex['question']}\nAnswer:"
    completion = f" {answer}"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_tokens.append(tokenizer.eos_token_id)

    all_tokens = prompt_tokens + completion_tokens
    all_weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = all_weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


print(f"Connecting to MinT server...")
service_client = mint.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=MODEL, rank=RANK, train_mlp=True, train_attn=True, train_unembed=True
)
tokenizer = training_client.get_tokenizer()
print(f"Model: {MODEL}, Vocab: {tokenizer.vocab_size:,}\n")

print("=" * 50)
print("STAGE 1: Supervised Fine-Tuning (SFT)")
print("=" * 50)

sft_examples = generate_sft_examples(100)
sft_data = [process_sft_example(ex, tokenizer) for ex in sft_examples]
print(f"Prepared {len(sft_data)} training examples\n")

for step in range(SFT_STEPS):
    fb = training_client.forward_backward(sft_data, loss_fn="cross_entropy").result()
    total_loss, total_w = 0.0, 0.0
    for i, out in enumerate(fb.loss_fn_outputs):
        lp = out["logprobs"]
        if hasattr(lp, "tolist"):
            lp = lp.tolist()
        w = sft_data[i].loss_fn_inputs["weights"]
        if hasattr(w, "tolist"):
            w = w.tolist()
        for l, wt in zip(lp, w):
            total_loss += -l * wt
            total_w += wt
    loss = total_loss / max(total_w, 1)
    training_client.optim_step(types.AdamParams(learning_rate=SFT_LR)).result()
    print(f"  Step {step + 1:2d}/{SFT_STEPS}: loss = {loss:.4f}")

ckpt = training_client.save_state(name="quickstart-sft").result()
print(f"\nSFT checkpoint: {ckpt.path}")

print("\n" + "=" * 50)
print("STAGE 2: Reinforcement Learning (RL)")
print("=" * 50)

for step in range(RL_STEPS):
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"qs-rl-{step}"
    )
    all_rewards: list[float] = []
    datums: list[types.Datum] = []

    for _ in range(RL_BATCH):
        a, b = random.randint(10, 199), random.randint(10, 199)
        answer = str(a * b)
        prompt = f"Question: What is {a} * {b}?\nAnswer:"
        prompt_tokens = tokenizer.encode(prompt)

        res = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=RL_GROUP,
            sampling_params=types.SamplingParams(
                max_tokens=MAX_TOK,
                temperature=TEMPERATURE,
                stop_token_ids=[tokenizer.eos_token_id],
            ),
        ).result()

        g_rewards, g_responses, g_logprobs = [], [], []
        for seq in res.sequences:
            txt = tokenizer.decode(seq.tokens)
            reward = 1.0 if extract_answer(txt) == answer else 0.0
            g_rewards.append(reward)
            g_responses.append(list(seq.tokens))
            g_logprobs.append(list(seq.logprobs or [0.0] * len(seq.tokens)))

        all_rewards.extend(g_rewards)
        mean_r = sum(g_rewards) / len(g_rewards)
        advs = [r - mean_r for r in g_rewards]
        if all(a == 0 for a in advs):
            continue

        for resp_tok, lp, adv in zip(g_responses, g_logprobs, advs):
            if not resp_tok:
                continue
            full = prompt_tokens + resp_tok
            prefix = len(prompt_tokens) - 1
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full[:-1]),
                    loss_fn_inputs={
                        "target_tokens": full[1:],
                        "weights": [0.0] * prefix + [1.0] * len(resp_tok),
                        "logprobs": [0.0] * prefix + lp,
                        "advantages": [0.0] * prefix + [adv] * len(resp_tok),
                    },
                )
            )

    if datums:
        training_client.forward_backward(datums, loss_fn="importance_sampling").result()
        training_client.optim_step(types.AdamParams(learning_rate=RL_LR)).result()

    acc = sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0
    print(f"  Step {step + 1:2d}/{RL_STEPS}: accuracy = {acc:5.1%}")

ckpt = training_client.save_state(name="quickstart-rl-final").result()
print(f"\nRL checkpoint: {ckpt.path}")
print("\nDone! See demos/ for more advanced examples.")
