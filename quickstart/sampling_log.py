#!/usr/bin/env python3
"""MinT Sampling Log Demo — Train then inspect model responses.

Runs a quick SFT on arithmetic, then samples the trained model on a set
of test questions and prints every response so you can inspect what the
model actually outputs (the "sampling log").

Prerequisites:
  - Python >= 3.11
  - pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git
  - MINT_API_KEY set in environment or .env file

Run:
  python quickstart/sampling_log.py

Env vars:
  MINT_BASE_MODEL     default Qwen/Qwen3-0.6B
  MINT_SFT_STEPS      default 5
  MINT_SFT_LR         default 5e-5
  MINT_MAX_TOKENS      default 16
  MINT_TEMPERATURE     default 0.7
  MINT_NUM_SAMPLES     default 3  (samples per question)
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

for base_dir in (REPO_ROOT.parent, REPO_ROOT):
    for src_dir in ("mindlab-toolkit-alpha/src", "mindlab-toolkit/src"):
        mint_src = base_dir / src_dir
        if mint_src.exists() and str(mint_src) not in sys.path:
            sys.path.insert(0, str(mint_src))
            break
    else:
        continue
    break

import mint  # noqa: E402
import tinker  # noqa: E402
from mint import types  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
SFT_STEPS = int(os.environ.get("MINT_SFT_STEPS", "5"))
SFT_LR = float(os.environ.get("MINT_SFT_LR", "5e-5"))
MAX_TOK = int(os.environ.get("MINT_MAX_TOKENS", "16"))
TEMPERATURE = float(os.environ.get("MINT_TEMPERATURE", "0.7"))
NUM_SAMPLES = int(os.environ.get("MINT_NUM_SAMPLES", "3"))

random.seed(42)

# ---------------------------------------------------------------------------
# Test questions to sample after training
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "What is 23 * 47?",
    "What is 56 * 12?",
    "What is 99 * 11?",
    "What is 38 * 74?",
    "What is 15 * 63?",
]


def _configured_base_url() -> str:
    base_url = os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")
    if not base_url:
        base_url = "https://mint.macaron.xin/"
    return base_url


def _require_api_key() -> str:
    api_key = (os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or "").strip()
    if api_key:
        return api_key
    raise RuntimeError(
        "MINT_API_KEY not found. Set `MINT_API_KEY=sk-your-api-key-here` in the shell "
        f"or add it to `{REPO_ROOT / '.env'}` before running."
    )


def _status_code_from_error(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def preflight_connection(service_client: mint.ServiceClient):
    base_url = _configured_base_url()
    try:
        return service_client.get_server_capabilities()
    except tinker.APITimeoutError as exc:
        raise RuntimeError(
            f"Auth preflight timed out contacting {base_url}. "
            "Check MINT_BASE_URL and retry."
        ) from exc
    except tinker.APIConnectionError as exc:
        raise RuntimeError(
            f"Auth preflight could not reach {base_url}. "
            "Check MINT_BASE_URL, network, and server status."
        ) from exc
    except tinker.APIStatusError as exc:
        status_code = _status_code_from_error(exc)
        if status_code in {401, 403}:
            raise RuntimeError(
                f"Auth preflight rejected (HTTP {status_code}). "
                f"Check MINT_API_KEY for {base_url}."
            ) from exc
        raise RuntimeError(
            f"Auth preflight failed (HTTP {status_code or 'unknown'}) "
            f"from {base_url}."
        ) from exc


def extract_answer(response: str) -> str | None:
    nums = re.findall(r"\d+", response)
    return nums[0] if nums else None


# ---------------------------------------------------------------------------
# SFT data generation (reused from quickstart.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print(f"Connecting to MinT server at {base_url} ...")

        service_client = mint.ServiceClient()
        preflight_connection(service_client)

        training_client = service_client.create_lora_training_client(
            base_model=MODEL, rank=RANK,
            train_mlp=True, train_attn=True, train_unembed=True,
        )
        tokenizer = training_client.get_tokenizer()
        print(f"Model: {MODEL}, Vocab: {tokenizer.vocab_size:,}\n")

        # --- Phase 1: Quick SFT -----------------------------------------------
        print("=" * 60)
        print("Phase 1: Quick SFT (arithmetic)")
        print("=" * 60)

        sft_data = [process_sft_example(ex, tokenizer) for ex in generate_sft_examples(100)]
        print(f"Training data: {len(sft_data)} examples, {SFT_STEPS} steps\n")

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

        # --- Phase 2: Sampling Log ---------------------------------------------
        print("\n" + "=" * 60)
        print("Phase 2: Sampling Log")
        print("=" * 60)
        print(f"Config: num_samples={NUM_SAMPLES}, max_tokens={MAX_TOK}, "
              f"temperature={TEMPERATURE}\n")

        sampling_client = training_client.save_weights_and_get_sampling_client(
            name="sampling-log-demo"
        )

        for qi, question in enumerate(TEST_QUESTIONS, 1):
            prompt = f"Question: {question}\nAnswer:"
            prompt_tokens = tokenizer.encode(prompt)

            result = sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                num_samples=NUM_SAMPLES,
                sampling_params=types.SamplingParams(
                    max_tokens=MAX_TOK,
                    temperature=TEMPERATURE,
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
            ).result()

            a, b = map(int, re.findall(r"\d+", question))
            correct = str(a * b)

            print(f"[Q{qi}] {question}  (correct: {correct})")
            for si, seq in enumerate(result.sequences, 1):
                text = tokenizer.decode(seq.tokens).strip()
                n_tok = len(seq.tokens)
                extracted = extract_answer(text)
                match = "Y" if extracted == correct else "N"
                print(f"  [Sample {si}] {text!r}  ({n_tok} tokens, match={match})")
            print()

        print("Sampling log complete.")
        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
