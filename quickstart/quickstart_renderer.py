#!/usr/bin/env python3
"""MinT Quickstart (renderer variant) - SFT using mintx.renderers.

Same arithmetic SFT as quickstart.py stage 1, but uses the renderer API
(build_supervised_example + datum_from_model_input_weights) instead
of manual token encoding. Does NOT include the RL stage (stage 2) from
quickstart.py — see quickstart.py for the full SFT+RL workflow.

Run:
  python quickstart/quickstart_renderer.py

All training runs against a remote MinT server.
This script does NOT start any backend services locally.
"""

from __future__ import annotations

import os
import random
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

import mint
import mint.mint as mintx
import tinker
from mint import types

MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
SFT_STEPS = int(os.environ.get("MINT_SFT_STEPS", "10"))
SFT_LR = float(os.environ.get("MINT_SFT_LR", "5e-5"))
# datum_from_model_input_weights requires max_length; quickstart.py has no equivalent truncation
MAX_LENGTH = int(os.environ.get("MINT_MAX_LENGTH", "2048"))

random.seed(42)


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
        f"or add it to `{REPO_ROOT / '.env'}` before running quickstart_renderer.py."
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
            f"Auth preflight timed out while contacting {base_url}. "
            "Check `MINT_BASE_URL` and retry."
        ) from exc
    except tinker.APIConnectionError as exc:
        raise RuntimeError(
            f"Auth preflight could not reach {base_url}. "
            "Check `MINT_BASE_URL`, network access, and server status."
        ) from exc
    except tinker.APIStatusError as exc:
        status_code = _status_code_from_error(exc)
        if status_code in {401, 403}:
            raise RuntimeError(
                "Auth preflight was rejected by the MinT server "
                f"(HTTP {status_code}). Check that `MINT_API_KEY` is valid for {base_url}."
            ) from exc
        raise RuntimeError(
            "Auth preflight failed with an unexpected MinT server response "
            f"(HTTP {status_code or 'unknown'}) from {base_url}."
        ) from exc


def generate_sft_messages(n: int = 100) -> list[list[dict]]:
    examples = []
    for _ in range(n):
        a, b = random.randint(10, 99), random.randint(10, 99)
        answer = str(a * b)
        examples.append([
            {"role": "user", "content": f"What is {a} * {b}?"},
            {"role": "assistant", "content": answer},
        ])
    return examples


def main() -> int:
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        preflight_connection(service_client)

        training_client = service_client.create_lora_training_client(
            base_model=MODEL, rank=RANK, train_mlp=True, train_attn=True, train_unembed=True
        )
        tokenizer = training_client.get_tokenizer()
        print(f"Model: {MODEL}, Vocab: {tokenizer.vocab_size:,}\n")

        renderer_name = mintx.renderers.get_recommended_renderer_name(MODEL)
        renderer = mintx.renderers.get_renderer(renderer_name, tokenizer)
        print(f"Renderer: {renderer_name}")

        print("\n" + "=" * 50)
        print("SFT with Renderer")
        print("=" * 50)

        all_messages = generate_sft_messages(100)
        sft_data = []
        for messages in all_messages:
            model_input, weights = renderer.build_supervised_example(messages)
            datum = mintx.renderers.datum_from_model_input_weights(
                model_input, weights, max_length=MAX_LENGTH
            )
            sft_data.append(datum)
        print(f"Prepared {len(sft_data)} training examples via renderer\n")

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

        ckpt = training_client.save_state(name="quickstart-renderer-sft").result()
        print(f"\nSFT checkpoint: {ckpt.path}")
        print("\nDone! Compare with quickstart.py (manual tokenization) for the same task.")
        return 0
    except RuntimeError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
