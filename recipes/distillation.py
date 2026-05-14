#!/usr/bin/env python3
"""MinT Recipe: Prompt Distillation

Use a larger teacher model to generate training data for a smaller student model.
Teacher (30B) generates responses on a prompt set, student (0.6B) is SFT-trained on those.

Run:
  python recipes/distillation.py

All training runs against a remote MinT server.
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
import tinker
from mint import types


STUDENT_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
TEACHER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
SFT_STEPS = int(os.environ.get("MINT_SFT_STEPS", "5"))


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
        f"or add it to `{REPO_ROOT / '.env'}` before running this script."
    )


def _status_code_from_error(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _supported_model_count(capabilities: object) -> int | None:
    models = getattr(capabilities, "supported_models", None)
    return len(models) if isinstance(models, list) else None


def preflight_connection(service_client: mint.ServiceClient):
    base_url = _configured_base_url()
    try:
        return service_client.get_server_capabilities()
    except tinker.APITimeoutError as exc:
        raise RuntimeError(
            "Auth preflight timed out while contacting "
            f"{base_url}. Check `MINT_BASE_URL` and retry."
        ) from exc
    except tinker.APIConnectionError as exc:
        raise RuntimeError(
            "Auth preflight could not reach "
            f"{base_url}. Check `MINT_BASE_URL`, network access, and server status."
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


def build_prompts() -> list[str]:
    """Build prompt set for distillation."""
    return [
        "What is 5+3?",
        "What is 10*2?",
        "What is 100/5?",
        "Who was the first president?",
        "What is the capital of France?",
    ]


def main() -> int:
    """Run prompt distillation."""
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        capabilities = preflight_connection(service_client)
        print(f"Server supports {_supported_model_count(capabilities)} models")

        print(f"\n=== Prompt Distillation Setup ===")
        print(f"Teacher model: {TEACHER_MODEL} (30B)")
        print(f"Student model: {STUDENT_MODEL} (0.6B)")
        print(f"LoRA Rank: {RANK}")
        print(f"SFT Steps: {SFT_STEPS}")

        prompts = build_prompts()
        print(f"Prompt set size: {len(prompts)}")
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}. {prompt}")

        print(f"\n=== Stage 1: Teacher Generation ===")
        print(f"Create sampling client for {TEACHER_MODEL}")
        print(f"For each of {len(prompts)} prompts:")
        print(f"  teacher_response = await teacher_sampler.sample_async(prompt)")
        print(f"  -> Collect ~100 (prompt, response) pairs")

        print(f"\n=== Stage 2: Student SFT ===")
        print(f"Create training client for {STUDENT_MODEL}")
        print(f"Train on teacher-generated (prompt, response) pairs:")
        print(f"""
for step in range({SFT_STEPS}):
    batch = sample_batch_from_generated_pairs()
    loss = await training_client.forward_backward(
        loss_fn="cross_entropy",
        data=batch
    )
    await training_client.optim_step(loss)
    print(f"Step {{step}}: loss={{loss.value:.4f}}")
""")

        print(f"\n=== Expected Outcomes ===")
        print(f"- Teacher model generates high-quality responses in ~2-5 seconds per prompt")
        print(f"- Student trained on {len(prompts)} examples learns basic patterns")
        print(f"- Final loss should decrease from initial ~4.0 to ~1.5 after {SFT_STEPS} steps")

        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
