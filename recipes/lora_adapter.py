#!/usr/bin/env python3
"""MinT Recipe: LoRA Adapter Export

Export trained LoRA weights to PEFT format for use with vLLM/SGLang.
Demonstrates checkpoint saving and PEFT-compatible weight export.

Run:
  python recipes/lora_adapter.py

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


MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))


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


def main() -> int:
    """Demonstrate LoRA adapter export."""
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        capabilities = preflight_connection(service_client)
        print(f"Server supports {_supported_model_count(capabilities)} models")

        print(f"\n=== LoRA Adapter Export ===")
        print(f"Base Model: {MODEL}")
        print(f"LoRA Rank: {RANK}")

        print(f"\n=== Training Phase ===")
        print(f"""
# Train a LoRA adapter
training_client = await service_client.create_lora_training_client_async(
    base_model=MODEL, rank=RANK
)

for step in range(10):
    batch = load_batch()
    loss = await training_client.forward_backward(
        loss_fn="cross_entropy",
        data=batch
    )
    await training_client.optim_step(loss)

# Save weights and get checkpoint reference
weights_path = await training_client.save_weights_and_get_sampling_client()
""")

        print(f"\n=== Export to PEFT Format ===")
        print(f"""
# Export trained LoRA to PEFT-compatible format
export_dir = "/tmp/mint-lora-export-run-123/"

# MinT provides export utilities
export_result = await service_client.export_lora_to_peft(
    weights=weights_path,
    output_dir=export_dir,
    base_model=MODEL,
    target_modules=["q_proj", "v_proj"],  # Which modules have LoRA
)

print(f"Exported to {{export_dir}}")
print(f"Files: {{export_result.files}}")
""")

        print(f"\n=== Expected Output Files ===")
        print(f"/tmp/mint-lora-export-run-123/")
        print(f"  adapter_config.json        # PEFT configuration")
        print(f"  adapter_model.bin          # LoRA weights (PyTorch format)")
        print(f"  training_args.bin          # Training hyperparameters")
        print(f"  README.md                  # Documentation")

        print(f"\n=== Using in vLLM/SGLang ===")
        print(f"""
# With vLLM:
from vllm import LLM

llm = LLM(
    model="{MODEL}",
    enable_lora=True,
    max_lora_rank={RANK}
)

# Load the exported LoRA
outputs = llm.generate(
    prompts=["What is 5+3?"],
    lora_request=LoRARequest("math", "/tmp/mint-lora-export-run-123/")
)
print(outputs[0].outputs[0].text)
# -> " 8"
""")

        print(f"\n=== Export Formats ===")
        print(f"Default (PEFT):       adapter_config.json + adapter_model.bin")
        print(f"Alternative (HF):     Hugging Face safe_tensors format")
        print(f"Alternative (GGUF):   GGML quantized format for cpu inference")

        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
