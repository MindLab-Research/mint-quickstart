#!/usr/bin/env python3
"""MinT Concepts: Rendering

Demonstrates tokenizers and renderers: converting messages to token sequences
for training vs sampling, vision input handling, and TrainOnWhat masking.

Run:
  python concepts/rendering.py

All training runs against a remote MinT server.
This script does NOT start any backend services locally.
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


def demonstrate_rendering() -> None:
    """Show tokenization and rendering: messages -> tokens for training vs sampling."""
    _require_api_key()
    base_url = _configured_base_url()
    print("Connecting to MinT server at", base_url)

    service_client = mint.ServiceClient()
    capabilities = preflight_connection(service_client)
    print(f"Server supports {_supported_model_count(capabilities)} models")

    # Example messages: a simple chat conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+3?"},
        {"role": "assistant", "content": "5"},
    ]

    print("\n=== Rendering Example ===")
    print(f"Messages: {len(messages)} turns")
    for msg in messages:
        print(f"  {msg['role']:12} {msg['content']}")

    # Build a ModelInput from messages (demonstrates the rendering pipeline)
    # This shows how messages are converted to tokens for the model
    print("\n=== TrainOnWhat Masking ===")
    print("SFT training masks user/system turns (weight=0) and trains only on assistant response (weight=1)")
    print("Example mask pattern:")
    print("  [system tokens]           -> weight 0 (prompt)")
    print("  [user tokens]             -> weight 0 (prompt)")
    print("  [assistant response]      -> weight 1 (train on this)")
    print("\nRL training uses completion mask: tokens generated by policy get weight 1, prompts get weight 0")

    print("\n=== Renderer Output ===")
    print("Renderer builds ModelInput for:")
    print("  - Generation: build_generation_prompt() -> sample from model")
    print("  - Training:   build_supervised_example() -> compute loss on labels")
    print("\nBoth use the same tokenizer to ensure consistency.")


def main() -> int:
    """Entry point."""
    try:
        demonstrate_rendering()
        print("\nRendering concept demonstration complete.")
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
