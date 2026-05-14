#!/usr/bin/env python3
"""MinT Concepts: Completers

Demonstrates TokenCompleter vs MessageCompleter:
  - TokenCompleter: operates on token IDs, used by RL algorithms
  - MessageCompleter: operates on messages, used by evaluators and judges

Run:
  python concepts/completers.py

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


def demonstrate_completers() -> None:
    """Show TokenCompleter vs MessageCompleter patterns."""
    _require_api_key()
    base_url = _configured_base_url()
    print("Connecting to MinT server at", base_url)

    service_client = mint.ServiceClient()
    capabilities = preflight_connection(service_client)
    print(f"Server supports {_supported_model_count(capabilities)} models")

    print("\n=== TokenCompleter vs MessageCompleter ===")
    print("\nTokenCompleter:")
    print("  Input:  ModelInput (token IDs)")
    print("  Output: TokensWithLogprobs (tokens + logprobs)")
    print("  Use:    RL algorithms that compute advantages at token level")

    print("\nMessageCompleter:")
    print("  Input:  list[Message] (role + content dicts)")
    print("  Output: Message (assistant response)")
    print("  Use:    Evaluators, judges, LLM-as-judge patterns")

    print("\n=== LLM-as-Judge Pattern ===")
    print("A MessageCompleter can compare two responses and emit a preference:")
    print("  judge_prompt = f'{prompt}\\n--- Response A ---\\n{a}\\n--- Response B ---\\n{b}'")
    print("  preference = sampling_client.sample(judge_prompt)")
    print("  if 'better' in preference.lower(): return 'A', else: return 'B'")

    print("\n=== Stop Token Handling ===")
    print("TokenCompleter respects stop sequences from the renderer:")
    print("  stops = renderer.get_stop_sequences()")
    print("  tokens = token_completer.complete(prompt, stop=stops)")
    print("  -> Stops generation when encountering user turn or EOS")


def main() -> int:
    """Entry point."""
    try:
        demonstrate_completers()
        print("\nCompleters concept demonstration complete.")
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
