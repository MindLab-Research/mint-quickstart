"""Shared helpers for MinT recipe scripts.

The recipe files are meant to run from a checkout or from an installed package.
This module centralizes environment loading, local source discovery, API-key
checks, and MinT preflight errors so individual recipes can focus on training
logic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://mint.macaron.xin/"


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE or export KEY=VALUE lines if not already set."""
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


def _add_local_mint_source_to_path() -> None:
    """Prefer sibling local mindlab-toolkit sources when available."""
    for base_dir in (REPO_ROOT.parent, REPO_ROOT):
        for src_dir in ("mindlab-toolkit-alpha/src", "mindlab-toolkit/src"):
            mint_src = base_dir / src_dir
            if mint_src.exists() and str(mint_src) not in sys.path:
                sys.path.insert(0, str(mint_src))
                return


load_env_file(REPO_ROOT / ".env")
_add_local_mint_source_to_path()

import mint  # noqa: E402
import tinker  # noqa: E402


def configured_base_url() -> str:
    """Return configured MinT/Tinker endpoint, defaulting to production MinT."""
    return os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL


def require_api_key() -> str:
    """Return the configured API key or raise a clear setup error."""
    api_key = (os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or "").strip()
    if api_key:
        return api_key
    raise RuntimeError(
        "MINT_API_KEY not found. Set `MINT_API_KEY=sk-your-api-key-here` in the shell "
        f"or add it to `{REPO_ROOT / '.env'}` before running this script."
    )


def status_code_from_error(exc: Exception) -> int | None:
    """Extract HTTP status code from common Tinker API exception shapes."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def supported_model_count(capabilities: object) -> int | None:
    """Return supported model count when the capabilities response exposes it."""
    models = getattr(capabilities, "supported_models", None)
    return len(models) if isinstance(models, list) else None


def supported_model_names(capabilities: object) -> set[str]:
    """Return supported model names from several possible capability shapes."""
    models = getattr(capabilities, "supported_models", None)
    if not isinstance(models, list):
        return set()

    names: set[str] = set()
    for model in models:
        if isinstance(model, str):
            names.add(model)
            continue
        for attr in ("model_name", "name", "id"):
            value = getattr(model, attr, None)
            if isinstance(value, str):
                names.add(value)
                break
    return names


def is_model_supported(capabilities: object, model_name: str) -> bool | None:
    """Return True/False if support is known; None if the response has no list."""
    names = supported_model_names(capabilities)
    if not names:
        return None
    return model_name in names


def extract_content(message: dict) -> str:
    """Extract text from message content (handles string or list of blocks)."""
    raw = message.get("content", "")
    if isinstance(raw, list):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in raw)
    return str(raw)


def preflight_connection(service_client: mint.ServiceClient) -> Any:
    """Call get_server_capabilities() and classify common auth/network errors."""
    base_url = configured_base_url()
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
        status_code = status_code_from_error(exc)
        if status_code in {401, 403}:
            raise RuntimeError(
                "Auth preflight was rejected by the MinT server "
                f"(HTTP {status_code}). Check that `MINT_API_KEY` is valid for {base_url}."
            ) from exc
        raise RuntimeError(
            "Auth preflight failed with an unexpected MinT server response "
            f"(HTTP {status_code or 'unknown'}) from {base_url}."
        ) from exc


def make_service_client() -> tuple[mint.ServiceClient, Any]:
    """Create a ServiceClient after API-key validation and server preflight."""
    require_api_key()
    service_client = mint.ServiceClient()
    capabilities = preflight_connection(service_client)
    return service_client, capabilities


__all__ = [
    "REPO_ROOT",
    "DEFAULT_BASE_URL",
    "load_env_file",
    "mint",
    "tinker",
    "make_service_client",
    "preflight_connection",
    "configured_base_url",
    "require_api_key",
    "status_code_from_error",
    "supported_model_count",
    "supported_model_names",
    "is_model_supported",
    "extract_content",
]
