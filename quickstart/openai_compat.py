#!/usr/bin/env python3
"""MinT OpenAI-Compatible Inference Demo.

Calls a deployed MinT model through its OpenAI-compatible endpoint
(`/oai/api/v1`) using the official `openai` SDK. No `mint`/`tinker` import
needed — this is plain inference against an already-deployed model.

Prerequisites:
  - Python >= 3.11
  - pip install openai
  - MINT_API_KEY set in environment or .env file (use "dummy" if the
    server runs without authentication)

Run:
  python quickstart/openai_compat.py chat \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --user-message "Reply with exactly: pong"

  python quickstart/openai_compat.py completions \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --prompt "The capital of France is"

  python quickstart/openai_compat.py tool \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --user-message "What is the weather in Beijing?"

  python quickstart/openai_compat.py smoke \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507

Env vars:
  MINT_BASE_URL    Server base URL WITHOUT the /oai/api/v1 suffix
  MINT_API_KEY     API key (falls back to "dummy")
  MINT_OAI_MODEL   Default model name for --model

All calls run against a remote MinT server. This script does NOT start any
backend services locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


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

DEFAULT_MODEL = os.environ.get("MINT_OAI_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_BASE_URL = "https://mint.macaron.xin/"


# ---------------------------------------------------------------------------
# Configuration helpers (pure — safe to unit test offline)
# ---------------------------------------------------------------------------
def resolve_base_url(cli_base_url: Optional[str] = None) -> str:
    """Server base URL WITHOUT the /oai/api/v1 suffix."""
    base = cli_base_url or os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")
    return base or DEFAULT_BASE_URL


def oai_base_url(cli_base_url: Optional[str] = None) -> str:
    """OpenAI client base_url — server URL plus the compatible prefix."""
    return resolve_base_url(cli_base_url).rstrip("/") + "/oai/api/v1"


def resolve_api_key(cli_api_key: Optional[str] = None) -> str:
    return cli_api_key or os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or "dummy"


def tool_spec(name: str, description: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]


# ---------------------------------------------------------------------------
# Request payload builders (pure — the test asserts on these)
# ---------------------------------------------------------------------------
def build_completions_payload(model: str, prompt: str, *, max_tokens: int, temperature: float,
                              top_p: float, stop: Optional[str] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop is not None:
        payload["stop"] = stop
    return payload


def build_chat_payload(model: str, user_message: str, *, system: Optional[str] = None,
                       max_tokens: int, temperature: float, top_p: float,
                       stop: Optional[str] = None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop is not None:
        payload["stop"] = stop
    return payload


def build_tool_payload(model: str, user_message: str, *, tool_name: str, tool_description: str,
                      tool_choice: str, max_tokens: int, temperature: float,
                      top_p: float) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "tools": tool_spec(tool_name, tool_description),
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def _mask(value: str) -> str:
    return value if len(value) <= 8 else value[:4] + "..." + value[-4:]


def _print_io(*, base_url: str, api_key: str, endpoint: str, payload: dict, response: Any) -> None:
    print("=== input ===")
    print(json.dumps({"base_url": base_url, "api_key": _mask(api_key),
                      "endpoint": endpoint, "payload": payload}, ensure_ascii=False, indent=2))
    print("=== output ===")
    print(response.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Runners (require the openai SDK + a reachable server)
# ---------------------------------------------------------------------------
def run_completions(client: Any, args: argparse.Namespace) -> None:
    payload = build_completions_payload(args.model, args.prompt, max_tokens=args.max_tokens,
                                        temperature=args.temperature, top_p=args.top_p, stop=args.stop)
    resp = client.completions.create(**payload)
    _print_io(base_url=args.oai_base_url, api_key=args.api_key, endpoint="/completions",
              payload=payload, response=resp)


def run_chat(client: Any, args: argparse.Namespace) -> None:
    payload = build_chat_payload(args.model, args.user_message, system=args.system,
                                 max_tokens=args.max_tokens, temperature=args.temperature,
                                 top_p=args.top_p, stop=args.stop)
    resp = client.chat.completions.create(**payload)
    _print_io(base_url=args.oai_base_url, api_key=args.api_key, endpoint="/chat/completions",
              payload=payload, response=resp)


def run_tool(client: Any, args: argparse.Namespace) -> None:
    payload = build_tool_payload(args.model, args.user_message, tool_name=args.tool_name,
                                 tool_description=args.tool_description, tool_choice=args.tool_choice,
                                 max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
    resp = client.chat.completions.create(**payload)
    calls = resp.choices[0].message.tool_calls
    if not calls and not args.allow_no_tool_call:
        raise SystemExit("model returned no tool_calls (use --allow-no-tool-call to permit)")
    _print_io(base_url=args.oai_base_url, api_key=args.api_key, endpoint="/chat/completions",
              payload=payload, response=resp)


def run_smoke(client: Any, args: argparse.Namespace) -> None:
    """Exercise the supported surface and confirm unsupported calls error out."""
    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"[{'ok ' if ok else 'BAD'}] {name}: {detail}")

    try:
        models = client.models.list()
        record("models.list", True, f"{len(models.data)} models")
    except Exception as exc:  # noqa: BLE001
        record("models.list", False, f"{type(exc).__name__}: {exc}")

    try:
        client.chat.completions.create(**build_chat_payload(
            args.model, "Reply with exactly: pong", system=None,
            max_tokens=16, temperature=0.0, top_p=1.0))
        record("chat.completions", True, "ok")
    except Exception as exc:  # noqa: BLE001
        record("chat.completions", False, f"{type(exc).__name__}: {exc}")

    # Unsupported: must raise, not silently succeed.
    try:
        client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "hi"}],
                                       stream=True)
        record("stream=True rejected", False, "unexpected success")
    except Exception as exc:  # noqa: BLE001
        record("stream=True rejected", True, f"{type(exc).__name__}")

    if not all(ok for _, ok, _ in results):
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=None, help="Server base URL WITHOUT /oai/api/v1 suffix")
    p.add_argument("--api-key", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    cmpl = sub.add_parser("completions", help="Legacy /completions example")
    cmpl.add_argument("--model", default=DEFAULT_MODEL)
    cmpl.add_argument("--prompt", required=True)
    cmpl.add_argument("--max-tokens", type=int, default=32)
    cmpl.add_argument("--temperature", type=float, default=0.2)
    cmpl.add_argument("--top-p", type=float, default=0.9)
    cmpl.add_argument("--stop", default=None)

    chat = sub.add_parser("chat", help="/chat/completions example")
    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument("--system", "--system-message", dest="system", default=None)
    chat.add_argument("--user", "--user-message", dest="user_message", required=True)
    chat.add_argument("--max-tokens", type=int, default=32)
    chat.add_argument("--temperature", type=float, default=0.2)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.add_argument("--stop", default=None)

    tool = sub.add_parser("tool", help="Single tool-calling example")
    tool.add_argument("--model", default=DEFAULT_MODEL)
    tool.add_argument("--user", "--user-message", dest="user_message",
                      default="What is the weather in Beijing?")
    tool.add_argument("--tool-name", default="get_weather")
    tool.add_argument("--tool-description", default="Get current weather for a city")
    tool.add_argument("--tool-choice", default="auto")
    tool.add_argument("--max-tokens", type=int, default=128)
    tool.add_argument("--temperature", type=float, default=0.1)
    tool.add_argument("--top-p", type=float, default=0.9)
    tool.add_argument("--allow-no-tool-call", action="store_true")

    smoke = sub.add_parser("smoke", help="Check supported + unsupported surface")
    smoke.add_argument("--model", default=DEFAULT_MODEL)
    smoke.add_argument("--max-tokens", type=int, default=128)

    args = p.parse_args(argv)
    args.oai_base_url = oai_base_url(args.base_url)
    args.api_key = resolve_api_key(args.api_key)
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("openai SDK not installed. Run: pip install openai", file=sys.stderr)
        return 1

    client = OpenAI(base_url=args.oai_base_url, api_key=args.api_key)
    runners = {"completions": run_completions, "chat": run_chat, "tool": run_tool, "smoke": run_smoke}
    runners[args.cmd](client, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
