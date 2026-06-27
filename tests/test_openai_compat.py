from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "quickstart" / "openai_compat.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_openai_compat", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def test_oai_base_url_appends_prefix(mod):
    assert mod.oai_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/oai/api/v1"
    # trailing slash must not double up
    assert mod.oai_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000/oai/api/v1"


def test_resolve_base_url_env_fallback(mod, monkeypatch):
    monkeypatch.delenv("MINT_BASE_URL", raising=False)
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    assert mod.resolve_base_url() == mod.DEFAULT_BASE_URL
    monkeypatch.setenv("MINT_BASE_URL", "http://host:9000")
    assert mod.resolve_base_url() == "http://host:9000"


def test_resolve_api_key_defaults_to_dummy(mod, monkeypatch):
    monkeypatch.delenv("MINT_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    assert mod.resolve_api_key() == "dummy"
    assert mod.resolve_api_key("real-key") == "real-key"


def test_completions_payload_omits_stop_when_none(mod):
    payload = mod.build_completions_payload(
        "Qwen/Qwen3-30B-A3B-Instruct-2507", "The capital of France is",
        max_tokens=16, temperature=0.1, top_p=0.9,
    )
    assert payload["prompt"] == "The capital of France is"
    assert "stop" not in payload
    payload2 = mod.build_completions_payload("M", "p", max_tokens=8, temperature=0.0, top_p=1.0, stop="\n")
    assert payload2["stop"] == "\n"


def test_chat_payload_system_message_order(mod):
    payload = mod.build_chat_payload("M", "hi", system="sys", max_tokens=16, temperature=0.0, top_p=1.0)
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user"]
    # no system message -> user only
    payload2 = mod.build_chat_payload("M", "hi", max_tokens=16, temperature=0.0, top_p=1.0)
    assert [m["role"] for m in payload2["messages"]] == ["user"]


def test_tool_payload_shape(mod):
    payload = mod.build_tool_payload(
        "M", "weather?", tool_name="get_weather", tool_description="d",
        tool_choice="required", max_tokens=8, temperature=0.1, top_p=0.9,
    )
    assert payload["tool_choice"] == "required"
    fn = payload["tools"][0]["function"]
    assert fn["name"] == "get_weather"
    assert fn["parameters"]["required"] == ["location"]


def test_parse_args_resolves_endpoint(mod):
    args = mod.parse_args(["--base-url", "http://127.0.0.1:8000", "chat", "--user-message", "hi"])
    assert args.cmd == "chat"
    assert args.oai_base_url == "http://127.0.0.1:8000/oai/api/v1"
    assert args.user_message == "hi"
