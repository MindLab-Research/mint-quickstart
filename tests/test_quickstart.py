from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "quickstart" / "quickstart.py"


def _load_quickstart_module():
    spec = importlib.util.spec_from_file_location("mint_quickstart_script", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_api_key_errors_with_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_quickstart_module()
    monkeypatch.delenv("MINT_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=r"MINT_API_KEY not found"):
        module._require_api_key()


def test_preflight_connection_returns_capabilities() -> None:
    module = _load_quickstart_module()
    capabilities = SimpleNamespace(supported_models=["a", "b", "c"])

    class _FakeServiceClient:
        def get_server_capabilities(self):
            return capabilities

    result = module.preflight_connection(_FakeServiceClient())

    assert result is capabilities
    assert module._supported_model_count(result) == 3


def test_preflight_connection_relabels_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_quickstart_module()

    class FakeTimeoutError(Exception):
        pass

    class FakeConnectionError(Exception):
        pass

    class FakeStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"http {status_code}")
            self.status_code = status_code

    monkeypatch.setattr(module.tinker, "APITimeoutError", FakeTimeoutError)
    monkeypatch.setattr(module.tinker, "APIConnectionError", FakeConnectionError)
    monkeypatch.setattr(module.tinker, "APIStatusError", FakeStatusError)
    monkeypatch.setenv("MINT_BASE_URL", "https://mint.macaron.xin/")

    class _FakeServiceClient:
        def get_server_capabilities(self):
            raise FakeStatusError(401)

    with pytest.raises(RuntimeError, match=r"Auth preflight was rejected"):
        module.preflight_connection(_FakeServiceClient())
