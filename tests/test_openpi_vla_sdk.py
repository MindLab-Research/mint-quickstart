from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TOOLKIT_SRC = WORKSPACE_ROOT / "mindlab-toolkit-alpha" / "src"
MODULE_PATH = Path(__file__).resolve().parents[1] / "demos" / "embodied" / "openpi_vla_sdk.py"


def _clear_mint_modules() -> None:
    for name in list(sys.modules):
        if name == "mint" or name.startswith("mint."):
            sys.modules.pop(name, None)


def _load_module():
    _clear_mint_modules()
    sys.path.insert(0, str(TOOLKIT_SRC))
    try:
        spec = importlib.util.spec_from_file_location("mint_openpi_vla_sdk", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == str(TOOLKIT_SRC):
            sys.path.pop(0)


def test_build_demo_batch_uses_mintx_openpi_shape() -> None:
    module = _load_module()

    batch = module.build_demo_batch()

    assert len(batch) == 1
    datum = batch[0]
    chunks = datum.model_input.chunks
    assert [type(chunk).__name__ for chunk in chunks] == ["ImageChunk", "ImageChunk", "ImageChunk", "EncodedTextChunk"]
    assert chunks[-1].tokens == [11, 12, 13]
    assert datum.loss_fn_inputs["state"].shape == [7]
    assert datum.loss_fn_inputs["token_ar_mask"].dtype == "int64"


def test_run_example_uses_mintx_client_and_returns_sdk_facing_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    captured: dict[str, object] = {}

    class _FakeFuture:
        def __init__(self, payload):
            self._payload = payload

        def result(self, timeout: float):
            captured.setdefault("timeouts", []).append(timeout)
            return self._payload

    class _FakeOpenPIClient:
        model_id = "model-123"

        def get_info(self):
            return SimpleNamespace(model_name="openpi/pi0-fast-libero-low-mem-finetune", lora_rank=16)

        def train_step(self, data, loss_fn, adam_params=None):
            captured["train_data"] = data
            captured["train_loss_fn"] = loss_fn
            captured["train_lr"] = adam_params.learning_rate
            return _FakeFuture(SimpleNamespace(metrics={"loss": 1.25}))

        def save_weights_for_sampler(self, name: str, ttl_seconds: int | None = None):
            captured["sampler_name"] = name
            captured["sampler_ttl"] = ttl_seconds
            return _FakeFuture(SimpleNamespace(path="mint://sampler/path"))

    class _FakeServiceClient:
        def __init__(self):
            self.holder = SimpleNamespace(close=lambda: captured.setdefault("closed", True))

    monkeypatch.setattr(module.mint, "ServiceClient", _FakeServiceClient)
    def _fake_create_openpi_training_client(service_client, **kwargs):
        captured["create_kwargs"] = kwargs
        return _FakeOpenPIClient()

    monkeypatch.setattr(module.mintx, "create_openpi_training_client", _fake_create_openpi_training_client)
    monkeypatch.setenv("MINT_OPENPI_SDK_LR", "0.005")
    monkeypatch.setenv("MINT_OPENPI_SDK_TTL_SECONDS", "1800")
    monkeypatch.setenv("MINT_OPENPI_SDK_CREATE_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("MINT_OPENPI_SDK_STEP_TIMEOUT_SECONDS", "22")
    monkeypatch.setenv("MINT_OPENPI_SDK_SAVE_TIMEOUT_SECONDS", "33")

    result = module.run_example()

    assert captured["create_kwargs"] == {
        "base_model": module.mintx.OPENPI_FAST_MODEL,
        "rank": module.mintx.OPENPI_FAST_LORA_RANK,
        "create_timeout_seconds": 11.0,
        "user_metadata": {"example": "demos/embodied/openpi_vla_sdk.py"},
    }
    assert captured["train_loss_fn"] == "cross_entropy"
    assert captured["train_lr"] == 0.005
    assert captured["sampler_name"] == module.DEFAULT_SAMPLER_NAME
    assert captured["sampler_ttl"] == 1800
    assert captured["timeouts"] == [22.0, 33.0]
    assert captured["closed"] is True
    assert result == {
        "model_id": "model-123",
        "model_name": "openpi/pi0-fast-libero-low-mem-finetune",
        "lora_rank": 16,
        "train_step_metrics": {"loss": 1.25},
        "sampler_path": "mint://sampler/path",
    }
