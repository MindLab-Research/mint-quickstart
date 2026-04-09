from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "demos" / "embodied" / "openpi_vla_http.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_openpi_vla_http", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, status_code: int, payload, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400 and self.status_code != 408:
            raise RuntimeError(f"unexpected status {self.status_code}")


def test_build_openpi_fast_datum_payload_keeps_camera_order_and_shapes() -> None:
    module = _load_module()

    payload = module.build_openpi_fast_datum_payload(
        prefix_tokens=[11, 12, 13],
        image_bytes_by_camera={
            "base_0_rgb": b"base-image",
            "left_wrist_0_rgb": b"left-image",
            "right_wrist_0_rgb": b"right-image",
        },
        state=[0.1] * 7,
        target_tokens=[21, 22],
        weights=[1.0, 1.0],
        token_ar_mask=[1, 1],
    )

    chunks = payload["model_input"]["chunks"]
    assert [chunk["type"] for chunk in chunks] == ["image", "image", "image", "encoded_text"]
    assert [base64.b64decode(chunk["data"]) for chunk in chunks[:3]] == [
        b"base-image",
        b"left-image",
        b"right-image",
    ]
    assert payload["loss_fn_inputs"]["state"] == {
        "data": [0.1] * 7,
        "shape": [7],
        "dtype": "float32",
    }
    assert payload["loss_fn_inputs"]["target_tokens"]["shape"] == [2]


def test_build_openpi_fast_datum_payload_rejects_mismatched_lengths() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="share one length"):
        module.build_openpi_fast_datum_payload(
            prefix_tokens=[11, 12],
            image_bytes_by_camera={camera_name: b"img" for camera_name in module.CAMERA_LAYOUT},
            state=[0.1] * 7,
            target_tokens=[21, 22],
            weights=[1.0],
            token_ar_mask=[1, 1],
        )


def test_poll_future_retries_408_until_success() -> None:
    module = _load_module()
    sleep_calls: list[float] = []
    responses = [
        _FakeResponse(408, {"queue_state": "active"}, {"Retry-After": "0.5"}),
        _FakeResponse(408, {"queue_state": "paused_capacity"}),
        _FakeResponse(200, {"result": "ok"}),
    ]

    class _FakeClient:
        def post(self, path: str, json):
            assert path == "/api/v1/retrieve_future"
            assert json == {"request_id": "req-1"}
            return responses.pop(0)

    result = module.poll_future(
        _FakeClient(),
        request_id="req-1",
        timeout_seconds=10.0,
        sleep=sleep_calls.append,
    )

    assert result == {"result": "ok"}
    assert sleep_calls == [0.5, 1.0]


def test_poll_future_reports_last_queue_state_on_timeout() -> None:
    module = _load_module()

    class _FakeClient:
        def post(self, path: str, json):
            return _FakeResponse(408, {"queue_state": "paused_rate_limit"})

    with pytest.raises(TimeoutError, match="paused_rate_limit"):
        module.poll_future(
            _FakeClient(),
            request_id="req-timeout",
            timeout_seconds=0.1,
            sleep=lambda _: None,
        )
