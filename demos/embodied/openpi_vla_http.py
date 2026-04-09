#!/usr/bin/env python3
"""Embodied-1 OpenPI FAST HTTP example for MinT.

This is a direct HTTP example for the current OpenPI FAST VLA wire format:
create a training session, create a LoRA model, submit one train_step with
three camera images + state + target action tokens, export sampler weights,
and delete the model.

Run:
  pip install httpx python-dotenv
  python demos/embodied/openpi_vla_http.py
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional helper for repo ergonomics
    load_dotenv = None

OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"
OPENPI_FAST_LORA_RANK = 16
CAMERA_LAYOUT = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jwvoAAAAASUVORK5CYII="
)
DEFAULT_BASE_URL = "https://mint.macaron.xin"
DEFAULT_SAMPLER_PATH = "mint-openpi-vla-http-example"


def _tensor(data: Sequence[int] | Sequence[float], *, dtype: str) -> dict[str, Any]:
    return {"data": list(data), "shape": [len(data)], "dtype": dtype}


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def build_create_session_request() -> dict[str, Any]:
    return {
        "tags": ["openpi-vla-http-example"],
        "user_metadata": {"example": "demos/embodied/openpi_vla_http.py"},
        "sdk_version": "mint-openpi-vla-http-example",
        "type": "create_session",
    }


def build_create_model_request(
    *,
    session_id: str,
    model_seq_id: int,
    base_model: str = OPENPI_FAST_MODEL,
    rank: int = OPENPI_FAST_LORA_RANK,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "model_seq_id": model_seq_id,
        "base_model": base_model,
        "lora_config": {
            "rank": rank,
            "train_attn": True,
            "train_mlp": True,
            "train_unembed": True,
        },
        "type": "create_model",
    }


def build_openpi_fast_datum_payload(
    *,
    prefix_tokens: Sequence[int],
    image_bytes_by_camera: Mapping[str, bytes],
    state: Sequence[float],
    target_tokens: Sequence[int],
    weights: Sequence[float],
    token_ar_mask: Sequence[int],
    logprobs: Sequence[float] | None = None,
    advantages: Sequence[float] | None = None,
) -> dict[str, Any]:
    if set(image_bytes_by_camera) != set(CAMERA_LAYOUT):
        raise ValueError(f"image_bytes_by_camera must contain exactly {CAMERA_LAYOUT}")
    if len(target_tokens) != len(weights) or len(target_tokens) != len(token_ar_mask):
        raise ValueError("target_tokens, weights, and token_ar_mask must share one length")
    if (logprobs is None) != (advantages is None):
        raise ValueError("logprobs and advantages must be provided together")
    if logprobs is not None and len(logprobs) != len(target_tokens):
        raise ValueError("logprobs must share one length with target_tokens")
    if advantages is not None and len(advantages) != len(target_tokens):
        raise ValueError("advantages must share one length with target_tokens")

    # OpenPI FAST expects three fixed-order camera images, then prompt tokens.
    loss_fn_inputs: dict[str, Any] = {
        "state": _tensor(state, dtype="float32"),
        "target_tokens": _tensor(target_tokens, dtype="int64"),
        "weights": _tensor(weights, dtype="float32"),
        "token_ar_mask": _tensor(token_ar_mask, dtype="int32"),
    }
    if logprobs is not None and advantages is not None:
        loss_fn_inputs["logprobs"] = _tensor(logprobs, dtype="float32")
        loss_fn_inputs["advantages"] = _tensor(advantages, dtype="float32")

    return {
        "model_input": {
            "chunks": [
                *[
                    {
                        "data": base64.b64encode(image_bytes_by_camera[camera_name]).decode("utf-8"),
                        "format": "png",
                        "expected_tokens": 256,
                        "type": "image",
                    }
                    for camera_name in CAMERA_LAYOUT
                ],
                {"tokens": list(prefix_tokens), "type": "encoded_text"},
            ]
        },
        "loss_fn_inputs": loss_fn_inputs,
    }


def build_train_step_request(
    *,
    model_id: str,
    prefix_tokens: Sequence[int],
    image_bytes_by_camera: Mapping[str, bytes],
    state: Sequence[float],
    target_tokens: Sequence[int],
    weights: Sequence[float],
    token_ar_mask: Sequence[int],
    adam_params: dict[str, float] | None = None,
    loss_fn: str = "cross_entropy",
    loss_fn_config: dict[str, float] | None = None,
    logprobs: Sequence[float] | None = None,
    advantages: Sequence[float] | None = None,
    seq_id: int | None = 1,
) -> dict[str, Any]:
    return {
        "forward_backward_input": {
            "data": [
                build_openpi_fast_datum_payload(
                    prefix_tokens=prefix_tokens,
                    image_bytes_by_camera=image_bytes_by_camera,
                    state=state,
                    target_tokens=target_tokens,
                    weights=weights,
                    token_ar_mask=token_ar_mask,
                    logprobs=logprobs,
                    advantages=advantages,
                )
            ],
            "loss_fn": loss_fn,
            "loss_fn_config": dict(loss_fn_config or {}),
        },
        "adam_params": dict(adam_params or {"learning_rate": 0.003}),
        "model_id": model_id,
        "seq_id": seq_id,
        "type": "train_step",
    }


def build_get_info_request(model_id: str) -> dict[str, Any]:
    return {"model_id": model_id, "type": "get_info"}


def build_save_weights_for_sampler_request(
    model_id: str,
    *,
    path: str | None = None,
    ttl_seconds: int | None = 3600,
    sampling_session_seq_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_id": model_id,
        "ttl_seconds": ttl_seconds,
        "type": "save_weights_for_sampler",
    }
    if path is not None:
        payload["path"] = path
    if sampling_session_seq_id is not None:
        payload["sampling_session_seq_id"] = sampling_session_seq_id
    return payload


def _poll_delay_seconds(payload: dict[str, Any], headers: Mapping[str, str]) -> float:
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    queue_state = payload.get("queue_state")
    if queue_state == "active":
        return 0.3
    if queue_state == "paused_capacity":
        return 1.0
    if queue_state == "paused_rate_limit":
        return 2.0
    return 1.0


def poll_future(
    client: httpx.Client,
    *,
    request_id: str,
    timeout_seconds: float | None = None,
    sleep=time.sleep,
) -> dict[str, Any]:
    timeout_seconds = (
        _env_float("MINT_OPENPI_HTTP_FUTURE_TIMEOUT_SECONDS", 1200.0)
        if timeout_seconds is None
        else timeout_seconds
    )
    deadline = time.monotonic() + timeout_seconds
    last_queue_state: str | None = None
    while True:
        response = client.post("/api/v1/retrieve_future", json={"request_id": request_id})
        if response.status_code == 408:
            payload = response.json()
            delay_seconds = _poll_delay_seconds(payload, response.headers)
            queue_state = payload.get("queue_state")
            last_queue_state = str(queue_state) if isinstance(queue_state, str) else None
            if time.monotonic() + delay_seconds > deadline:
                raise TimeoutError(
                    f"retrieve_future timed out after {timeout_seconds}s for request_id={request_id}; "
                    f"last_queue_state={last_queue_state!r}"
                )
            sleep(delay_seconds)
            continue
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return payload


def _auth_headers() -> dict[str, str]:
    api_key = os.getenv("MINT_API_KEY") or os.getenv("TINKER_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def run_example() -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv()

    base_url = os.getenv("MINT_BASE_URL") or os.getenv("TINKER_BASE_URL") or DEFAULT_BASE_URL
    base_model = os.getenv("MINT_OPENPI_HTTP_BASE_MODEL", OPENPI_FAST_MODEL)
    sampler_path = os.getenv("MINT_OPENPI_HTTP_SAMPLER_PATH", DEFAULT_SAMPLER_PATH)
    learning_rate = _env_float("MINT_OPENPI_HTTP_LR", 0.003)
    client_timeout_seconds = _env_float("MINT_OPENPI_HTTP_CLIENT_TIMEOUT_SECONDS", 120.0)
    lora_rank = _env_int("MINT_OPENPI_HTTP_LORA_RANK", OPENPI_FAST_LORA_RANK)

    model_id: str | None = None
    delete_model: dict[str, Any] | None = None
    with httpx.Client(base_url=base_url, headers=_auth_headers(), timeout=client_timeout_seconds) as client:
        session = client.post("/api/v1/create_session", json=build_create_session_request())
        session.raise_for_status()
        session_id = session.json()["session_id"]

        create_future = client.post(
            "/api/v1/create_model",
            json=build_create_model_request(
                session_id=session_id,
                model_seq_id=0,
                base_model=base_model,
                rank=lora_rank,
            ),
        )
        create_future.raise_for_status()
        create_result = poll_future(client, request_id=create_future.json()["request_id"])
        model_id = create_result["model_id"]

        try:
            model_info = client.post("/api/v1/get_info", json=build_get_info_request(model_id))
            model_info.raise_for_status()

            step_future = client.post(
                "/api/v1/train_step",
                json=build_train_step_request(
                    model_id=model_id,
                    prefix_tokens=[11, 12, 13],
                    image_bytes_by_camera={camera_name: ONE_PIXEL_PNG for camera_name in CAMERA_LAYOUT},
                    state=[0.1] * 7,
                    target_tokens=[21, 22],
                    weights=[1.0, 1.0],
                    token_ar_mask=[1, 1],
                    adam_params={"learning_rate": learning_rate},
                ),
            )
            step_future.raise_for_status()
            step_result = poll_future(client, request_id=step_future.json()["request_id"])

            sampler_future = client.post(
                "/api/v1/save_weights_for_sampler",
                json=build_save_weights_for_sampler_request(
                    model_id,
                    path=sampler_path,
                    ttl_seconds=3600,
                ),
            )
            sampler_future.raise_for_status()
            sampler_result = poll_future(client, request_id=sampler_future.json()["request_id"])

            models = client.get("/api/v1/models")
            models.raise_for_status()
            result = {
                "session_id": session_id,
                "model_id": model_id,
                "model_info": model_info.json(),
                "train_step": step_result,
                "sampler": sampler_result,
                "models": models.json(),
            }
        finally:
            if model_id is not None:
                delete_response = client.delete(f"/api/v1/models/{model_id}")
                delete_response.raise_for_status()
                delete_model = delete_response.json()

    result["delete_model"] = delete_model
    return result


if __name__ == "__main__":
    from pprint import pprint

    pprint(run_example())
