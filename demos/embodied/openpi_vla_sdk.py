#!/usr/bin/env python3
"""OpenPI FAST SDK example for MinT.

This is the recommended SDK path for OpenPI / VLA usage in MinT:
- use top-level `mint` for the Tinker-compatible client surface
- use `mint.mint as mintx` for the MinT-only OpenPI extension surface

Run:
  python demos/embodied/openpi_vla_sdk.py
"""

from __future__ import annotations

import base64
import os
from typing import Any

import mint
import mint.mint as mintx

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional helper for repo ergonomics
    load_dotenv = None

ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jwvoAAAAASUVORK5CYII="
)
DEFAULT_SAMPLER_NAME = "mint-openpi-sdk-example"


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def build_demo_batch() -> list[mint.types.Datum]:
    return [
        mintx.build_openpi_fast_datum(
            prefix_tokens=[11, 12, 13],
            image_bytes_by_camera={camera_name: ONE_PIXEL_PNG for camera_name in mintx.CAMERA_LAYOUT},
            state=[0.1] * 7,
            target_tokens=[21, 22],
            weights=[1.0, 1.0],
            token_ar_mask=[1, 1],
        )
    ]


def run_example() -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv()

    service_client = mint.ServiceClient()
    try:
        training_client = mintx.create_openpi_training_client(
            service_client,
            base_model=os.getenv("MINT_OPENPI_SDK_BASE_MODEL", mintx.OPENPI_FAST_MODEL),
            rank=_env_int("MINT_OPENPI_SDK_LORA_RANK", mintx.OPENPI_FAST_LORA_RANK),
            create_timeout_seconds=_env_float("MINT_OPENPI_SDK_CREATE_TIMEOUT_SECONDS", 1200.0),
            user_metadata={"example": "demos/embodied/openpi_vla_sdk.py"},
        )
        info = training_client.get_info()
        step_result = training_client.train_step(
            build_demo_batch(),
            "cross_entropy",
            adam_params=mint.types.AdamParams(learning_rate=_env_float("MINT_OPENPI_SDK_LR", 0.003)),
        ).result(timeout=_env_float("MINT_OPENPI_SDK_STEP_TIMEOUT_SECONDS", 1200.0))
        sampler = training_client.save_weights_for_sampler(
            os.getenv("MINT_OPENPI_SDK_SAMPLER_NAME", DEFAULT_SAMPLER_NAME),
            ttl_seconds=_env_int("MINT_OPENPI_SDK_TTL_SECONDS", 3600),
        ).result(timeout=_env_float("MINT_OPENPI_SDK_SAVE_TIMEOUT_SECONDS", 1200.0))
        return {
            "model_id": training_client.model_id,
            "model_name": info.model_name,
            "lora_rank": info.lora_rank,
            "train_step_metrics": step_result.metrics,
            "sampler_path": sampler.path,
        }
    finally:
        service_client.holder.close()


if __name__ == "__main__":
    from pprint import pprint

    pprint(run_example())
