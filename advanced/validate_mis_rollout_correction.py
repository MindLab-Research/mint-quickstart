#!/usr/bin/env python3
"""Validate session-level MIS rollout correction end-to-end on a remote MinT server."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests


def _load_env_file(path: Path) -> None:
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


REPO_ROOT = Path(__file__).resolve().parents[1]
_load_env_file(REPO_ROOT / ".env")


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _fail(stage: str, msg: str) -> int:
    print(f"FAIL [{stage}]: {msg}", file=sys.stderr)
    return 1


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-dict json: {type(data)}")
    return data


def _poll_future(
    base_url: str,
    headers: dict[str, str],
    request_id: str,
    *,
    timeout_s: float,
    interval_s: float,
    stage: str,
) -> dict[str, Any]:
    url = f"{base_url}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    print(f"[{stage}] polling request_id={request_id} timeout_s={timeout_s}")
    while time.time() < deadline:
        resp = requests.post(url, headers=headers, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"POST {url} returned non-dict json: {type(data)}")
            return data
        if resp.status_code == 408:
            time.sleep(interval_s)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    raise TimeoutError(f"POST {url} timed out after {timeout_s}s (request_id={request_id})")


def _delete_model(base_url: str, headers: dict[str, str], model_id: str) -> None:
    url = f"{base_url}/api/v1/models/{model_id}"
    try:
        resp = requests.delete(url, headers=headers, timeout=60)
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(f"DELETE {url} returned {resp.status_code}: {resp.text[:400]!r}")
        print(f"[cleanup] deleted model_id={model_id}")
    except Exception as exc:
        print(f"[cleanup] warning: model_id={model_id} delete failed: {exc}", file=sys.stderr)


def _build_rl_datum() -> dict[str, Any]:
    tokens = [10, 11, 12, 13, 14, 15]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    weights = [0.0, 1.0, 1.0, 1.0, 1.0]
    old_logprobs = [0.0, 0.0, 0.0, 0.0, 0.0]
    advantages = [0.0, 1.0, -1.5, 0.7, -0.2]

    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
            "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
            "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate session-level MIS rollout correction on MinT")
    parser.add_argument(
        "--base-url",
        default=_first_env("MINT_BASE_URL", "TINKER_BASE_URL", default="https://mint.macaron.im"),
    )
    parser.add_argument("--api-key", default=_first_env("MINT_API_KEY", "TINKER_API_KEY"))
    parser.add_argument(
        "--base-model",
        default=_first_env("MINT_BASE_MODEL", "TINKER_MODEL", default="Qwen/Qwen3-0.6B"),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=int(_first_env("MINT_LORA_RANK", "TINKER_LORA_RANK", default="8")),
    )
    parser.add_argument(
        "--mis-threshold",
        type=float,
        default=float(_first_env("MINT_MIS_THRESHOLD", "TINKER_MIS_THRESHOLD", default="1.1")),
        help="rollout_correction_config.rollout_is_threshold",
    )
    parser.add_argument(
        "--create-timeout-s",
        type=float,
        default=float(_first_env("MINT_CREATE_MODEL_TIMEOUT_S", "TINKER_CREATE_MODEL_TIMEOUT_S", default="3600")),
    )
    parser.add_argument(
        "--forward-backward-timeout-s",
        type=float,
        default=float(
            _first_env("MINT_FORWARD_BACKWARD_TIMEOUT_S", "TINKER_FORWARD_BACKWARD_TIMEOUT_S", default="1800")
        ),
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=float(_first_env("MINT_POLL_INTERVAL_S", "TINKER_POLL_INTERVAL_S", default="2.0")),
    )
    parser.add_argument("--skip-cleanup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = str(args.base_url).rstrip("/")
    api_key = str(args.api_key)
    headers = _headers(api_key)
    model_id: str | None = None

    if not api_key:
        return _fail("config", "missing API key; set MINT_API_KEY or TINKER_API_KEY")

    print(
        "[config] "
        f"base_url={base_url} base_model={args.base_model} "
        f"lora_rank={args.lora_rank} mis_threshold={args.mis_threshold}"
    )

    try:
        session_id = f"validate-mis-{uuid.uuid4().hex[:8]}"
        print(f"[create_model] submitted session_id={session_id}")
        created = _post_json(
            f"{base_url}/api/v1/create_model",
            headers,
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": args.base_model,
                "lora_config": {"rank": int(args.lora_rank)},
                "rollout_correction_config": {
                    "rollout_is": "sequence",
                    "rollout_is_threshold": float(args.mis_threshold),
                    "rollout_rs": "seq_sum_k1",
                    "rollout_rs_threshold": "0.5_2.0",
                    "bypass_mode": True,
                    "loss_type": "reinforce",
                },
            },
            timeout_s=60.0,
        )
        if "request_id" in created:
            created = _poll_future(
                base_url,
                headers,
                str(created["request_id"]),
                timeout_s=float(args.create_timeout_s),
                interval_s=float(args.poll_interval_s),
                stage="create_model",
            )
        if "error" in created:
            return _fail("create_model", repr(created.get("error")))
        model_id = created.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            return _fail("create_model", f"missing/invalid model_id: {created!r}")
        print(f"[create_model] resolved model_id={model_id}")

        print(f"[forward_backward] submitted model_id={model_id} loss_fn=importance_sampling")
        datum = _build_rl_datum()
        fb = _post_json(
            f"{base_url}/api/v1/forward_backward",
            headers,
            {
                "model_id": model_id,
                "forward_backward_input": {
                    "data": [datum],
                    "loss_fn": "importance_sampling",
                },
            },
            timeout_s=60.0,
        )
        if "request_id" in fb:
            fb = _poll_future(
                base_url,
                headers,
                str(fb["request_id"]),
                timeout_s=float(args.forward_backward_timeout_s),
                interval_s=float(args.poll_interval_s),
                stage="forward_backward",
            )
        if "error" in fb:
            return _fail("forward_backward", repr(fb.get("error")))

        outputs = fb.get("loss_fn_outputs")
        if not isinstance(outputs, list) or not outputs:
            return _fail("malformed_response", f"missing loss_fn_outputs: {fb!r}")
        print(f"[forward_backward] resolved outputs={len(outputs)}")
        print("PASS: MIS rollout_correction request succeeded and response was valid")
        return 0
    except TimeoutError as exc:
        stage = "forward_backward" if model_id else "create_model"
        return _fail(stage, str(exc))
    except requests.RequestException as exc:
        stage = "forward_backward" if model_id else "create_model"
        return _fail(stage, f"request submission failure: {exc}")
    except Exception as exc:
        stage = "forward_backward" if model_id else "create_model"
        return _fail(stage, str(exc))
    finally:
        if model_id and not args.skip_cleanup:
            _delete_model(base_url, headers, model_id)


if __name__ == "__main__":
    raise SystemExit(main())
