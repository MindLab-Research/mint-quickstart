#!/usr/bin/env python3
"""Queue status polling — read pending queue fields on 408 via raw response.

Demonstrates the backpressure API: submit a sample request, then poll
with `futures.with_raw_response.retrieve()` to read queue depth,
position, ETA, and status from both the response body and headers.

Env:
  MINT_API_KEY or TINKER_API_KEY     (optional if server auth disabled)
  MINT_BASE_URL or TINKER_BASE_URL   (optional, default mint)
  MINT_BASE_MODEL                    (optional, default Qwen/Qwen3-0.6B)

Run:
  python advanced/queue_status.py
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
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

import mint as _mint  # noqa: E402  # triggers env sync (MINT_* -> TINKER_*)
import tinker  # noqa: E402
from tinker._client import AsyncTinker  # noqa: E402
from tinker import types  # noqa: E402


# ---------------------------------------------------------------------------
# Header parsing helpers
# ---------------------------------------------------------------------------
def _int_header(headers, name: str) -> int:
    value = headers.get(name)
    if value is None:
        raise RuntimeError(f"missing header {name}")
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError(f"header {name} is not int: {value!r}") from exc


def _maybe_int_header(headers, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError(f"header {name} is not int: {value!r}") from exc


def _maybe_float_header(headers, name: str) -> float | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:
        raise RuntimeError(f"header {name} is not float: {value!r}") from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    base_model = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
    session_id = f"queue_status_{int(time.time())}"

    async with AsyncTinker() as client:
        # 1. Create sampling session
        create_req = types.CreateSamplingSessionRequest(
            session_id=session_id,
            sampling_session_seq_id=0,
            base_model=base_model,
        )
        create_resp = await client.service.create_sampling_session(request=create_req)
        sampling_session_id = create_resp.sampling_session_id
        print(
            "sampling_session_created",
            f"sampling_session_id={sampling_session_id}",
            flush=True,
        )

        # 2. Submit sample request with backpressure header
        sample_req = types.SampleRequest(
            sampling_session_id=sampling_session_id,
            seq_id=1,
            num_samples=1,
            prompt=types.ModelInput.from_ints([1] * 256),
            sampling_params=types.SamplingParams(
                max_tokens=128,
                temperature=0.7,
                top_k=-1,
                top_p=1.0,
            ),
        )
        future = await client.sampling.asample(
            request=sample_req,
            extra_headers={"X-Tinker-Sampling-Backpressure": "1"},
        )

        request_id = future.request_id
        print("request_submitted", f"request_id={request_id}", flush=True)

        # 3. Poll until result or unexpected error
        while True:
            try:
                raw = await client.futures.with_raw_response.retrieve(
                    request=types.FutureRetrieveRequest(
                        request_id=request_id,
                    )
                )
                status_code = raw.status_code
                body = raw.json()
                if inspect.isawaitable(body):
                    body = await body
                headers = raw.headers
            except tinker.APIStatusError as e:
                status_code = e.status_code
                headers = e.response.headers if e.response is not None else {}
                if isinstance(e.body, dict):
                    body = e.body
                else:
                    try:
                        body = e.response.json()
                    except Exception:
                        body = {"error": str(e)}

            # 4. Handle result
            if status_code == 200:
                if isinstance(body, dict):
                    print(
                        "result_received",
                        f"request_id={request_id}",
                        f"keys={list(body)[:8]}",
                        flush=True,
                    )
                else:
                    print("result_received", f"request_id={request_id}", flush=True)
                return

            if status_code != 408:
                raise RuntimeError(
                    f"unexpected status {status_code} body={str(body)[:500]!r}"
                )

            # 5. Parse queue status from 408 response
            retry_after_s = _int_header(headers, "Retry-After")
            queue_depth = _maybe_int_header(headers, "X-Queue-Depth")
            queue_position = _maybe_int_header(headers, "X-Queue-Position")
            queue_eta_s = _maybe_float_header(headers, "X-Queue-ETA-S")
            queue_status = headers.get("X-Queue-Status")

            print(
                "queue_fields",
                f"request_id={body.get('request_id')!r}",
                f"type={body.get('type')!r}",
                f"status={body.get('status')!r}",
                f"queue_state={body.get('queue_state')!r}",
                f"queue_state_reason={body.get('queue_state_reason')!r}",
                f"queue_depth={body.get('queue_depth')!r}",
                f"queue_position={body.get('queue_position')!r}",
                f"estimated_wait_s={body.get('estimated_wait_s')!r}",
                f"progress={body.get('progress')!r}",
                f"retry_after_s={body.get('retry_after_s')!r}",
                f"x_queue_depth={queue_depth!r}",
                f"x_queue_position={queue_position!r}",
                f"x_queue_status={queue_status!r}",
                f"x_queue_eta_s={queue_eta_s!r}",
                f"retry_after={retry_after_s!r}",
                flush=True,
            )

            body_request_id = body.get("request_id")
            if (
                isinstance(body_request_id, str)
                and body_request_id
                and body_request_id != request_id
            ):
                raise RuntimeError(
                    f"request_id mismatch: body={body_request_id!r} "
                    f"expected {request_id!r}"
                )

            await asyncio.sleep(max(1, int(retry_after_s)))


if __name__ == "__main__":
    asyncio.run(main())
