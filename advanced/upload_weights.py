#!/usr/bin/env python3
"""Minimal MinT checkpoint upload demo.

Run:
  MINT_API_KEY=... MINT_UPLOAD_ARCHIVE=/path/to/ckpt.tar.gz \
    python demo/upload_weights/upload_weights_demo.py

Optional:
  MINT_BASE_URL=...            # default https://mint.macaron.im
  MINT_UPLOAD_TIMEOUT_S=300    # upload timeout seconds

Notes:
- The server validates the archive; many deployments require optimizer state.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


def load_env_file(path: Path) -> None:
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


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    print(f"Missing {name}. Set {name}=... and retry.")
    sys.exit(1)


def upload_checkpoint_archive(base_url: str, api_key: str, archive_path: str) -> str:
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    upload_url = f"{base_url.rstrip('/')}/api/v1/checkpoints/upload"
    parsed = urlparse(upload_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base URL: {base_url}")

    boundary = f"----mint-upload-{os.urandom(8).hex()}"
    filename = os.path.basename(archive_path)
    preamble = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/gzip\r\n\r\n"
    ).encode("utf-8")
    postamble = f"\r\n--{boundary}--\r\n".encode("utf-8")
    content_length = len(preamble) + os.path.getsize(archive_path) + len(postamble)

    timeout_s = float(os.environ.get("MINT_UPLOAD_TIMEOUT_S", "300"))
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout_s)
    try:
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        conn.putrequest("POST", path)
        conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        conn.putheader("Content-Length", str(content_length))
        conn.putheader("X-API-Key", api_key)
        conn.putheader("User-Agent", "mint-upload-demo/1.0")
        conn.endheaders()

        conn.send(preamble)
        with open(archive_path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                conn.send(chunk)
        conn.send(postamble)

        resp = conn.getresponse()
        body = resp.read()
        if resp.status >= 400:
            raise RuntimeError(f"Upload failed ({resp.status}): {body.decode('utf-8', 'ignore')}")
        payload = json.loads(body.decode("utf-8"))
        checkpoint_path = payload.get("path") or payload.get("checkpoint_id")
        if not checkpoint_path:
            raise RuntimeError(f"Upload response missing path: {payload!r}")
        return str(checkpoint_path)
    finally:
        conn.close()


def main() -> None:
    demo_dir = Path(__file__).resolve().parents[1]
    load_env_file(demo_dir / ".env")

    archive_path = os.environ.get("MINT_UPLOAD_ARCHIVE")
    if not archive_path and len(sys.argv) > 1:
        archive_path = sys.argv[1]
    if not archive_path:
        print("Missing MINT_UPLOAD_ARCHIVE or CLI arg. Provide a .tar.gz checkpoint archive.")
        sys.exit(1)

    api_key = os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY")
    if not api_key:
        require_env("MINT_API_KEY")

    base_url = os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")
    if not base_url:
        base_url = "https://mint.macaron.im"

    uploaded_path = upload_checkpoint_archive(base_url, api_key, archive_path)
    print(f"Uploaded: {uploaded_path}")


if __name__ == "__main__":
    main()
