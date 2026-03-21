from __future__ import annotations

import importlib.util
import ssl
import urllib.error
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "advanced"
    / "checkpoint.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_checkpoint_script", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_checkpoint_path_accepts_tinker_scheme() -> None:
    module = _load_module()

    assert module._parse_checkpoint_path("tinker://run-1/weights/demo") == (
        "tinker",
        "run-1",
        "weights",
        "demo",
    )


def test_normalize_mint_path_canonicalizes_tinker_scheme() -> None:
    module = _load_module()

    assert module._normalize_mint_path("tinker://run-1/sampler_weights/demo") == [
        "mint://run-1/sampler_weights/demo"
    ]


def test_normalize_mint_path_expands_legacy_name_without_path_type() -> None:
    module = _load_module()

    assert module._normalize_mint_path("tinker://run-1/demo", checkpoint_type="auto") == [
        "mint://run-1/weights/demo",
        "mint://run-1/sampler_weights/demo",
    ]


def test_maybe_retry_over_http_for_wrong_tls_version() -> None:
    module = _load_module()
    err = urllib.error.URLError(ssl.SSLError("WRONG_VERSION_NUMBER"))

    assert module._maybe_retry_over_http("https://10.0.0.1:18000/archive", err) == (
        "http://10.0.0.1:18000/archive"
    )


def test_looks_like_remote_checkpoint_path_supports_ckpt_and_uri_schemes() -> None:
    module = _load_module()

    assert module._looks_like_remote_checkpoint_path("ckpt_abc123")
    assert module._looks_like_remote_checkpoint_path("mint://run-1/weights/demo")
    assert module._looks_like_remote_checkpoint_path("tinker://run-1/weights/demo")
