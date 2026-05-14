from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "recipes" / "dpo_native.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_dpo_native_script", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports() -> None:
    """Test that the dpo_native module imports without errors."""
    module = _load_module()
    assert module is not None


def test_preflight_connection_callable() -> None:
    """Test that preflight_connection is defined and callable."""
    module = _load_module()
    assert hasattr(module, "preflight_connection")
    assert callable(module.preflight_connection)


def test_build_preference_pairs_callable() -> None:
    """Test that build_preference_pairs is defined and callable."""
    module = _load_module()
    assert hasattr(module, "build_preference_pairs")
    assert callable(module.build_preference_pairs)


def test_build_preference_pairs_returns_list() -> None:
    """Test that build_preference_pairs returns a non-empty list."""
    module = _load_module()
    pairs = module.build_preference_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) > 0
