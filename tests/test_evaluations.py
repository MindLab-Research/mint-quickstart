from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "concepts" / "evaluations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_evaluations_script", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports() -> None:
    """Test that the evaluations module imports without errors."""
    module = _load_module()
    assert module is not None


def test_preflight_connection_callable() -> None:
    """Test that preflight_connection is defined and callable."""
    module = _load_module()
    assert hasattr(module, "preflight_connection")
    assert callable(module.preflight_connection)


def test_demonstrate_evaluations_callable() -> None:
    """Test that demonstrate_evaluations is defined and callable."""
    module = _load_module()
    assert hasattr(module, "demonstrate_evaluations")
    assert callable(module.demonstrate_evaluations)
