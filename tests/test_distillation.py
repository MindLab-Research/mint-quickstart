from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "recipes" / "distillation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_distillation_script", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports() -> None:
    """Test that the distillation module imports without errors."""
    module = _load_module()
    assert module is not None


def test_preflight_connection_callable() -> None:
    """Test that preflight_connection is defined and callable."""
    module = _load_module()
    assert hasattr(module, "preflight_connection")
    assert callable(module.preflight_connection)


def test_build_prompts_callable() -> None:
    """Test that build_prompts is defined and callable."""
    module = _load_module()
    assert hasattr(module, "build_prompts")
    assert callable(module.build_prompts)


def test_build_prompts_returns_list() -> None:
    """Test that build_prompts returns a non-empty list."""
    module = _load_module()
    prompts = module.build_prompts()
    assert isinstance(prompts, list)
    assert len(prompts) > 0
