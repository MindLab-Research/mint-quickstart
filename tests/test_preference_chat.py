from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "rl"
    / "adapters"
    / "preference_chat.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_preference_chat", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coerce_chat_template_tokens_accepts_input_ids_mapping() -> None:
    module = _load_module()

    assert module._coerce_chat_template_tokens(
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    ) == [1, 2, 3]


def test_coerce_chat_template_tokens_flattens_batched_input_ids() -> None:
    module = _load_module()

    assert module._coerce_chat_template_tokens({"input_ids": [[4, 5, 6]]}) == [4, 5, 6]
