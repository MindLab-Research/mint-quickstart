from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "quickstart" / "custom_loss.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_custom_loss", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flatten_preference_pairs_preserves_chosen_rejected_order() -> None:
    module = _load_module()

    class _FakeTokenizer:
        eos_token_id = 99

        def encode(self, text: str, add_special_tokens: bool = True):
            base = [len(text) % 11 + 1]
            if not add_special_tokens:
                return base
            return [7] + base

    pairs = [
        module.PreferencePair(prompt="p1", chosen="chosen-1", rejected="rejected-1"),
        module.PreferencePair(prompt="p2", chosen="chosen-2", rejected="rejected-2"),
    ]

    data = module.flatten_preference_pairs(pairs, _FakeTokenizer())

    assert len(data) == 4
    assert data[0].loss_fn_inputs["weights"].data[-1] == 1.0
    assert data[1].loss_fn_inputs["weights"].data[-1] == 1.0
    assert data[2].loss_fn_inputs["weights"].data[-1] == 1.0
    assert data[3].loss_fn_inputs["weights"].data[-1] == 1.0


def test_sequence_logprob_only_counts_weighted_tokens() -> None:
    module = _load_module()

    score = module.sequence_logprob(
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        [0.0, 1.0, 1.0],
    )

    assert torch.isclose(score, torch.tensor(0.5))


def test_pairwise_preference_loss_reports_metrics() -> None:
    module = _load_module()

    data = [
        SimpleNamespace(loss_fn_inputs={"weights": [0.0, 1.0, 1.0]}),
        SimpleNamespace(loss_fn_inputs={"weights": [0.0, 1.0, 1.0]}),
    ]
    logprobs = [
        torch.tensor([0.0, 1.5, 1.0], dtype=torch.float32),
        torch.tensor([0.0, 0.1, 0.1], dtype=torch.float32),
    ]

    loss, metrics = module.pairwise_preference_loss(data, logprobs)

    assert loss.item() > 0.0
    assert metrics["pair_accuracy"] == 1.0
    assert metrics["mean_margin"] > 0.0


def test_pairwise_preference_loss_requires_even_number_of_datums() -> None:
    module = _load_module()

    data = [SimpleNamespace(loss_fn_inputs={"weights": [1.0]})]
    logprobs = [torch.tensor([0.1], dtype=torch.float32)]

    with pytest.raises(ValueError, match="even number of datums"):
        module.pairwise_preference_loss(data, logprobs)
