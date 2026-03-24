from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "quickstart" / "custom_reward.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_custom_reward", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_prediction_returns_first_integer() -> None:
    module = _load_module()

    assert module.extract_prediction("Answer: 314 with confidence") == 314
    assert module.extract_prediction("No numeric answer") is None


def test_compute_reward_breakdown_gives_full_credit_for_exact_match() -> None:
    module = _load_module()

    breakdown = module.compute_reward_breakdown("The answer is 144.", 144)

    assert breakdown.total == 1.0
    assert breakdown.format_reward == 0.2
    assert breakdown.distance_reward == 0.5
    assert breakdown.exact_bonus == 0.3


def test_compute_reward_breakdown_gives_partial_credit_for_close_answer() -> None:
    module = _load_module()

    breakdown = module.compute_reward_breakdown("I think it is 140.", 144)

    assert 0.0 < breakdown.total < 1.0
    assert breakdown.format_reward == 0.2
    assert 0.0 < breakdown.distance_reward < 0.5
    assert breakdown.exact_bonus == 0.0


def test_summarize_breakdowns_reports_rates() -> None:
    module = _load_module()

    breakdowns = [
        module.RewardBreakdown(1.0, 0.2, 0.5, 0.3),
        module.RewardBreakdown(0.4, 0.2, 0.2, 0.0),
        module.RewardBreakdown(0.0, 0.0, 0.0, 0.0),
    ]

    summary = module.summarize_breakdowns(breakdowns)

    assert summary["avg_reward"] == (1.0 + 0.4 + 0.0) / 3
    assert summary["exact_rate"] == 1 / 3
    assert summary["format_rate"] == 2 / 3
