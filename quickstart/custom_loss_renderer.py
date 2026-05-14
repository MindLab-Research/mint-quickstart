#!/usr/bin/env python3
"""MinT custom loss quickstart (renderer variant).

Same DPO preference training as custom_loss.py, but uses the renderer
API (build_supervised_example + datum_from_model_input_weights) instead
of manual token encoding for chosen/rejected datum construction.

Run:
  python quickstart/custom_loss_renderer.py

All training runs against a remote MinT server.
This script does NOT start any backend services locally.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

import mint
import mint.recipe as recipe
import torch
import torch.nn.functional as F
import tinker
from mint import types

MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
LOSS_STEPS = int(os.environ.get("MINT_CUSTOM_LOSS_STEPS", "8"))
LOSS_LR = float(os.environ.get("MINT_CUSTOM_LOSS_LR", "5e-5"))
MAX_TOK = int(os.environ.get("MINT_CUSTOM_LOSS_MAX_TOKENS", "96"))


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


PREFERENCE_PAIRS = [
    PreferencePair(
        prompt="Explain why regular backups matter.",
        chosen=(
            "Regular backups reduce recovery time after mistakes, hardware failures, "
            "or ransomware. They also let you restore a known-good state instead of "
            "rebuilding systems from scratch."
        ),
        rejected="Backups are good.",
    ),
    PreferencePair(
        prompt="Give three concrete tips for better sleep.",
        chosen=(
            "Keep a consistent bedtime, avoid caffeine late in the day, and dim "
            "screens for at least 30 minutes before sleep."
        ),
        rejected="Try sleeping more.",
    ),
    PreferencePair(
        prompt="What should a code review comment optimize for?",
        chosen=(
            "A good review comment should be specific, actionable, and tied to a "
            "user-visible risk such as correctness, maintainability, or safety."
        ),
        rejected="It should just say this looks wrong.",
    ),
    PreferencePair(
        prompt="How would you describe an API timeout to a user?",
        chosen=(
            "State which request timed out, what the user can retry, and whether the "
            "server may still be processing the request."
        ),
        rejected="Tell them it failed.",
    ),
]


def _configured_base_url() -> str:
    base_url = os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")
    if not base_url:
        return "https://mint.macaron.xin/"
    return base_url


def _require_api_key() -> str:
    api_key = (os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or "").strip()
    if api_key:
        return api_key
    raise RuntimeError(
        "MINT_API_KEY not found. Set `MINT_API_KEY=sk-your-api-key-here` in the shell "
        f"or add it to `{REPO_ROOT / '.env'}` before running custom_loss_renderer.py."
    )


def _status_code_from_error(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def preflight_connection(service_client: mint.ServiceClient):
    base_url = _configured_base_url()
    try:
        return service_client.get_server_capabilities()
    except tinker.APITimeoutError as exc:
        raise RuntimeError(
            f"Auth preflight timed out while contacting {base_url}. "
            "Check `MINT_BASE_URL` and retry."
        ) from exc
    except tinker.APIConnectionError as exc:
        raise RuntimeError(
            f"Auth preflight could not reach {base_url}. "
            "Check `MINT_BASE_URL`, network access, and server status."
        ) from exc
    except tinker.APIStatusError as exc:
        status_code = _status_code_from_error(exc)
        if status_code in {401, 403}:
            raise RuntimeError(
                "Auth preflight was rejected by the MinT server "
                f"(HTTP {status_code}). Check that `MINT_API_KEY` is valid for {base_url}."
            ) from exc
        raise RuntimeError(
            "Auth preflight failed with an unexpected MinT server response "
            f"(HTTP {status_code or 'unknown'}) from {base_url}."
        ) from exc


def flatten_preference_pairs_renderer(
    pairs: list[PreferencePair], renderer, max_length: int
) -> list[types.Datum]:
    data: list[types.Datum] = []
    for pair in pairs:
        chosen_messages = [
            {"role": "user", "content": pair.prompt},
            {"role": "assistant", "content": pair.chosen},
        ]
        rejected_messages = [
            {"role": "user", "content": pair.prompt},
            {"role": "assistant", "content": pair.rejected},
        ]

        chosen_mi, chosen_w = renderer.build_supervised_example(chosen_messages)
        chosen_datum = recipe.datum_from_model_input_weights(
            chosen_mi, chosen_w, max_length=max_length
        )

        rejected_mi, rejected_w = renderer.build_supervised_example(rejected_messages)
        rejected_datum = recipe.datum_from_model_input_weights(
            rejected_mi, rejected_w, max_length=max_length
        )

        data.append(chosen_datum)
        data.append(rejected_datum)
    return data


def _to_float_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "to_torch"):
        tensor = value.to_torch()
    elif hasattr(value, "tolist"):
        tensor = torch.tensor(value.tolist(), dtype=torch.float32)
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    return tensor.flatten().float()


def sequence_logprob(logprobs: torch.Tensor, weights: Any) -> torch.Tensor:
    logprob_tensor = logprobs.flatten().float()
    weight_tensor = _to_float_tensor(weights)
    if logprob_tensor.shape != weight_tensor.shape:
        raise ValueError(
            "logprobs and weights must have the same shape, "
            f"got {tuple(logprob_tensor.shape)} and {tuple(weight_tensor.shape)}"
        )
    return torch.dot(logprob_tensor, weight_tensor)


def pairwise_preference_loss(
    data: list[types.Datum], logprobs_list: list[torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(data) % 2 != 0:
        raise ValueError(
            "pairwise_preference_loss expects an even number of datums ordered as "
            "(chosen, rejected) pairs."
        )

    chosen_scores: list[torch.Tensor] = []
    rejected_scores: list[torch.Tensor] = []

    for chosen_datum, rejected_datum, chosen_logprobs, rejected_logprobs in zip(
        data[::2], data[1::2], logprobs_list[::2], logprobs_list[1::2]
    ):
        chosen_scores.append(
            sequence_logprob(chosen_logprobs, chosen_datum.loss_fn_inputs["weights"])
        )
        rejected_scores.append(
            sequence_logprob(rejected_logprobs, rejected_datum.loss_fn_inputs["weights"])
        )

    chosen_scores_tensor = torch.stack(chosen_scores)
    rejected_scores_tensor = torch.stack(rejected_scores)
    margins = chosen_scores_tensor - rejected_scores_tensor
    loss = -F.logsigmoid(margins).mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "pair_accuracy": float((margins > 0).float().mean().detach().cpu()),
        "mean_margin": float(margins.mean().detach().cpu()),
        "mean_chosen_score": float(chosen_scores_tensor.mean().detach().cpu()),
        "mean_rejected_score": float(rejected_scores_tensor.mean().detach().cpu()),
    }
    return loss, metrics


def main() -> int:
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        preflight_connection(service_client)

        training_client = service_client.create_lora_training_client(
            base_model=MODEL, rank=RANK, train_mlp=True, train_attn=True, train_unembed=True,
        )
        tokenizer = training_client.get_tokenizer()
        print(f"Model: {MODEL}, Vocab: {tokenizer.vocab_size:,}")

        renderer_name = recipe.get_recommended_renderer_name(MODEL)
        renderer = recipe.renderers.get_renderer(renderer_name, tokenizer)
        print(f"Renderer: {renderer_name}")

        data = flatten_preference_pairs_renderer(PREFERENCE_PAIRS, renderer, MAX_TOK)
        print(f"Preference pairs: {len(data) // 2}\n")

        for step in range(1, LOSS_STEPS + 1):
            result = training_client.forward_backward_custom(
                data, pairwise_preference_loss
            ).result()
            metrics = result.metrics or {}
            training_client.optim_step(types.AdamParams(learning_rate=LOSS_LR)).result()
            print(
                f"Step {step}: loss={metrics.get('loss', float('nan')):.4f}, "
                f"pair_accuracy={metrics.get('pair_accuracy', 0.0):.1%}, "
                f"mean_margin={metrics.get('mean_margin', float('nan')):.4f}"
            )

        final_checkpoint = training_client.save_weights_for_sampler(
            name="custom-loss-renderer-final"
        ).result()
        print(f"\nSaved: {final_checkpoint.path}")

        sampling_client = service_client.create_sampling_client(
            model_path=final_checkpoint.path, base_model=MODEL,
        )
        prompt_messages = [{"role": "user", "content": PREFERENCE_PAIRS[0].prompt}]
        prompt = renderer.build_generation_prompt(prompt_messages)
        stop_sequences = renderer.get_stop_sequences()
        stop_token_ids = [t for t in stop_sequences if isinstance(t, int)]
        stop_strings = [t for t in stop_sequences if isinstance(t, str)]
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": MAX_TOK,
            "temperature": 0.0,
            "stop_token_ids": stop_token_ids,
        }
        if stop_strings:
            sampling_kwargs["stop"] = stop_strings
        preview = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(**sampling_kwargs),
        ).result()
        message, success = renderer.parse_response(list(preview.sequences[0].tokens))
        if not success:
            print("Warning: parse_response failed, output may be incomplete", file=sys.stderr)
        print(f"Preview prompt: {PREFERENCE_PAIRS[0].prompt}")
        print(f"Preview response: {message.get('content', '')}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
