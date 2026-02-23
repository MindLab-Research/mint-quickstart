from __future__ import annotations

import http.client
import json
import os
import re
import sys
from dataclasses import dataclass
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


DEMO_DIR = Path(__file__).resolve().parents[1]
load_env_file(DEMO_DIR / ".env")

REPO_ROOT = Path(__file__).resolve().parents[2]
for src_dir in ("mindlab-toolkit-alpha/src", "mindlab-toolkit/src"):
    mint_src = REPO_ROOT / src_dir
    if mint_src.exists() and str(mint_src) not in sys.path:
        sys.path.insert(0, str(mint_src))
        break

import mint
from mint import types

PROMPT = "Bug: add() returns wrong sum.\nFix:\n"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    print(f"Missing {name}. Set {name}=... and retry.")
    sys.exit(1)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class DemoConfig:
    base_url: str | None
    model: str
    rank: int
    lr: float
    group: int
    max_tokens: int
    temperature: float
    resume_path: str | None
    upload_archive: str | None
    upload_only: bool
    upload_timeout_s: float
    total_steps: int
    checkpoint_every: int


def parse_config_from_env() -> DemoConfig:
    require_env("MINT_API_KEY")

    cfg = DemoConfig(
        base_url=os.environ.get("MINT_BASE_URL"),
        model=os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B"),
        rank=int(os.environ.get("MINT_LORA_RANK", "16")),
        lr=float(os.environ.get("MINT_RL_LR", "5e-5")),
        group=int(os.environ.get("MINT_GROUP_SIZE", "4")),
        max_tokens=int(os.environ.get("MINT_MAX_TOKENS", "256")),
        temperature=float(os.environ.get("MINT_TEMPERATURE", "1.0")),
        resume_path=os.environ.get("MINT_RESUME_PATH"),
        upload_archive=os.environ.get("MINT_UPLOAD_ARCHIVE"),
        upload_only=parse_bool(os.environ.get("MINT_UPLOAD_ONLY", "")),
        upload_timeout_s=float(os.environ.get("MINT_UPLOAD_TIMEOUT_S", "300")),
        total_steps=int(os.environ.get("MINT_TOTAL_STEPS", "100")),
        checkpoint_every=int(os.environ.get("MINT_CHECKPOINT_EVERY_STEPS", "20")),
    )

    if cfg.total_steps <= 0:
        raise ValueError("MINT_TOTAL_STEPS must be > 0.")
    if cfg.checkpoint_every <= 0:
        raise ValueError("MINT_CHECKPOINT_EVERY_STEPS must be > 0.")
    if cfg.group <= 0:
        raise ValueError("MINT_GROUP_SIZE must be > 0.")
    if cfg.max_tokens <= 0:
        raise ValueError("MINT_MAX_TOKENS must be > 0.")

    return cfg


def extract_code(response: str) -> str | None:
    match = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match[-1].strip()
    if "def " in response:
        return response[response.find("def "):].strip()
    return None


def grade(response: str) -> float:
    """Run generated add() against two test cases. Returns 1.0 or 0.0."""
    code = extract_code(response)
    if not code:
        return 0.0
    try:
        ns = {}
        exec(code, ns)
        if ns.get("add") is None:
            return 0.0
        if ns["add"](2, 3) != 5:
            return 0.0
        if ns["add"](-1, 1) != 0:
            return 0.0
        return 1.0
    except Exception:
        return 0.0


def upload_checkpoint_archive(
    base_url: str, api_key: str, archive_path: str, timeout_s: float
) -> str:
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

    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
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
            raise RuntimeError(
                f"Upload failed ({resp.status}): {body.decode('utf-8', 'ignore')}"
            )
        payload = json.loads(body.decode("utf-8"))
        checkpoint_path = payload.get("path") or payload.get("checkpoint_id")
        if not checkpoint_path:
            raise RuntimeError(f"Upload response missing path: {payload!r}")
        return str(checkpoint_path)
    finally:
        conn.close()


def rl_train_step(
    training_client: mint.TrainingClient, cfg: DemoConfig, step: int
) -> dict:
    """Sample from current policy, score with reward, train with GRPO."""
    tokenizer = training_client.get_tokenizer()
    prompt_tokens = tokenizer.encode(PROMPT)

    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"rl-resume-step-{step}"
    )
    sample_result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=cfg.group,
        sampling_params=types.SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop_token_ids=[tokenizer.eos_token_id],
        ),
    ).result()

    group_rewards: list[float] = []
    group_responses: list[list[int]] = []
    group_logprobs: list[list[float]] = []

    for seq in sample_result.sequences:
        response = tokenizer.decode(seq.tokens)
        reward = grade(response)
        group_rewards.append(reward)
        group_responses.append(list(seq.tokens))
        group_logprobs.append(list(seq.logprobs or [0.0] * len(seq.tokens)))

    mean_reward = sum(group_rewards) / len(group_rewards)
    advantages = [reward - mean_reward for reward in group_rewards]
    if all(adv == 0.0 for adv in advantages):
        return {"rewards": group_rewards, "num_datums": 0}

    datums = []
    prefix_len = len(prompt_tokens) - 1
    for resp_tokens, logprobs, adv in zip(group_responses, group_logprobs, advantages):
        if not resp_tokens:
            continue
        full_tokens = prompt_tokens + resp_tokens
        response_len = len(resp_tokens)
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": full_tokens[1:],
                    "weights": [0.0] * prefix_len + [1.0] * response_len,
                    "logprobs": [0.0] * prefix_len + logprobs,
                    "advantages": [0.0] * prefix_len + [adv] * response_len,
                },
            )
        )

    if datums:
        training_client.forward_backward(datums, loss_fn="importance_sampling").result()
        training_client.optim_step(types.AdamParams(learning_rate=cfg.lr)).result()

    return {"rewards": group_rewards, "num_datums": len(datums)}


def create_service_client(cfg: DemoConfig) -> mint.ServiceClient:
    if cfg.base_url:
        return mint.ServiceClient(base_url=cfg.base_url)
    return mint.ServiceClient()


def create_training_client(
    service_client: mint.ServiceClient, cfg: DemoConfig
) -> mint.TrainingClient:
    return service_client.create_lora_training_client(
        base_model=cfg.model,
        rank=cfg.rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )


def parse_step_from_checkpoint_path(path: str) -> int | None:
    matches = re.findall(r"step[-_]?(\d+)", path)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def save_training_state(
    training_client: mint.TrainingClient, step: int, reason: str
) -> str:
    checkpoint = training_client.save_state(
        name=f"rl-resume-{reason}-step-{step:06d}"
    ).result()
    checkpoint_path = str(checkpoint.path)
    print(f"[checkpoint] step={step} reason={reason} path={checkpoint_path}")
    return checkpoint_path


def load_resume_state(training_client: mint.TrainingClient, resume_path: str) -> int:
    print(f"[resume] source={resume_path}")
    try:
        training_client.load_state_with_optimizer(resume_path).result()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MINT_RESUME_PATH={resume_path}. "
            "Set a valid checkpoint path and retry."
        ) from exc

    inferred_step = parse_step_from_checkpoint_path(resume_path)
    if inferred_step is None:
        print(
            "[resume] state loaded, but step could not be inferred from path. "
            "Starting from step=0."
        )
        return 0

    print(f"[resume] loaded successfully; inferred_start_step={inferred_step}")
    return inferred_step


def resolve_resume_path(cfg: DemoConfig) -> str | None:
    resume_path = cfg.resume_path
    if cfg.upload_archive:
        api_key = os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing API key for upload.")
        upload_base_url = cfg.base_url or os.environ.get(
            "TINKER_BASE_URL", "https://mint.macaron.im"
        )
        uploaded_path = upload_checkpoint_archive(
            upload_base_url, api_key, cfg.upload_archive, cfg.upload_timeout_s
        )
        print(f"[upload] uploaded={uploaded_path}")
        if not resume_path:
            resume_path = uploaded_path

    return resume_path


def main() -> None:
    cfg = parse_config_from_env()
    service_client = create_service_client(cfg)
    resume_path = resolve_resume_path(cfg)
    if cfg.upload_archive and cfg.upload_only:
        return

    training_client = create_training_client(service_client, cfg)
    start_step = 0
    if resume_path:
        start_step = load_resume_state(training_client, resume_path)

    resume_source = resume_path or "none"
    print(
        f"[run] model={cfg.model} total_steps={cfg.total_steps} "
        f"checkpoint_every={cfg.checkpoint_every} resume_source={resume_source} "
        f"start_step={start_step}"
    )

    last_completed_step = start_step
    latest_checkpoint = resume_path

    try:
        for step in range(start_step + 1, cfg.total_steps + 1):
            stats = rl_train_step(training_client, cfg, step)
            accuracy = (
                sum(1 for reward in stats["rewards"] if reward > 0)
                / len(stats["rewards"])
                if stats["rewards"]
                else 0.0
            )
            print(
                f"[train] step={step}/{cfg.total_steps} accuracy={accuracy:.1%} "
                f"datums={stats['num_datums']}"
            )
            last_completed_step = step

            if step % cfg.checkpoint_every == 0:
                latest_checkpoint = save_training_state(
                    training_client, step, "periodic"
                )
    except KeyboardInterrupt:
        print("[interrupt] received keyboard interrupt; saving recovery checkpoint.")
        latest_checkpoint = save_training_state(
            training_client, last_completed_step, "interrupt"
        )
        print(
            f"[interrupt] resume_source={resume_source} "
            f"latest_checkpoint={latest_checkpoint}"
        )
        raise SystemExit(130)

    latest_checkpoint = save_training_state(
        training_client, last_completed_step, "final"
    )
    print(
        f"[done] final_step={last_completed_step} resume_source={resume_source} "
        f"latest_checkpoint={latest_checkpoint}"
    )


if __name__ == "__main__":
    main()
