from __future__ import annotations

import os
import random
import re
import sys
import http.client
import json
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

# Allow running against a local checkout without installing `mindlab-toolkit`.
REPO_ROOT = Path(__file__).resolve().parents[2]
MINT_SRC = REPO_ROOT / "mindlab-toolkit" / "src"
if MINT_SRC.exists() and str(MINT_SRC) not in sys.path:
    sys.path.insert(0, str(MINT_SRC))

import mint
from mint import types


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    print(f"Missing {name}. Set {name}=... and retry.")
    sys.exit(1)


require_env("MINT_API_KEY")
BASE_URL = os.environ.get("MINT_BASE_URL")
MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
LR = float(os.environ.get("MINT_RL_LR", "5e-5"))
GROUP = int(os.environ.get("MINT_GROUP_SIZE", "4"))
MAX_TOK = int(os.environ.get("MINT_MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("MINT_TEMPERATURE", "1.0"))
RESUME_PATH = os.environ.get("MINT_RESUME_PATH")
UPLOAD_ARCHIVE = os.environ.get("MINT_UPLOAD_ARCHIVE")
UPLOAD_ONLY = os.environ.get("MINT_UPLOAD_ONLY", "").strip().lower() in {"1", "true", "yes", "y", "on"}
UPLOAD_TIMEOUT_S = float(os.environ.get("MINT_UPLOAD_TIMEOUT_S", "300"))
TOTAL_STEPS = int(os.environ.get("MINT_TOTAL_STEPS", "100"))
INTERRUPT_COUNT = int(os.environ.get("MINT_INTERRUPT_COUNT", "5"))
INTERRUPT_SEED = os.environ.get("MINT_INTERRUPT_SEED")

PROMPT = "Bug: add() returns wrong sum.\nFix:\n"


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

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, parsed.port, timeout=UPLOAD_TIMEOUT_S)
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


def rl_train_step(training_client: mint.TrainingClient, step: int) -> dict:
    """Sample from current policy, score with reward, train with GRPO.

    Returns dict with keys: rewards (list[float]), num_datums (int).
    """
    tokenizer = training_client.get_tokenizer()
    prompt_tokens = tokenizer.encode(PROMPT)

    # Snapshot weights for on-policy sampling
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"rft-{step}"
    )

    sample_result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=GROUP,
        sampling_params=types.SamplingParams(
            max_tokens=MAX_TOK,
            temperature=TEMPERATURE,
            stop_token_ids=[tokenizer.eos_token_id],
        ),
    ).result()

    # Score each sequence
    # seq.tokens: list[int], seq.logprobs: list[float]
    group_rewards = []
    group_responses = []  # list[list[int]]
    group_logprobs = []   # list[list[float]]

    for seq in sample_result.sequences:
        response = tokenizer.decode(seq.tokens)
        reward = grade(response)
        group_rewards.append(reward)
        group_responses.append(list(seq.tokens))
        group_logprobs.append(list(seq.logprobs or [0.0] * len(seq.tokens)))

    # Group advantage (GRPO): reward - mean_reward
    mean_reward = sum(group_rewards) / len(group_rewards)
    advantages = [r - mean_reward for r in group_rewards]

    # Skip if all same reward (no gradient signal)
    if all(adv == 0.0 for adv in advantages):
        return {"rewards": group_rewards, "num_datums": 0}

    # Build training datums
    # Each datum shape:
    #   input_tokens:  [prompt[:-1] + response]  len = P-1 + R
    #   target_tokens: [prompt[1:] + response]    len = P-1 + R
    #   weights:       [0]*prefix + [1]*R
    #   logprobs:      [0]*prefix + seq_logprobs
    #   advantages:    [0]*prefix + [adv]*R
    datums = []
    prefix_len = len(prompt_tokens) - 1

    for resp_tokens, logprobs, adv in zip(
        group_responses, group_logprobs, advantages
    ):
        if not resp_tokens:
            continue
        full_tokens = prompt_tokens + resp_tokens
        input_tokens = full_tokens[:-1]    # shape: (P + R - 1,)
        target_tokens = full_tokens[1:]    # shape: (P + R - 1,)
        r = len(resp_tokens)
        weights = [0.0] * prefix_len + [1.0] * r
        full_logprobs = [0.0] * prefix_len + logprobs
        full_advantages = [0.0] * prefix_len + [adv] * r

        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "weights": weights,
                    "logprobs": full_logprobs,
                    "advantages": full_advantages,
                },
            )
        )

    if datums:
        training_client.forward_backward(
            datums, loss_fn="importance_sampling"
        ).result()
        training_client.optim_step(types.AdamParams(learning_rate=LR)).result()

    return {"rewards": group_rewards, "num_datums": len(datums)}


def create_training_client(
    service_client: mint.ServiceClient,
) -> mint.TrainingClient:
    return service_client.create_lora_training_client(
        base_model=MODEL,
        rank=RANK,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )


service_client = (
    mint.ServiceClient(base_url=BASE_URL) if BASE_URL else mint.ServiceClient()
)
if UPLOAD_ARCHIVE:
    api_key = os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key for upload.")
    upload_base_url = BASE_URL or os.environ.get("TINKER_BASE_URL", "https://mint.macaron.im")
    uploaded = upload_checkpoint_archive(upload_base_url, api_key, UPLOAD_ARCHIVE)
    print(f"Uploaded: {uploaded}")
    if not RESUME_PATH:
        RESUME_PATH = uploaded
    if UPLOAD_ONLY:
        sys.exit(0)

rng = random.Random(INTERRUPT_SEED) if INTERRUPT_SEED is not None else random.Random()
interrupt_steps: set[int] = set()
if TOTAL_STEPS > 1 and INTERRUPT_COUNT > 0:
    max_interrupts = min(INTERRUPT_COUNT, TOTAL_STEPS - 1)
    interrupt_steps = set(rng.sample(range(1, TOTAL_STEPS), max_interrupts))
if interrupt_steps:
    print(f"Planned interruptions at steps: {sorted(interrupt_steps)}")

resume_path = RESUME_PATH
training_client: mint.TrainingClient | None = None
started = False

for step in range(1, TOTAL_STEPS + 1):
    if training_client is None:
        training_client = create_training_client(service_client)
        if resume_path:
            print(f"Resuming from {resume_path}")
            training_client.load_state(resume_path).result()
        if not started:
            print("It works: training client ready.")
            started = True

    stats = rl_train_step(training_client, step)
    accuracy = sum(1 for r in stats["rewards"] if r > 0) / len(stats["rewards"])
    print(
        f"Step {step}/{TOTAL_STEPS}: accuracy={accuracy:.1%}, "
        f"datums={stats['num_datums']}"
    )

    if step in interrupt_steps:
        ckpt = training_client.save_state(name=f"rft-resume-step{step}").result()
        print(f"Saved: {ckpt.path} (interrupt at step {step})")
        resume_path = ckpt.path
        training_client = None

if training_client is not None:
    ckpt = training_client.save_state(name=f"rft-resume-step{TOTAL_STEPS}").result()
    print(f"Saved: {ckpt.path}")
