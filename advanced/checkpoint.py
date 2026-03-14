#!/usr/bin/env python3
"""Checkpoint operations for MinT: save, download, upload, resume.

Usage:
    python advanced/checkpoint.py save     --name my-ckpt
    python advanced/checkpoint.py download mint://run-id/weights/step-100 -o ./ckpts
    python advanced/checkpoint.py upload   ./ckpts/step-100.tar.gz
    python advanced/checkpoint.py resume   mint://run-id/weights/step-100 --with-optimizer --steps 3

All commands require MINT_API_KEY in the environment or a .env file one level up.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter, sleep
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Bootstrap: load .env, add local mindlab-toolkit to sys.path
# ---------------------------------------------------------------------------

def _load_env_file(path: Path) -> None:
    """Parse a .env file into os.environ (no overwrite)."""
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


DEMO_DIR = Path(__file__).resolve().parents[1]
_load_env_file(DEMO_DIR / ".env")

REPO_ROOT = Path(__file__).resolve().parents[2]
for _src_dir in ("mindlab-toolkit-alpha/src", "mindlab-toolkit/src"):
    _mint_src = REPO_ROOT / _src_dir
    if _mint_src.exists() and str(_mint_src) not in sys.path:
        sys.path.insert(0, str(_mint_src))
        break

import mint  # noqa: E402
from mint import types  # noqa: E402

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    print(f"Missing {name}. Set {name}=... and retry.")
    sys.exit(1)


def _api_key() -> str:
    return os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or ""


def _base_url() -> str | None:
    return os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")


def _service_client() -> mint.ServiceClient:
    base_url = _base_url()
    if base_url:
        return mint.ServiceClient(base_url=base_url)
    return mint.ServiceClient()


# ---------------------------------------------------------------------------
# SFT datum (shared by save + resume)
# ---------------------------------------------------------------------------

PROMPT = "Bug: add() returns wrong sum.\nFix:\n"
ANSWER = "def add(a, b):\n    return a + b\n"


def _build_sft_datum(training_client: mint.TrainingClient) -> types.Datum:
    """Build a single SFT datum from hardcoded prompt/answer.

    Returns: types.Datum with shape:
        input_tokens  = full_tokens[:-1]  (len = T-1)
        target_tokens = full_tokens[1:]   (len = T-1)
        weights       = [0]*prefix_len + [1]*(T-1-prefix_len)
    where T = len(tokenize(PROMPT + ANSWER)), prefix_len = len(tokenize(PROMPT)) - 1.
    """
    tokenizer = training_client.get_tokenizer()
    prompt_tokens = tokenizer.encode(PROMPT)       # shape: (P,)
    full_tokens = tokenizer.encode(PROMPT + ANSWER) # shape: (T,)
    assert len(full_tokens) >= 2, "Prompt+answer must tokenize to >= 2 tokens."

    input_tokens = full_tokens[:-1]   # shape: (T-1,)
    target_tokens = full_tokens[1:]   # shape: (T-1,)
    prefix_len = max(0, len(prompt_tokens) - 1)
    weights = [0.0] * prefix_len + [1.0] * (len(target_tokens) - prefix_len)
    # weights shape: (T-1,)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


# ---------------------------------------------------------------------------
# Upload (HTTP multipart, zero external dependencies)
# ---------------------------------------------------------------------------

def _upload_checkpoint_archive(
    base_url: str, api_key: str, archive_path: str, timeout_s: float = 300.0,
) -> str:
    """Upload a .tar.gz archive to the MinT server.

    POST /api/v1/checkpoints/upload with multipart/form-data.
    Returns: server-side checkpoint path (str).
    """
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
    file_size = os.path.getsize(archive_path)
    content_length = len(preamble) + file_size + len(postamble)

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
        conn.putheader("User-Agent", "mint-checkpoint/1.0")
        conn.endheaders()

        conn.send(preamble)
        with open(archive_path, "rb") as handle:
            sent = 0
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                conn.send(chunk)
                sent += len(chunk)
                pct = 100.0 * sent / max(file_size, 1)
                print(f"   uploading: {sent / (1024*1024):.1f} / {file_size / (1024*1024):.1f} MiB ({pct:.0f}%)", end="\r")
        print()
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


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _parse_mint_path(mint_path: str) -> tuple[str, str | None, str]:
    """Parse mint://run/[weights|sampler_weights]/name.

    Returns: (run_id, path_type_or_None, checkpoint_name).
    """
    prefix = "mint://"
    if not mint_path.startswith(prefix):
        raise ValueError(f"Invalid mint path '{mint_path}'. Expected prefix 'mint://'.")

    remainder = mint_path[len(prefix):].strip("/")
    parts = [p for p in remainder.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid mint path '{mint_path}'. "
            "Expected mint://<run>/<name> or mint://<run>/(weights|sampler_weights)/<name>."
        )

    run_id = parts[0]
    tail = parts[1:]
    if tail[0] in {"weights", "sampler_weights"}:
        if len(tail) < 2:
            raise ValueError(
                f"Invalid mint path '{mint_path}'. Missing checkpoint name after '{tail[0]}'."
            )
        checkpoint_name = "/".join(tail[1:])
        return run_id, tail[0], checkpoint_name

    checkpoint_name = "/".join(tail)
    return run_id, None, checkpoint_name


def _normalize_mint_path(mint_path: str, checkpoint_type: str = "auto") -> list[str]:
    """Normalize into canonical mint://<run>/(weights|sampler_weights)/<name> form.

    Returns: list of candidate paths (1 or 2 depending on checkpoint_type).
    """
    run_id, path_type, checkpoint_name = _parse_mint_path(mint_path)

    if path_type in {"weights", "sampler_weights"}:
        return [f"mint://{run_id}/{path_type}/{checkpoint_name}"]

    if checkpoint_type == "training":
        return [f"mint://{run_id}/weights/{checkpoint_name}"]
    if checkpoint_type == "sampler":
        return [f"mint://{run_id}/sampler_weights/{checkpoint_name}"]
    if checkpoint_type == "auto":
        return [
            f"mint://{run_id}/weights/{checkpoint_name}",
            f"mint://{run_id}/sampler_weights/{checkpoint_name}",
        ]
    raise ValueError(
        f"Invalid checkpoint_type '{checkpoint_type}'. Use sampler, training, or auto."
    )


def _mint_path_to_tinker_path(mint_path: str) -> str:
    """Convert canonical mint:// path to tinker:// path for older SDK."""
    run_id, path_type, checkpoint_name = _parse_mint_path(mint_path)
    if path_type not in {"weights", "sampler_weights"}:
        raise ValueError("Mint path must be canonical for tinker conversion.")
    return f"tinker://{run_id}/{path_type}/{checkpoint_name}"


def _get_archive_url(rest_client: Any, mint_path: str) -> str:
    """Resolve a mint:// path to a signed download URL.

    Tries get_checkpoint_archive_url_from_mint_path first,
    falls back to get_checkpoint_archive_url_from_tinker_path.
    Returns: signed URL string.
    """
    # Try mint-native endpoint first
    mint_fn = getattr(rest_client, "get_checkpoint_archive_url_from_mint_path", None)
    if callable(mint_fn):
        resp = mint_fn(mint_path).result()
        return resp.url

    # Fallback: convert to tinker:// path
    tinker_fn = getattr(rest_client, "get_checkpoint_archive_url_from_tinker_path", None)
    if callable(tinker_fn):
        tinker_path = _mint_path_to_tinker_path(mint_path)
        resp = tinker_fn(tinker_path).result()
        return resp.url

    raise AttributeError(
        "REST client has neither get_checkpoint_archive_url_from_mint_path "
        "nor get_checkpoint_archive_url_from_tinker_path."
    )


def _download_with_progress(*, url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download url to dest_path with progress reporting.

    Prints progress line every 1s: MiB done / MiB total (pct%) @ speed ETA.
    """
    def _format_eta(seconds: float | None) -> str:
        if seconds is None:
            return "?"
        seconds = max(0.0, seconds)
        if seconds < 60:
            return f"{seconds:0.0f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes < 60:
            return f"{minutes:02d}:{secs:02d}"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:d}:{minutes:02d}:{secs:02d}"

    req = urllib.request.Request(url, headers={"User-Agent": "mint-checkpoint/1.0"})
    start = perf_counter()
    last_print = start
    downloaded = 0
    total = None

    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out:
        try:
            total_hdr = resp.headers.get("Content-Length")
            if total_hdr:
                total = int(total_hdr)
        except Exception:
            total = None

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            now = perf_counter()
            if now - last_print >= 1.0:
                elapsed = max(now - start, 1e-6)
                mb_done = downloaded / (1024 * 1024)
                speed = mb_done / elapsed
                if total and total > 0:
                    mb_total = total / (1024 * 1024)
                    pct = 100.0 * downloaded / total
                    remaining = max(0, total - downloaded) / (1024 * 1024)
                    eta = remaining / max(speed, 1e-6)
                    print(
                        f"   {mb_done:,.1f} / {mb_total:,.1f} MiB ({pct:5.1f}%) "
                        f"@ {speed:,.1f} MiB/s ETA {_format_eta(eta)}",
                        end="\r",
                    )
                else:
                    print(f"   {mb_done:,.1f} MiB @ {speed:,.1f} MiB/s", end="\r")
                last_print = now

    # Final summary line
    end = perf_counter()
    elapsed = max(end - start, 1e-6)
    mb_done = downloaded / (1024 * 1024)
    speed = mb_done / elapsed
    if total and total > 0:
        mb_total = total / (1024 * 1024)
        print(f"   {mb_done:,.1f} / {mb_total:,.1f} MiB (100.0%) @ {speed:,.1f} MiB/s")
    else:
        print(f"   {mb_done:,.1f} MiB @ {speed:,.1f} MiB/s")


def _extract_archive(archive_path: Path, out_dir: Path) -> Path:
    """Extract .tar.gz with path-traversal safety check.

    Returns: extraction directory path.
    """
    archive_name = archive_path.name
    extract_name = archive_name[:-7] if archive_name.endswith(".tar.gz") else archive_path.stem
    extract_dir = out_dir / extract_name
    extract_dir.mkdir(parents=True, exist_ok=True)

    def _is_safe(m: tarfile.TarInfo) -> bool:
        p = Path(m.name)
        return not p.is_absolute() and ".." not in p.parts

    print(f"   extracting to: {extract_dir}")
    with tarfile.open(archive_path, "r:*") as tf:
        safe = [m for m in tf.getmembers() if _is_safe(m)]
        try:
            tf.extractall(path=extract_dir, members=safe, filter="data")
        except TypeError:
            # Python < 3.12 lacks filter= parameter
            tf.extractall(path=extract_dir, members=safe)

    print(f"   extracted {len(safe)} members to {extract_dir}")
    return extract_dir


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_save(args: argparse.Namespace) -> None:
    """Save: 1 SFT step, then save_state + save_weights_for_sampler."""
    _require_env("MINT_API_KEY")
    model = args.model
    rank = args.rank
    lr = args.lr
    name = args.name

    print(f"[save] model={model} rank={rank} lr={lr} name={name}")

    sc = _service_client()
    tc = sc.create_lora_training_client(
        base_model=model, rank=rank,
        train_mlp=True, train_attn=True, train_unembed=True,
    )

    # 1 SFT step
    datum = _build_sft_datum(tc)
    print("[save] running forward_backward (1 datum, cross_entropy)...")
    tc.forward_backward([datum], loss_fn="cross_entropy").result()
    tc.optim_step(types.AdamParams(learning_rate=lr)).result()
    print("[save] step done")

    # Save state (weights + optimizer, for resume)
    state_ckpt = tc.save_state(name=f"{name}-state").result()
    state_path = str(state_ckpt.path)
    print(f"[save] state (weights+optimizer): {state_path}")

    # Save weights for sampler (weights only, for inference)
    sampler_ckpt = tc.save_weights_for_sampler(name=f"{name}-sampler").result()
    sampler_path = str(sampler_ckpt.path)
    print(f"[save] sampler (weights only):    {sampler_path}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download a checkpoint archive from mint:// path."""
    _require_env("MINT_API_KEY")
    mint_path: str = args.mint_path
    out_dir = Path(args.output).resolve()
    extract = args.extract
    checkpoint_type: str = args.checkpoint_type
    max_wait: float = args.max_wait
    poll_interval: float = args.poll_interval

    print(f"[download] path={mint_path} type={checkpoint_type} out={out_dir}")

    sc = _service_client()
    rest_client = sc.create_rest_client()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _normalize_mint_path(mint_path, checkpoint_type)
    # candidates shape: list of 1 or 2 canonical mint:// paths

    for i, candidate in enumerate(candidates):
        is_last = i == len(candidates) - 1
        _, _, ckpt_name = _parse_mint_path(candidate)
        archive_filename = f"{Path(ckpt_name).name}.tar.gz"
        archive_path = out_dir / archive_filename

        print(f"[download] trying: {candidate}")

        start = perf_counter()
        while True:
            try:
                url = _get_archive_url(rest_client, candidate)
                print(f"   signed URL obtained, downloading...")
                _download_with_progress(url=url, dest_path=archive_path)
                print(f"[download] saved to: {archive_path}")

                if extract:
                    _extract_archive(archive_path, out_dir)
                return  # done

            except mint.ConflictError as e:
                elapsed = perf_counter() - start
                if elapsed >= max_wait:
                    print(f"[download] timed out after {elapsed:.0f}s waiting for archive: {e}")
                    sys.exit(1)
                wait = min(poll_interval, max_wait - elapsed)
                print(f"   archive creation in progress, waiting {wait:.0f}s ({elapsed:.0f}/{max_wait:.0f}s)...")
                sleep(wait)

            except (mint.BadRequestError, urllib.error.HTTPError) as e:
                print(f"[download] failed for {candidate}: {e}")
                if not is_last:
                    print("   trying next candidate...")
                    break
                sys.exit(1)

            except Exception as e:
                print(f"[download] error: {e}")
                sys.exit(1)

    print("[download] all candidates failed")
    sys.exit(1)


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload a local .tar.gz archive to the MinT server."""
    _require_env("MINT_API_KEY")
    archive_path: str = args.archive
    timeout = args.timeout

    api_key = _api_key()
    base_url = _base_url() or "https://mint.macaron.xin"

    print(f"[upload] archive={archive_path} server={base_url}")
    uploaded_path = _upload_checkpoint_archive(base_url, api_key, archive_path, timeout)
    print(f"[upload] uploaded: {uploaded_path}")


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume training from a checkpoint path.

    Two modes:
    - Without --with-optimizer: create_training_client_from_state(path).
      Auto-detects model/rank. Optimizer resets.
    - With --with-optimizer: create_lora_training_client() + load_state_with_optimizer(path).
      Preserves optimizer momentum. Requires MINT_BASE_MODEL + MINT_LORA_RANK.
    """
    _require_env("MINT_API_KEY")
    resume_path: str = args.path
    with_optimizer: bool = args.with_optimizer
    steps: int = args.steps
    lr: float = args.lr
    save_name: str = args.save_name

    print(f"[resume] path={resume_path} with_optimizer={with_optimizer} steps={steps}")

    sc = _service_client()

    if with_optimizer:
        # Need explicit model/rank to create training client first
        model = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
        rank = int(os.environ.get("MINT_LORA_RANK", "16"))
        print(f"[resume] creating training client: model={model} rank={rank}")
        tc = sc.create_lora_training_client(
            base_model=model, rank=rank,
            train_mlp=True, train_attn=True, train_unembed=True,
        )
        print(f"[resume] loading state with optimizer from {resume_path}...")
        tc.load_state_with_optimizer(resume_path).result()
    else:
        print(f"[resume] creating training client from state (optimizer resets)...")
        tc = sc.create_training_client_from_state(resume_path)

    print(f"[resume] loaded, running {steps} SFT step(s)...")

    for i in range(1, steps + 1):
        datum = _build_sft_datum(tc)
        tc.forward_backward([datum], loss_fn="cross_entropy").result()
        tc.optim_step(types.AdamParams(learning_rate=lr)).result()
        print(f"[resume] step {i}/{steps} done")

    ckpt = tc.save_state(name=save_name).result()
    print(f"[resume] saved: {ckpt.path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="checkpoint.py",
        description="MinT checkpoint operations: save, download, upload, resume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- save --
    p_save = sub.add_parser("save", help="Run 1 SFT step and save checkpoint")
    p_save.add_argument("--name", default="my-ckpt", help="Checkpoint name prefix (default: my-ckpt)")
    p_save.add_argument("--model", default=os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B"),
                         help="Base model (default: $MINT_BASE_MODEL or Qwen/Qwen3-0.6B)")
    p_save.add_argument("--rank", type=int, default=int(os.environ.get("MINT_LORA_RANK", "16")),
                         help="LoRA rank (default: $MINT_LORA_RANK or 16)")
    p_save.add_argument("--lr", type=float, default=float(os.environ.get("MINT_RL_LR", "5e-5")),
                         help="Learning rate (default: $MINT_RL_LR or 5e-5)")

    # -- download --
    p_dl = sub.add_parser("download", help="Download checkpoint from mint:// path")
    p_dl.add_argument("mint_path", help="mint://<run>/(weights|sampler_weights)/<name>")
    p_dl.add_argument("-o", "--output", default="./checkpoints",
                       help="Output directory (default: ./checkpoints)")
    p_dl.add_argument("--checkpoint-type", choices=["sampler", "training", "auto"], default="auto",
                       help="Type hint for legacy paths without weights/ prefix (default: auto)")
    p_dl.add_argument("--no-extract", dest="extract", action="store_false", default=True,
                       help="Skip tar extraction after download")
    p_dl.add_argument("--max-wait", type=float, default=600.0,
                       help="Max seconds to wait for archive creation (409) (default: 600)")
    p_dl.add_argument("--poll-interval", type=float, default=15.0,
                       help="Poll interval during archive creation wait (default: 15)")

    # -- upload --
    p_up = sub.add_parser("upload", help="Upload .tar.gz checkpoint to MinT server")
    p_up.add_argument("archive", help="Path to .tar.gz checkpoint archive")
    p_up.add_argument("--timeout", type=float, default=300.0,
                       help="Upload timeout in seconds (default: 300)")

    # -- resume --
    p_re = sub.add_parser("resume", help="Resume training from checkpoint")
    p_re.add_argument("path", help="Checkpoint path (server-side, e.g. ckpt_... or mint://...)")
    p_re.add_argument("--with-optimizer", action="store_true", default=False,
                       help="Preserve optimizer state (requires MINT_BASE_MODEL + MINT_LORA_RANK)")
    p_re.add_argument("--steps", type=int, default=3,
                       help="Number of SFT steps to run after resume (default: 3)")
    p_re.add_argument("--lr", type=float, default=float(os.environ.get("MINT_RL_LR", "5e-5")),
                       help="Learning rate (default: $MINT_RL_LR or 5e-5)")
    p_re.add_argument("--save-name", default="resumed-checkpoint",
                       help="Name for the checkpoint saved after training (default: resumed-checkpoint)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "save": cmd_save,
        "download": cmd_download,
        "upload": cmd_upload,
        "resume": cmd_resume,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
