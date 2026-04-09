#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_REPO = "MindLab-Research/mint-quickstart"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a GitHub issue with gh issue create.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Target GitHub repo (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Issue title.",
    )
    parser.add_argument(
        "--body-file",
        required=True,
        help="Path to the Markdown body file.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=[],
        help="Issue label. Repeat for multiple labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gh command instead of executing it.",
    )
    return parser


def build_gh_issue_command(
    *,
    repo: str,
    title: str,
    body_file: str | Path,
    labels: Sequence[str] | None = None,
) -> list[str]:
    body_path = Path(body_file).expanduser().resolve()
    command = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        title,
        "--body-file",
        str(body_path),
    ]
    for label in labels or []:
        cleaned = label.strip()
        if cleaned:
            command.extend(["--label", cleaned])
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    body_path = Path(args.body_file).expanduser().resolve()
    if not body_path.exists():
        sys.stderr.write(f"Body file not found: {body_path}\n")
        return 1

    command = build_gh_issue_command(
        repo=args.repo,
        title=args.title,
        body_file=body_path,
        labels=args.labels,
    )

    if args.dry_run:
        sys.stdout.write(shlex.join(command) + "\n")
        return 0

    gh_path = shutil.which("gh")
    if gh_path is None:
        sys.stderr.write("GitHub CLI `gh` is not installed or not on PATH.\n")
        return 1

    auth_status = subprocess.run(
        [gh_path, "auth", "status"],
        capture_output=True,
        text=True,
        check=False,
    )
    if auth_status.returncode != 0:
        stderr = auth_status.stderr.strip() or auth_status.stdout.strip() or "GitHub CLI auth check failed."
        sys.stderr.write(stderr + "\n")
        return 1

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "gh issue create failed."
        sys.stderr.write(stderr + "\n")
        return 1

    output = result.stdout.strip() or "GitHub issue created."
    sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
