#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_BUNDLE_ROOT = Path(".issue-handoffs")
MAX_ERROR_CHARS = 4000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draft a GitHub-ready issue.md file for mint-quickstart.",
    )
    parser.add_argument("--slug", required=True, help="Issue slug and bundle directory name.")
    parser.add_argument("--title", required=True, help="Issue title.")
    parser.add_argument("--summary", required=True, help="Short summary paragraph.")
    parser.add_argument("--current", required=True, help="Current behavior description.")
    parser.add_argument("--expected", required=True, help="Expected behavior description.")
    parser.add_argument(
        "--repro-step",
        action="append",
        dest="repro_steps",
        default=[],
        help="Reproduction step. Repeat for multiple steps.",
    )
    parser.add_argument(
        "--affected-file",
        action="append",
        dest="affected_files",
        default=[],
        help="Affected repo file. Repeat for multiple files.",
    )
    parser.add_argument(
        "--impact",
        action="append",
        dest="impact_items",
        default=[],
        help="Impact bullet. Repeat for multiple items.",
    )
    parser.add_argument(
        "--root-cause",
        default="Not yet proven. This is the current best hypothesis based on the evidence collected so far.",
        help="Root cause hypothesis.",
    )
    parser.add_argument(
        "--error-text",
        default="",
        help="Inline error or log snippet.",
    )
    parser.add_argument(
        "--error-file",
        default="",
        help="Path to a log file to include in the Error section.",
    )
    parser.add_argument(
        "--bundle-root",
        default=str(DEFAULT_BUNDLE_ROOT),
        help="Root directory for local issue bundles.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit output file path. Defaults to .issue-handoffs/<slug>/issue.md.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated Markdown instead of writing the file.",
    )
    return parser


def _trim_text(text: str, *, limit: int = MAX_ERROR_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "\n... [truncated]"


def _load_error_block(*, error_text: str, error_file: str) -> str:
    if error_text.strip():
        return _trim_text(error_text)
    if error_file:
        path = Path(error_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Error file not found: {path}")
        return _trim_text(path.read_text(encoding="utf-8"))
    return "<no error snippet provided>"


def _load_bundle_targets(bundle_dir: Path) -> list[str]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return [str(bundle_dir)]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = [str(bundle_dir)]
    for entry in manifest.get("entries", []):
        target = entry.get("target")
        if isinstance(target, str) and target:
            lines.append(target)
    commands_path = bundle_dir / "repro_commands.sh"
    if commands_path.exists():
        lines.append(str(commands_path))
    return lines


def draft_issue_markdown(
    *,
    title: str,
    summary: str,
    current_behavior: str,
    expected_behavior: str,
    repro_steps: Sequence[str],
    affected_files: Sequence[str],
    impact_items: Sequence[str],
    root_cause: str,
    error_block: str,
    bundle_paths: Sequence[str],
) -> str:
    if not repro_steps:
        raise ValueError("At least one reproduction step is required.")

    affected_lines = [f"- `{path}`" for path in affected_files] or ["- `<fill in affected files>`"]
    impact_lines = [f"- {item}" for item in impact_items] or ["- user needs a workaround or manual investigation"]
    bundle_lines = [f"- `{path}`" for path in bundle_paths]
    repro_lines = [f"{index}. {step}" for index, step in enumerate(repro_steps, start=1)]

    sections = [
        f"# Issue: {title}",
        "",
        "## Summary",
        summary.strip(),
        "",
        "## Current Behavior",
        current_behavior.strip(),
        "",
        "## Error",
        "```text",
        error_block.strip(),
        "```",
        "",
        "## Reproduction",
        *repro_lines,
        "",
        "## Expected Behavior",
        expected_behavior.strip(),
        "",
        "## Root Cause (hypothesis)",
        root_cause.strip(),
        "",
        "## Affected Files",
        *affected_lines,
        "",
        "## Local Repro Bundle",
        *bundle_lines,
        "",
        "## Impact",
        *impact_lines,
        "",
        "## Acceptance Criteria",
        "- [ ] minimal repro is covered",
        "- [ ] behavior matches expectation after fix",
        "- [ ] tests or docs updated when needed",
        "",
    ]
    return "\n".join(sections)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        error_block = _load_error_block(error_text=args.error_text, error_file=args.error_file)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    bundle_root = Path(args.bundle_root).expanduser().resolve()
    bundle_dir = bundle_root / args.slug
    bundle_paths = _load_bundle_targets(bundle_dir)

    try:
        markdown = draft_issue_markdown(
            title=args.title,
            summary=args.summary,
            current_behavior=args.current,
            expected_behavior=args.expected,
            repro_steps=args.repro_steps,
            affected_files=args.affected_files,
            impact_items=args.impact_items,
            root_cause=args.root_cause,
            error_block=error_block,
            bundle_paths=bundle_paths,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    if args.dry_run:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    output_path = Path(args.output).expanduser().resolve() if args.output else (bundle_dir / "issue.md").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    sys.stdout.write(str(output_path) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
