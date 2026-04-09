#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_BUNDLE_ROOT = Path(".issue-handoffs")
SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class BundleEntry:
    source: str
    target: str
    entry_type: str
    group: str


@dataclass(frozen=True)
class SnippetSpec:
    source_path: Path
    start_line: int
    end_line: int

    @property
    def source_label(self) -> str:
        return f"{self.source_path}:{self.start_line}:{self.end_line}"


@dataclass(frozen=True)
class BundleResult:
    slug: str
    bundle_root: str
    bundle_dir: str
    manifest_path: str | None
    dry_run: bool
    entries: list[BundleEntry]
    commands: list[str]

    def markdown_paths(self) -> list[str]:
        bundle_dir = Path(self.bundle_dir)
        lines = [f"- `{bundle_dir}`"]
        for entry in self.entries:
            lines.append(f"- `{entry.target}`")
        if self.commands:
            lines.append(f"- `{bundle_dir / 'repro_commands.sh'}`")
        return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy explicit repro files, directories, logs, snippets, and commands into a local issue bundle.",
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Bundle slug used to create .issue-handoffs/<slug>/.",
    )
    parser.add_argument(
        "--src",
        action="append",
        dest="sources",
        default=[],
        help="File or directory to copy into sources/. Repeat for multiple inputs.",
    )
    parser.add_argument(
        "--log-file",
        action="append",
        dest="log_files",
        default=[],
        help="Log file to copy into logs/. Repeat for multiple files.",
    )
    parser.add_argument(
        "--snippet",
        action="append",
        dest="snippets",
        default=[],
        help="Code snippet to extract in the form path:start_line:end_line. Repeat for multiple snippets.",
    )
    parser.add_argument(
        "--command",
        action="append",
        dest="commands",
        default=[],
        help="Reproduction command to record in repro_commands.sh. Repeat for multiple commands.",
    )
    parser.add_argument(
        "--bundle-root",
        default=str(DEFAULT_BUNDLE_ROOT),
        help="Root directory for local repro bundles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the target Markdown paths without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing bundle with the same slug.",
    )
    return parser


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_slug(slug: str) -> str:
    cleaned = slug.strip()
    if not cleaned:
        raise ValueError("Slug must not be empty.")
    if not SLUG_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "Slug must use only letters, digits, dots, underscores, or hyphens, and must not contain path separators.",
        )
    return cleaned


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _resolve_entries(paths: Sequence[str], *, group: str) -> list[tuple[Path, BundleEntry]]:
    resolved_entries: list[tuple[Path, BundleEntry]] = []
    seen_basenames: dict[str, Path] = {}

    for raw_path in paths:
        source_path = Path(raw_path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        resolved_path = source_path.resolve()
        if resolved_path.is_dir():
            entry_type = "directory"
        elif resolved_path.is_file():
            entry_type = "file"
        else:
            raise ValueError(f"Unsupported source type: {resolved_path}")

        basename = resolved_path.name
        if basename in seen_basenames:
            first_path = seen_basenames[basename]
            raise ValueError(
                f"Multiple {group} entries map to the same basename `{basename}`: {first_path} and {resolved_path}",
            )
        seen_basenames[basename] = resolved_path
        resolved_entries.append(
            (
                resolved_path,
                BundleEntry(
                    source=str(resolved_path),
                    target="",
                    entry_type=entry_type,
                    group=group,
                ),
            )
        )
    return resolved_entries


def _parse_snippet_spec(raw_spec: str) -> SnippetSpec:
    parts = raw_spec.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid snippet specification `{raw_spec}`. Use path:start_line:end_line.",
        )

    raw_path, raw_start, raw_end = parts
    source_path = Path(raw_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")
    resolved_path = source_path.resolve()
    if not resolved_path.is_file():
        raise ValueError(f"Snippet source must be a file: {resolved_path}")

    try:
        start_line = int(raw_start)
        end_line = int(raw_end)
    except ValueError as exc:
        raise ValueError(
            f"Invalid snippet line range `{raw_spec}`. Start and end must be integers.",
        ) from exc

    if start_line < 1 or end_line < 1 or start_line > end_line:
        raise ValueError(
            f"Invalid snippet line range `{raw_spec}`. Use positive lines with start <= end.",
        )

    lines = resolved_path.read_text(encoding="utf-8").splitlines()
    if end_line > len(lines):
        raise ValueError(
            f"Snippet range `{raw_spec}` exceeds file length {len(lines)}.",
        )

    return SnippetSpec(
        source_path=resolved_path,
        start_line=start_line,
        end_line=end_line,
    )


def _resolve_snippets(raw_specs: Sequence[str]) -> list[SnippetSpec]:
    resolved_specs: list[SnippetSpec] = []
    seen_keys: set[tuple[Path, int, int]] = set()

    for raw_spec in raw_specs:
        spec = _parse_snippet_spec(raw_spec)
        key = (spec.source_path, spec.start_line, spec.end_line)
        if key in seen_keys:
            raise ValueError(f"Duplicate snippet specification: {raw_spec}")
        seen_keys.add(key)
        resolved_specs.append(spec)
    return resolved_specs


def _language_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "python",
        ".md": "markdown",
        ".sh": "bash",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".txt": "text",
    }.get(suffix, "text")


def _render_snippet_markdown(spec: SnippetSpec) -> str:
    lines = spec.source_path.read_text(encoding="utf-8").splitlines()
    selected_lines = lines[spec.start_line - 1 : spec.end_line]
    language = _language_from_path(spec.source_path)
    body = "\n".join(selected_lines)
    if body:
        body += "\n"
    return (
        f"Source: `{spec.source_path}:{spec.start_line}:{spec.end_line}`\n\n"
        f"```{language}\n"
        f"{body}"
        f"```\n"
    )


def _snippet_target_name(spec: SnippetSpec) -> str:
    return f"{spec.source_path.name}.L{spec.start_line}-L{spec.end_line}.md"


def collect_repro_bundle(
    slug: str,
    *,
    sources: Sequence[str] | None = None,
    log_files: Sequence[str] | None = None,
    snippets: Sequence[str] | None = None,
    commands: Sequence[str] | None = None,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    dry_run: bool = False,
    overwrite: bool = False,
) -> BundleResult:
    validated_slug = _validate_slug(slug)
    source_list = list(sources or [])
    log_list = list(log_files or [])
    snippet_list = list(snippets or [])
    command_list = [command.strip() for command in (commands or []) if command.strip()]

    if not source_list and not log_list and not snippet_list and not command_list:
        raise ValueError(
            "At least one --src, --log-file, --snippet, or --command value is required.",
        )

    resolved_source_entries = _resolve_entries(source_list, group="sources")
    resolved_log_entries = _resolve_entries(log_list, group="logs")
    resolved_snippets = _resolve_snippets(snippet_list)

    resolved_bundle_root = Path(bundle_root).expanduser().resolve()
    bundle_dir = resolved_bundle_root / validated_slug
    sources_dir = bundle_dir / "sources"
    logs_dir = bundle_dir / "logs"
    snippets_dir = bundle_dir / "snippets"
    manifest_path = bundle_dir / "manifest.json"
    commands_path = bundle_dir / "repro_commands.sh"

    entries: list[BundleEntry] = []
    copy_plan: list[tuple[Path, Path, BundleEntry]] = []
    snippet_plan: list[tuple[SnippetSpec, Path, BundleEntry]] = []

    for resolved_path, entry in resolved_source_entries:
        target_path = sources_dir / resolved_path.name
        final_entry = BundleEntry(
            source=entry.source,
            target=str(target_path),
            entry_type=entry.entry_type,
            group=entry.group,
        )
        entries.append(final_entry)
        copy_plan.append((resolved_path, target_path, final_entry))

    for resolved_path, entry in resolved_log_entries:
        target_path = logs_dir / resolved_path.name
        final_entry = BundleEntry(
            source=entry.source,
            target=str(target_path),
            entry_type=entry.entry_type,
            group=entry.group,
        )
        entries.append(final_entry)
        copy_plan.append((resolved_path, target_path, final_entry))

    for spec in resolved_snippets:
        target_path = snippets_dir / _snippet_target_name(spec)
        final_entry = BundleEntry(
            source=spec.source_label,
            target=str(target_path),
            entry_type="snippet",
            group="snippets",
        )
        entries.append(final_entry)
        snippet_plan.append((spec, target_path, final_entry))

    conflicting_paths = [Path(entry.target) for entry in entries if Path(entry.target).exists()]
    for maybe_conflict in (bundle_dir, manifest_path, commands_path):
        if maybe_conflict.exists():
            conflicting_paths.append(maybe_conflict)

    if conflicting_paths and not overwrite:
        rendered_conflicts = ", ".join(str(path) for path in conflicting_paths)
        raise FileExistsError(
            f"Target paths already exist for slug `{validated_slug}`: {rendered_conflicts}. Re-run with --overwrite to replace them.",
        )

    if dry_run:
        return BundleResult(
            slug=validated_slug,
            bundle_root=str(resolved_bundle_root),
            bundle_dir=str(bundle_dir),
            manifest_path=None,
            dry_run=True,
            entries=entries,
            commands=command_list,
        )

    resolved_bundle_root.mkdir(parents=True, exist_ok=True)
    if overwrite and bundle_dir.exists():
        _remove_path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if resolved_source_entries:
        sources_dir.mkdir(parents=True, exist_ok=True)
    if resolved_log_entries:
        logs_dir.mkdir(parents=True, exist_ok=True)
    if resolved_snippets:
        snippets_dir.mkdir(parents=True, exist_ok=True)

    for source_path, target_path, entry in copy_plan:
        if entry.entry_type == "directory":
            shutil.copytree(source_path, target_path)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

    for spec, target_path, _entry in snippet_plan:
        target_path.write_text(_render_snippet_markdown(spec), encoding="utf-8")

    if command_list:
        script_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""] + command_list + [""]
        commands_path.write_text("\n".join(script_lines), encoding="utf-8")

    manifest = {
        "slug": validated_slug,
        "generated_at": _utc_timestamp(),
        "bundle_root": str(resolved_bundle_root),
        "bundle_dir": str(bundle_dir),
        "commands": command_list,
        "entries": [
            {
                "source": entry.source,
                "target": entry.target,
                "type": entry.entry_type,
                "group": entry.group,
            }
            for entry in entries
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return BundleResult(
        slug=validated_slug,
        bundle_root=str(resolved_bundle_root),
        bundle_dir=str(bundle_dir),
        manifest_path=str(manifest_path),
        dry_run=False,
        entries=entries,
        commands=command_list,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = collect_repro_bundle(
            args.slug,
            sources=args.sources,
            log_files=args.log_files,
            snippets=args.snippets,
            commands=args.commands,
            bundle_root=args.bundle_root,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    for line in result.markdown_paths():
        sys.stdout.write(f"{line}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
