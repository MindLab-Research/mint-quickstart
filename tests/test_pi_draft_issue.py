from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / ".pi"
    / "skills"
    / "mint-issue-reporter"
    / "scripts"
    / "draft_issue.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_draft_issue", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_draft_issue_markdown_includes_sections_and_bundle_paths(tmp_path: Path) -> None:
    module = _load_module()
    bundle_dir = tmp_path / ".issue-handoffs" / "issue-123"
    bundle_dir.mkdir(parents=True)
    manifest = {
        "entries": [
            {
                "target": str(bundle_dir / "sources" / "demo.py"),
            }
        ]
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    markdown = module.draft_issue_markdown(
        title="Checkpoint bug",
        summary="Resume breaks for a valid checkpoint path.",
        current_behavior="The script exits after the resume call.",
        expected_behavior="The script should recover or show a clear fix.",
        repro_steps=["Run the resume command once."],
        affected_files=["advanced/checkpoint.py"],
        impact_items=["users cannot continue training"],
        root_cause="The fallback path is not explained clearly.",
        error_block="RuntimeError: bad fallback",
        bundle_paths=module._load_bundle_targets(bundle_dir),
    )

    assert "# Issue: Checkpoint bug" in markdown
    assert "## Summary" in markdown
    assert "## Local Repro Bundle" in markdown
    assert str(bundle_dir) in markdown
    assert "advanced/checkpoint.py" in markdown


def test_main_writes_issue_file(tmp_path: Path, capsys) -> None:
    module = _load_module()
    bundle_root = tmp_path / ".issue-handoffs"
    bundle_dir = bundle_root / "issue-write"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "manifest.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    error_file = tmp_path / "error.log"
    error_file.write_text("boom\n", encoding="utf-8")

    exit_code = module.main(
        [
            "--slug",
            "issue-write",
            "--title",
            "Write issue",
            "--summary",
            "A short summary.",
            "--current",
            "Current behavior.",
            "--expected",
            "Expected behavior.",
            "--repro-step",
            "Run the command.",
            "--affected-file",
            "advanced/checkpoint.py",
            "--error-file",
            str(error_file),
            "--bundle-root",
            str(bundle_root),
        ]
    )
    captured = capsys.readouterr()

    issue_path = bundle_dir / "issue.md"
    assert exit_code == 0
    assert captured.out.strip() == str(issue_path.resolve())
    assert issue_path.exists()
    assert "# Issue: Write issue" in issue_path.read_text(encoding="utf-8")


def test_main_dry_run_prints_markdown(tmp_path: Path, capsys) -> None:
    module = _load_module()
    bundle_root = tmp_path / ".issue-handoffs"
    (bundle_root / "issue-dry").mkdir(parents=True)

    exit_code = module.main(
        [
            "--slug",
            "issue-dry",
            "--title",
            "Dry run",
            "--summary",
            "A short summary.",
            "--current",
            "Current behavior.",
            "--expected",
            "Expected behavior.",
            "--repro-step",
            "Run the command.",
            "--bundle-root",
            str(bundle_root),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "# Issue: Dry run" in captured.out
    assert not (bundle_root / "issue-dry" / "issue.md").exists()
