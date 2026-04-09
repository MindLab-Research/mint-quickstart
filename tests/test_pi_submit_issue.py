from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / ".pi"
    / "skills"
    / "mint-issue-reporter"
    / "scripts"
    / "submit_github_issue.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_submit_issue", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_gh_issue_command_includes_repo_title_body_and_labels(tmp_path: Path) -> None:
    module = _load_module()
    body_file = tmp_path / "issue.md"
    body_file.write_text("# issue\n", encoding="utf-8")

    command = module.build_gh_issue_command(
        repo="MindLab-Research/mint-quickstart",
        title="Checkpoint bug",
        body_file=body_file,
        labels=["bug", "docs"],
    )

    assert command[:6] == [
        "gh",
        "issue",
        "create",
        "--repo",
        "MindLab-Research/mint-quickstart",
        "--title",
    ]
    assert "Checkpoint bug" in command
    assert str(body_file.resolve()) in command
    assert command.count("--label") == 2


def test_main_dry_run_prints_command(tmp_path: Path, capsys) -> None:
    module = _load_module()
    body_file = tmp_path / "issue.md"
    body_file.write_text("# issue\n", encoding="utf-8")

    exit_code = module.main(
        [
            "--title",
            "Checkpoint bug",
            "--body-file",
            str(body_file),
            "--label",
            "bug",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "gh issue create" in captured.out
    assert "--label bug" in captured.out


def test_main_returns_nonzero_for_missing_body_file(capsys) -> None:
    module = _load_module()

    exit_code = module.main(
        [
            "--title",
            "Missing body",
            "--body-file",
            "missing.md",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Body file not found" in captured.err
