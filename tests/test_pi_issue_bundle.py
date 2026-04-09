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
    / "collect_repro_bundle.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mint_issue_bundle", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_repro_bundle_copies_file_and_commands(tmp_path: Path) -> None:
    module = _load_module()
    source_file = tmp_path / "demo.py"
    source_file.write_text("print('ok')\n", encoding="utf-8")
    bundle_root = tmp_path / "bundles"

    result = module.collect_repro_bundle(
        "issue-123",
        sources=[str(source_file)],
        commands=["python demo.py"],
        bundle_root=bundle_root,
    )

    bundle_dir = bundle_root.resolve() / "issue-123"
    copied_file = bundle_dir / "sources" / "demo.py"
    manifest_path = bundle_dir / "manifest.json"
    commands_path = bundle_dir / "repro_commands.sh"

    assert copied_file.read_text(encoding="utf-8") == "print('ok')\n"
    assert "python demo.py" in commands_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["slug"] == "issue-123"
    assert manifest["commands"] == ["python demo.py"]
    assert result.markdown_paths()[0] == f"- `{bundle_dir}`"


def test_collect_repro_bundle_copies_directory_log_file_and_snippet(tmp_path: Path) -> None:
    module = _load_module()
    source_dir = tmp_path / "client"
    source_dir.mkdir()
    train_file = source_dir / "train.py"
    train_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
    log_file = tmp_path / "error.log"
    log_file.write_text("boom\n", encoding="utf-8")
    bundle_root = tmp_path / "bundles"

    module.collect_repro_bundle(
        "issue-dir",
        sources=[str(source_dir)],
        log_files=[str(log_file)],
        snippets=[f"{train_file}:2:3"],
        bundle_root=bundle_root,
    )

    copied_code = bundle_root.resolve() / "issue-dir" / "sources" / "client" / "train.py"
    copied_log = bundle_root.resolve() / "issue-dir" / "logs" / "error.log"
    snippet_file = bundle_root.resolve() / "issue-dir" / "snippets" / "train.py.L2-L3.md"
    snippet_text = snippet_file.read_text(encoding="utf-8")

    assert copied_code.read_text(encoding="utf-8") == "line1\nline2\nline3\n"
    assert copied_log.read_text(encoding="utf-8") == "boom\n"
    assert "line2" in snippet_text
    assert "line3" in snippet_text
    assert "line1" not in snippet_text


def test_collect_repro_bundle_requires_overwrite_for_existing_bundle(tmp_path: Path) -> None:
    module = _load_module()
    source_file = tmp_path / "demo.py"
    source_file.write_text("print('v1')\n", encoding="utf-8")
    bundle_root = tmp_path / "bundles"

    module.collect_repro_bundle("issue-overwrite", sources=[str(source_file)], bundle_root=bundle_root)

    source_file.write_text("print('v2')\n", encoding="utf-8")
    try:
        module.collect_repro_bundle("issue-overwrite", sources=[str(source_file)], bundle_root=bundle_root)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError when overwriting without --overwrite.")

    module.collect_repro_bundle(
        "issue-overwrite",
        sources=[str(source_file)],
        bundle_root=bundle_root,
        overwrite=True,
    )
    copied_file = bundle_root.resolve() / "issue-overwrite" / "sources" / "demo.py"
    assert copied_file.read_text(encoding="utf-8") == "print('v2')\n"


def test_main_dry_run_prints_markdown_paths_without_writing(tmp_path: Path, capsys) -> None:
    module = _load_module()
    source_file = tmp_path / "demo.py"
    source_file.write_text("print('dry-run')\n", encoding="utf-8")
    bundle_root = tmp_path / "bundles"
    expected_bundle = bundle_root.resolve() / "issue-dry-run"

    exit_code = module.main(
        [
            "--slug",
            "issue-dry-run",
            "--src",
            str(source_file),
            "--bundle-root",
            str(bundle_root),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines()[0] == f"- `{expected_bundle}`"
    assert not expected_bundle.exists()
