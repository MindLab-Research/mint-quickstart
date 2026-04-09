---
name: mint-issue-reporter
description: Prepare GitHub-ready issues for confirmed `mint-quickstart` bugs or docs gaps. Use after ownership is clear. Collects sanitized local repro bundles under `.issue-handoffs/`, drafts Markdown issues, and can optionally submit them with `gh issue create`.
---

# MinT Issue Reporter

Use this skill only after the problem is confirmed to belong to `MindLab-Research/mint-quickstart`.

## Output Contract

Produce:
- a GitHub-ready Markdown issue body
- a local repro bundle under `.issue-handoffs/<slug>/` when code, logs, or commands need to be handed off
- optionally, a submitted GitHub issue if `gh` is available and the user wants submission

## Hard Gates

Do not file a `mint-quickstart` issue for these by default:
- missing or bad `MINT_API_KEY`
- wrong endpoint or generic connectivity outage
- user-local environment mistakes already covered by docs
- likely upstream MinT SDK or backend problems with no repo-side fix

Before drafting, make sure you have:
- current behavior
- expected behavior
- minimal repro command or script
- affected file paths
- sanitized logs or code snippets only

Never include:
- API keys
- private absolute paths unless the user explicitly wants them
- private dataset contents
- large raw files pasted directly into the GitHub issue body

## Workflow

1. Confirm ownership.
2. Collect the smallest repro.
3. Sanitize secrets and private data.
4. Use `scripts/collect_repro_bundle.py` when code, logs, commands, or code snippets should be bundled.
5. Use `scripts/draft_issue.py` to generate `issue.md` from the collected facts.
6. If asked to submit, use `scripts/submit_github_issue.py`.

## Source Of Truth Files

- `issue_template.md`
- `scripts/collect_repro_bundle.py`
- `scripts/draft_issue.py`
- `scripts/submit_github_issue.py`
- `../../../README.md`
- `../../../docs/troubleshooting.md`
- the affected repo files for the actual bug or docs gap

## Bundle Workflow

Create a local bundle when the issue depends on client-side repro assets.

```bash
python .pi/skills/mint-issue-reporter/scripts/collect_repro_bundle.py \
  --slug checkpoint-resume-falls-back-poorly \
  --src advanced/checkpoint.py \
  --log-file /tmp/mint-error.log \
  --snippet advanced/checkpoint.py:580:645 \
  --command "python advanced/checkpoint.py resume tinker://run-id/weights/demo --with-optimizer"
```

That creates `.issue-handoffs/<slug>/` with copied sources, logs, snippet files, a manifest, and optional repro commands.

## Draft Workflow

After the bundle exists, generate `issue.md`:

```bash
python .pi/skills/mint-issue-reporter/scripts/draft_issue.py \
  --slug checkpoint-resume-falls-back-poorly \
  --title "Checkpoint resume fails after metadata 404 fallback" \
  --summary "Resume falls back, but the resulting guidance and error handling are still confusing for users." \
  --current "The resume flow does not make the recovery path clear after metadata lookup fails." \
  --expected "The resume path should either recover cleanly or fail with a precise, user-facing explanation." \
  --repro-step "Set MINT_API_KEY and point to a checkpoint path that triggers metadata 404." \
  --repro-step "Run python advanced/checkpoint.py resume tinker://run-id/weights/demo --with-optimizer." \
  --affected-file advanced/checkpoint.py \
  --error-file /tmp/mint-error.log
```

## Submission Workflow

Only submit when the user wants it and `gh` is ready.

```bash
python .pi/skills/mint-issue-reporter/scripts/submit_github_issue.py \
  --repo MindLab-Research/mint-quickstart \
  --title "Checkpoint resume fails after metadata 404 fallback" \
  --body-file .issue-handoffs/checkpoint-resume-falls-back-poorly/issue.md
```

Use `--dry-run` first if you want to preview the exact `gh` command.
