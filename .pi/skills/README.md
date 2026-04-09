# Pi Skills

This repo now ships project-local `pi` skills under `.pi/skills/`.

Available skills:
- `mint-api` - write, review, or explain MinT client code in this repo
- `mint-client-debugger` - diagnose client-side bugs, slow paths, and bad call shapes
- `mint-issue-reporter` - prepare GitHub-ready issues with a local repro bundle, code snippets, issue drafting, and optional `gh` submission

The older `mint-skill/` directory stays as the migration-focused skill for moving code from `verl`, `TRL`, or similar frameworks to MinT.

## Usage

Pi discovers these skills automatically from `.pi/skills/`.

You can also load them explicitly:

```bash
/skill:mint-api
/skill:mint-client-debugger
/skill:mint-issue-reporter
```

Use the issue reporter only after the problem is confirmed to belong to `MindLab-Research/mint-quickstart`.
