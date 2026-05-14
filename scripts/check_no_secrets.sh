#!/usr/bin/env bash
# Pre-commit hook: reject any staged file (run.log or otherwise) containing a literal
# MinT API key (sk-* token). The verification driver's sed mask should produce
# sk-***MASKED***; any unmasked sk-[A-Za-z0-9_-]+ is a leak.
#
# Wired via the local .git/hooks/pre-commit (or pre-commit-hooks framework).
# Exits non-zero on any match so git commit aborts.

set -euo pipefail

if [[ -n "${MINT_ALLOW_SK_TOKENS:-}" ]]; then
    exit 0
fi

STAGED=$(git diff --cached --name-only --diff-filter=ACM)
if [[ -z "$STAGED" ]]; then
    exit 0
fi

LEAK_FOUND=0
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    [[ ! -f "$file" ]] && continue
    case "$file" in
        scripts/run_verification.sh|scripts/check_no_secrets.sh|scripts/README.md|*.md|*.mdx) continue ;;
    esac
    if grep -E "sk-[A-Za-z0-9_-]{20,}" "$file" >/dev/null 2>&1; then
        echo "ERROR: possible MinT API key leak in staged file: $file" >&2
        grep -nE "sk-[A-Za-z0-9_-]{20,}" "$file" | sed 's/sk-[A-Za-z0-9_-]\{20,\}/sk-***LEAK***/g' >&2
        LEAK_FOUND=1
    fi
done <<< "$STAGED"

if (( LEAK_FOUND )); then
    echo >&2
    echo "Commit rejected. Mask the token (sk-***MASKED***) or rerun the verification driver" >&2
    echo "(scripts/run_verification.sh) which masks logs automatically." >&2
    echo "If you really mean to commit a literal token, set MINT_ALLOW_SK_TOKENS=1." >&2
    exit 1
fi

exit 0
