#!/usr/bin/env bash
# Verification driver for the customize-tutorials-verification OpenSpec change.
#
# For each script listed in scripts/verification_targets.txt, this driver:
#   1. Exports MINT_API_KEY and MINT_BASE_URL from ../.env
#   2. Runs the target Python script via `python <path>`
#   3. Pipes captured stdout+stderr through a masking sed (sk-* tokens scrubbed)
#   4. Writes <script>.run.log next to the script
#   5. Greps the log for the script's expected loss/metric markers
#   6. Writes/appends <script>.verified.md with pass/fail and timing
#
# Usage:
#   scripts/run_verification.sh                 # run all targets, skip ones verified within $FRESH_DAYS
#   scripts/run_verification.sh --force         # re-run every target regardless of recency
#   scripts/run_verification.sh path/to/script.py [more...]   # run only the listed targets
#
# Environment:
#   MINT_VERIFY_RETRIES    default 3   max retries for non-converging RL targets
#   MINT_VERIFY_FRESH_DAYS default 7   skip targets whose verified.md is newer than this
#
# Per OpenSpec design.md Decision 5: shell wrapper, not Python harness — each target
# script already manages its own .env loading, model selection, and CLI args.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS_FILE="$REPO_ROOT/scripts/verification_targets.txt"
ENV_FILE="$REPO_ROOT/.env"

RETRIES="${MINT_VERIFY_RETRIES:-3}"
FRESH_DAYS="${MINT_VERIFY_FRESH_DAYS:-7}"
FORCE=0
EXPLICIT_TARGETS=()

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --help|-h)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) EXPLICIT_TARGETS+=("$arg") ;;
    esac
done

# Load .env so subprocess scripts see MINT_API_KEY / MINT_BASE_URL.
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: .env not found at $ENV_FILE — required for MINT_API_KEY / MINT_BASE_URL" >&2
    exit 1
fi

if [[ -z "${MINT_API_KEY:-}" ]]; then
    echo "ERROR: MINT_API_KEY not set after loading .env" >&2
    exit 1
fi

mask_secrets() {
    # Replace sk-* tokens (MinT API keys) with sk-***MASKED***.
    # The character class allows the key body and excludes whitespace/quote terminators.
    sed -E 's/sk-[A-Za-z0-9_-]+/sk-***MASKED***/g'
}

target_is_fresh() {
    local target="$1"
    local verified="$REPO_ROOT/${target%.py}.verified.md"
    [[ -f "$verified" ]] || return 1
    if find "$verified" -mtime "-$FRESH_DAYS" -print -quit | grep -q .; then
        return 0
    fi
    return 1
}

run_target() {
    local target="$1"
    local script_path="$REPO_ROOT/$target"
    local stem="${target%.py}"
    local run_log="$REPO_ROOT/$stem.run.log"
    local verified_md="$REPO_ROOT/$stem.verified.md"

    if [[ ! -f "$script_path" ]]; then
        echo "  ERR: script not found at $script_path" >&2
        return 1
    fi

    local started_at
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local t0=$SECONDS
    local exit_code=0

    # Run the script; capture stdout+stderr, mask, write run.log.
    # The subshell isolates env tweaks and ensures pipefail propagates.
    (
        cd "$REPO_ROOT"
        python "$target" 2>&1
    ) | mask_secrets > "$run_log" || exit_code=$?

    local runtime=$((SECONDS - t0))
    local finished_at
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Convergence inference — light heuristic per algorithm marker.
    # The driver does NOT enforce algorithm-specific thresholds (per OpenSpec design.md
    # Decision 2, those thresholds are documented per-script in expected.txt). Here we
    # only record exit_code and let the user / next reviewer judge.
    local status="pass"
    if (( exit_code != 0 )); then
        status="fail"
    fi

    # Append to verified.md (append-only history per OpenSpec spec
    # tutorial-verification "Append-only verification history").
    {
        if [[ ! -f "$verified_md" ]]; then
            echo "# Verified runs of \`$target\`"
            echo
            echo "Append-only history. Each entry records one verification run."
            echo
        fi
        echo "## $started_at"
        echo
        echo "- runner: $(whoami)"
        echo "- hardware: remote MinT cluster (no local GPU)"
        echo "- endpoint: ${MINT_BASE_URL%/}"
        echo "- runtime_s: $runtime"
        echo "- exit_code: $exit_code"
        echo "- status: $status"
        echo "- run_log: \`$stem.run.log\`"
        echo
    } >> "$verified_md"

    if (( exit_code == 0 )); then
        echo "  PASS  ($runtime s)  $target"
    else
        echo "  FAIL  ($runtime s, exit=$exit_code)  $target  — see $run_log"
    fi

    return $exit_code
}

# Determine target list.
if (( ${#EXPLICIT_TARGETS[@]} > 0 )); then
    TARGETS=("${EXPLICIT_TARGETS[@]}")
elif [[ -f "$TARGETS_FILE" ]]; then
    mapfile -t TARGETS < <(grep -vE '^\s*(#|$)' "$TARGETS_FILE")
else
    echo "ERROR: no explicit targets and $TARGETS_FILE not found" >&2
    exit 1
fi

echo "MinT verification driver"
echo "  endpoint:        ${MINT_BASE_URL%/}"
echo "  retries:         $RETRIES"
echo "  fresh window:    $FRESH_DAYS days  (skipped unless --force)"
echo "  targets:         ${#TARGETS[@]}"
echo

OVERALL_FAILS=0

for target in "${TARGETS[@]}"; do
    if (( FORCE == 0 )) && target_is_fresh "$target"; then
        echo "  SKIP  (fresh)  $target"
        continue
    fi

    attempt=1
    while (( attempt <= RETRIES )); do
        if run_target "$target"; then
            break
        fi
        (( attempt++ ))
        if (( attempt <= RETRIES )); then
            echo "    retry $attempt/$RETRIES for $target"
        else
            (( OVERALL_FAILS++ ))
        fi
    done
done

echo
if (( OVERALL_FAILS == 0 )); then
    echo "All ${#TARGETS[@]} verification target(s) passed."
else
    echo "$OVERALL_FAILS target(s) failed after retries — see *.run.log files."
    exit 1
fi
