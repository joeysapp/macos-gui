#!/bin/zsh -f
# check.sh - regression checks for airplay-tool failure paths, flag validation, and contracts.
#
# Each case checks the exit code, JSON formatting, and ensuring no state pollution.
# Run in an isolated temporary XDG environment so local user configs are unaffected.

HERE=${0:A:h}
TOOL_BIN="$HERE/airplay-tool"

usage() {
    cat <<EOF
Usage:
  tools/airplay-tool/check.sh [options]

Runs airplay-tool's failure paths and contract checks.
Verifies exit codes (0 ok, 1 failed, 2 usage, 4 timeout), JSON error structures,
interval parser validity, alias CRUD, and option handling.

Options:
  --binary <path>    Run a specific binary instead of tools/airplay-tool/airplay-tool
  -j, --json         Output machine-readable JSON summary
  -v, --verbose      Display command lines, stdout and stderr per case
  -h, --help         Show this help message

Exit codes:
  0 every case passed
  1 at least one case failed
  2 usage error
EOF
}

json=0
verbose=0
custom_bin=""

while (( $# )); do
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        -j|--json)
            json=1
            ;;
        -v|--verbose)
            verbose=1
            ;;
        -jv|-vj)
            json=1
            verbose=1
            ;;
        --binary)
            shift
            custom_bin="$1"
            ;;
        --binary=*)
            custom_bin="${1#--binary=}"
            ;;
        *)
            print -u2 "error: unknown argument '$1'"
            usage
            exit 2
            ;;
    esac
    shift
done

if [[ -n "$custom_bin" ]]; then
    TOOL_BIN="$custom_bin"
fi

if [[ ! -x "$TOOL_BIN" ]]; then
    print -u2 "error: binary '$TOOL_BIN' not found or not executable"
    exit 2
fi

# Isolated temporary environment
TMP_DIR=$(mktemp -d "${${TMPDIR:-/tmp}%/}/airplay-check.XXXXXX") || exit 1
cleanup() {
    rm -rf "$TMP_DIR" 2>/dev/null
}
trap cleanup EXIT INT TERM

export XDG_CONFIG_HOME="$TMP_DIR/config"
export XDG_STATE_HOME="$TMP_DIR/state"
export XDG_CACHE_HOME="$TMP_DIR/cache"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"

typeset -a CASE_NAMES CASE_OK CASE_EXIT CASE_WANT CASE_WHY
typeset -i PASSED=0 FAILED=0

run_case() {
    local name="$1"
    local want_exit="$2"
    local check_type="$3" # "json_error", "json_array", "json_object", "text", "exit_only"
    shift 3
    local cmd=("$TOOL_BIN" "$@")

    local stdout_file="$TMP_DIR/stdout.$$"
    local stderr_file="$TMP_DIR/stderr.$$"
    rm -f "$stdout_file" "$stderr_file"

    if (( verbose )); then
        print -u2 "==> Case: $name"
        print -u2 "    Command: ${(q)cmd}"
    fi

    local actual_exit=0
    "$TOOL_BIN" "$@" >"$stdout_file" 2>"$stderr_file" || actual_exit=$?

    local out=$(<"$stdout_file")
    local err=$(<"$stderr_file")

    if (( verbose )); then
        [[ -n "$out" ]] && print -u2 "    stdout: $out"
        [[ -n "$err" ]] && print -u2 "    stderr: $err"
        print -u2 "    exit: $actual_exit (want $want_exit)"
    fi

    local ok=1
    local why=""

    if [[ "$actual_exit" != "$want_exit" ]]; then
        ok=0
        why="exit code was $actual_exit, expected $want_exit"
    elif [[ "$check_type" == "json_error" ]]; then
        if ! python3 -c "import json, sys; d=json.loads(sys.argv[1]); assert d.get('success') is False; assert 'error' in d; assert 'code' in d" "$out" 2>/dev/null; then
            ok=0
            why="stdout was not a valid JSON error object: $out"
        fi
    elif [[ "$check_type" == "json_array" ]]; then
        if ! python3 -c "import json, sys; d=json.loads(sys.argv[1]); assert isinstance(d, list)" "$out" 2>/dev/null; then
            ok=0
            why="stdout was not a valid JSON array: $out"
        fi
    elif [[ "$check_type" == "json_object" ]]; then
        if ! python3 -c "import json, sys; d=json.loads(sys.argv[1]); assert isinstance(d, dict)" "$out" 2>/dev/null; then
            ok=0
            why="stdout was not a valid JSON object: $out"
        fi
    fi

    CASE_NAMES+=("$name")
    CASE_OK+=("$ok")
    CASE_EXIT+=("$actual_exit")
    CASE_WANT+=("$want_exit")
    CASE_WHY+=("$why")

    if (( ok )); then
        (( PASSED++ ))
        (( ! json && ! verbose )) && print "  ✅ $name"
    else
        (( FAILED++ ))
        (( ! json )) && print -u2 "  ❌ $name ($why)"
    fi
}

# 1. Help & Version
run_case "help_flag" 0 "text" --help
run_case "version_flag" 0 "text" --version
run_case "subcmd_help_watch" 0 "text" watch --help
run_case "subcmd_help_start" 0 "text" start --help
run_case "subcmd_help_toggle" 0 "text" toggle --help
run_case "subcmd_help_wait" 0 "text" wait --help
run_case "subcmd_help_logs" 0 "text" logs --help
run_case "subcmd_help_alias" 0 "text" alias --help

# 2. Usage error paths & JSON error contract
run_case "unknown_command_text" 2 "text" unknowncmd
run_case "unknown_command_json" 2 "json_error" -j unknowncmd
run_case "start_missing_target" 2 "text" start
run_case "start_missing_target_json" 2 "json_error" -j start
run_case "wait_missing_target" 2 "text" wait
run_case "wait_missing_target_json" 2 "json_error" -j wait
run_case "wait_invalid_state" 2 "json_error" -j wait dummy_display invalid_state
run_case "watch_invalid_interval_text" 2 "text" watch dummy_display --interval abc
run_case "watch_invalid_interval_json" 2 "json_error" -j watch dummy_display --interval abc
run_case "watch_negative_interval" 2 "json_error" -j watch dummy_display --interval -5s
run_case "invalid_timeout_flag" 2 "json_error" -j start dummy_display --timeout invalid
run_case "invalid_limit_flag" 2 "json_error" -j logs --limit invalid

# 3. Alias validation & CRUD
run_case "alias_reserved_name" 2 "json_error" -j alias set start "LG OLED"
run_case "alias_invalid_chars" 2 "json_error" -j alias set "bad alias!" "LG OLED"
run_case "alias_set_success" 0 "json_object" -j alias set testtv "LG OLED"
run_case "alias_list_json" 0 "json_object" -j alias list
run_case "alias_remove_success" 0 "json_object" -j alias remove testtv
run_case "alias_remove_nonexistent" 1 "json_error" -j alias remove testtv

# 4. Wait timeout path (exit 4 UNCONFIRMED)
run_case "wait_timeout_exit4" 4 "json_error" -j wait "nonexistent_device_XYZ" connected --timeout 0.1

# 5. Logs inspection query (short window)
run_case "logs_json_array" 0 "json_array" -j logs --last 1m --limit 5

# Output results
if (( json )); then
    python3 -c "
import json, sys
data = {
    'tool': '$TOOL_BIN',
    'passed': $PASSED,
    'failed': $FAILED,
    'cases': [
        {'name': n, 'ok': bool(int(o)), 'exit': int(e), 'want': int(w), 'why': (y if y else None)}
        for n, o, e, w, y in zip(
            sys.argv[1].split(';;;'),
            sys.argv[2].split(';;;'),
            sys.argv[3].split(';;;'),
            sys.argv[4].split(';;;'),
            sys.argv[5].split(';;;')
        )
    ]
}
print(json.dumps(data, indent=2))
" "${(j:;;;:)CASE_NAMES}" "${(j:;;;:)CASE_OK}" "${(j:;;;:)CASE_EXIT}" "${(j:;;;:)CASE_WANT}" "${(j:;;;:)CASE_WHY}"
else
    print ""
    print "airplay-tool regression checks: $PASSED passed, $FAILED failed."
fi

if (( FAILED > 0 )); then
    exit 1
fi
exit 0
