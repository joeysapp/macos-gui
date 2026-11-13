#!/bin/zsh
set -e

DIR="${0:A:h}"
BIN="$DIR/hot-corner-tool"

# Compile if needed
if [[ ! -x "$BIN" || "$DIR/hot-corner-tool.swift" -nt "$BIN" ]]; then
    print "Compiling hot-corner-tool..."
    swiftc -module-cache-path /tmp/mcache -O "$DIR/hot-corner-tool.swift" -o "$BIN"
fi

print "Running regression suite for hot-corner-tool..."

PASSED=0
FAILED=0

assert_exit() {
    local expected="$1"
    local desc="$2"
    shift 2
    local code=0
    { "$@" >/dev/null 2>&1 && code=0; } || code=$?
    if [[ $code -eq $expected ]]; then
        print "  ✅ $desc (exit $code)"
        PASSED=$(( PASSED + 1 ))
    else
        print "  ❌ $desc (expected $expected, got $code)"
        FAILED=$(( FAILED + 1 ))
    fi
}

assert_json() {
    local field="$1"
    local expected="$2"
    local desc="$3"
    shift 3
    local out="$("$@" 2>/dev/null || true)"
    local val="$(echo "$out" | grep "\"$field\"" | head -1 | tr -d '",: ')"
    val="${val#$field}"
    if [[ "$val" == "$expected" ]]; then
        print "  ✅ $desc ($field=$val)"
        PASSED=$(( PASSED + 1 ))
    else
        print "  ❌ $desc ($field: expected $expected, got '$val')"
        FAILED=$(( FAILED + 1 ))
    fi
}

# 1. Version
assert_exit 0 "Version flag (-V)" "$BIN" -V
assert_exit 0 "Version flag (--version)" "$BIN" --version
assert_exit 0 "Version command (version)" "$BIN" version

# 2. Help
assert_exit 0 "Help flag (-h)" "$BIN" -h
assert_exit 0 "Help flag (--help)" "$BIN" --help
assert_exit 0 "Help command (help)" "$BIN" help
assert_exit 0 "Subcommand help (help desktop)" "$BIN" help desktop
assert_exit 0 "Subcommand help (help mission-control)" "$BIN" help mission-control
assert_exit 0 "Subcommand help (help app-windows)" "$BIN" help app-windows
assert_exit 0 "Subcommand help (help notification-center)" "$BIN" help notification-center

# 3. Usage Errors
assert_exit 2 "No arguments" "$BIN"
assert_exit 2 "Unknown command" "$BIN" non-existent-command
assert_exit 2 "Unknown flag" "$BIN" --invalid-flag
assert_exit 2 "Unknown command with JSON" "$BIN" non-existent-command -j
assert_json "code" "2" "JSON error code on usage error" "$BIN" bad-cmd -j
assert_json "success" "false" "JSON success=false on usage error" "$BIN" bad-cmd -j

# 4. Actions (testing exit code and JSON structure)
assert_exit 0 "Action desktop (-q)" "$BIN" desktop -q
assert_json "action" "desktop" "JSON output for desktop" "$BIN" desktop -j
assert_json "success" "true" "JSON success for desktop" "$BIN" desktop -j

assert_exit 0 "Action mission-control (-q)" "$BIN" mission-control -q
assert_json "action" "mission-control" "JSON output for mission-control" "$BIN" mission-control -j

assert_exit 0 "Action app-windows (-q)" "$BIN" app-windows -q
assert_json "action" "app-windows" "JSON output for app-windows" "$BIN" app-windows -j

# Aliases
assert_exit 0 "Alias desk (-q)" "$BIN" desk -q
assert_exit 0 "Alias mc (-q)" "$BIN" mc -q
assert_exit 0 "Alias windows (-q)" "$BIN" windows -q
assert_exit 0 "Alias expose (-q)" "$BIN" expose -q

print "\nSuite finished: $PASSED passed, $FAILED failed."
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
exit 0
