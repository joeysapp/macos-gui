#!/bin/zsh
set -e

DIR="${0:A:h}"
BIN="$DIR/background-tool"

# Compile if needed
if [[ ! -x "$BIN" || "$DIR/background-tool.swift" -nt "$BIN" ]]; then
    print "Compiling background-tool..."
    swiftc -module-cache-path /tmp/mcache -O "$DIR/background-tool.swift" -o "$BIN"
fi

print "Running regression suite for background-tool..."

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

assert_json_field() {
    local field="$1"
    local desc="$2"
    shift 2
    local out="$("$@" 2>/dev/null || true)"
    if echo "$out" | grep -q "\"$field\""; then
        print "  ✅ $desc (field \"$field\" present)"
        PASSED=$(( PASSED + 1 ))
    else
        print "  ❌ $desc (field \"$field\" missing in output)"
        FAILED=$(( FAILED + 1 ))
    fi
}

assert_json_val() {
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

# 1. Version & Help
assert_exit 0 "Version flag (-V)" "$BIN" -V
assert_exit 0 "Version command (version)" "$BIN" version
assert_exit 0 "Help flag (-h)" "$BIN" -h
assert_exit 0 "Help flag (--help)" "$BIN" --help

# 2. Usage Errors
assert_exit 2 "No arguments" "$BIN"
assert_exit 2 "Invalid option" "$BIN" --invalid-flag
assert_exit 2 "Color without arguments" "$BIN" color
assert_exit 2 "Dynamic without arguments" "$BIN" dynamic
assert_exit 2 "Dynamic single image error" "$BIN" dynamic /tmp/test_dynamic_apr/01_light.png
assert_json_val "code" "2" "JSON error code on usage error" "$BIN" dynamic -j
assert_json_val "success" "false" "JSON success=false on usage error" "$BIN" dynamic -j

# 3. List and UUID enumeration
assert_exit 0 "List displays" "$BIN" list -q
assert_json_field "uuid" "JSON list contains display UUID" "$BIN" list -j
assert_json_field "resolution" "JSON list contains display resolution" "$BIN" list -j
assert_json_field "wallpaper" "JSON list contains wallpaper path" "$BIN" list -j

# 4. Display targeting
FIRST_UUID="$("$BIN" list -j | awk -F'"' '/"uuid" *:/{print $4}' | head -1)"
if [[ -n "$FIRST_UUID" ]]; then
    assert_exit 0 "Get display by UUID ($FIRST_UUID)" "$BIN" get -d "$FIRST_UUID" -q
    assert_json_field "wallpaper" "Get display by UUID JSON" "$BIN" get -d "$FIRST_UUID" -j
fi

# 5. Dynamic HEIC builds
assert_exit 0 "Dynamic build (appearance)" "$BIN" dynamic /tmp/test_dynamic_apr --out /tmp/check_apr.heic -q
assert_json_val "schedule" "appearance" "JSON schedule mode for apr" "$BIN" dynamic /tmp/test_dynamic_apr --out /tmp/check_apr.heic -j
assert_json_val "frames" "2" "JSON frame count for apr" "$BIN" dynamic /tmp/test_dynamic_apr --out /tmp/check_apr.heic -j

assert_exit 0 "Dynamic build (time)" "$BIN" dynamic /tmp/test_dynamic_time --out /tmp/check_time.heic -q
assert_json_val "schedule" "time" "JSON schedule mode for time" "$BIN" dynamic /tmp/test_dynamic_time --out /tmp/check_time.heic -j

assert_exit 0 "Dynamic build (solar)" "$BIN" dynamic /tmp/test_dynamic_solar --out /tmp/check_solar.heic -q
assert_json_val "schedule" "solar" "JSON schedule mode for solar" "$BIN" dynamic /tmp/test_dynamic_solar --out /tmp/check_solar.heic -j
assert_json_val "frames" "16" "JSON frame count for solar" "$BIN" dynamic /tmp/test_dynamic_solar --out /tmp/check_solar.heic -j

# 6. Dry run
assert_exit 0 "Set dry run (-n)" "$BIN" set /tmp/check_solar.heic -n -q
assert_json_val "dry_run" "true" "JSON output on dry run" "$BIN" set /tmp/check_solar.heic -n -j

# Clean up test outputs
rm -f /tmp/check_apr.heic /tmp/check_time.heic /tmp/check_solar.heic

print "\nSuite finished: $PASSED passed, $FAILED failed."
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
exit 0
