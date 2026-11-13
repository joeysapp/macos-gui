#!/bin/zsh

set -e

DIR="${0:A:h}"
BIN="$DIR/tools/background-tool/background-tool"
SWIFT_SRC="$DIR/tools/background-tool/background-tool.swift"

# Auto-compile binary if not present
if [[ ! -x "$BIN" ]] && [[ -f "$SWIFT_SRC" ]]; then
    if command -v swiftc >/dev/null 2>&1; then
        print "Compiling native background-tool..." >&2
        swiftc -module-cache-path /tmp/mcache -O "$SWIFT_SRC" -o "$BIN"
    fi
fi

if [[ -x "$BIN" ]]; then
    exec "$BIN" "$@"
fi

# Fallback basic runner if swiftc was unavailable
print -u2 "error: Native binary not found at $BIN and swiftc is not installed."
exit 1
