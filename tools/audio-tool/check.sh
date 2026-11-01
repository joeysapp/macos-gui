#!/bin/zsh -f
# check.sh - regression checks for audio-tool's failure paths.
#
# Each case must exit with its expected code and print one JSON error object, must not crash, and
# must leave the files as it found them. Every case fails before a device is opened: nothing is
# played or recorded. By default the cases run on an AddressSanitizer build made in a temp dir, so
# a memory bug (like 2026-11-01's double free in WavWriter.init) exits 86 with a report instead of
# crashing, since a crash pops macOS's "quit unexpectedly" dialog on screen.

HERE=${0:A:h}
TOOL=audio-tool
SRC=$HERE/$TOOL.swift
PLIST=$HERE/Info.plist
ASAN_EXIT=86

usage() {
    cat <<EOF
Usage:
  tools/audio-tool/check.sh [--binary <path>] [-j] [-v]

Runs audio-tool's failure paths and checks each one's exit code, its JSON error, that it didn't
crash, and that it created or changed no file. Nothing is played or recorded.

Cases: record --out that exists (no --force), is read-only (--force), is a dangling symlink, is in
a missing or read-only directory, is a directory, has an over-long name, isn't .wav; record or level
from a device without inputs; a channel the device lacks; play of a missing, directory, non-audio
or truncated file; play to a device without outputs; a target that isn't there.

Options:
  --binary <path>    Run this build as is, instead of an AddressSanitizer copy built from
                     tools/audio-tool (a crash there is a real crash, with its dialog)
  -j, --json         {"binary":"...","asan":true,"passed":18,"failed":0,"skipped":0,
                      "cases":[{"case":"...","ok":true,"exit":3,"want":"3","why":null}],"crash_reports":[]}
  -v, --verbose      Each case's command, stdout and stderr on stderr
  -h, --help         Show this help (also: help)

Exit codes: 0 every case passed (skips allowed), 1 a case failed, 2 usage.
A case is skipped when this Mac has no device of the kind it needs (e.g. output-only).
EOF
}

json=0 verbose=0 binary="" asan=0
while (( $# )); do
    case $1 in
        -h|--help|help) usage; exit 0 ;;
        -j|--json) json=1 ;;
        -v|--verbose) verbose=1 ;;
        -jv|-vj) json=1; verbose=1 ;;
        --binary) shift; binary=$1; [[ -n $binary ]] || { print -u2 "error: --binary needs a path"; exit 2 } ;;
        --binary=*) binary=${1#--binary=} ;;
        *) print -u2 "error: unknown argument '$1'"; print -u2 "Run '${0:t} --help' for usage."; exit 2 ;;
    esac
    shift
done

note() { (( json && ! verbose )) || print -u2 -- "$*" }

tmp=$(mktemp -d "${${TMPDIR:-/tmp}%/}/$TOOL-check.XXXXXX") || exit 1
cleanup() { chmod -R u+w "${tmp:?}" 2>/dev/null; rm -rf "${tmp:?}" }
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ -z $binary ]]; then
    note "Building an AddressSanitizer copy of $TOOL..."
    mkdir -p "$tmp/bin"
    if ! swiftc -Onone -sanitize=address "$SRC" -o "$tmp/bin/$TOOL" -Xlinker -sectcreate -Xlinker __TEXT \
            -Xlinker __info_plist -Xlinker "$PLIST" > "$tmp/build.log" 2>&1; then
        print -u2 "error: the AddressSanitizer build failed:"
        cat "$tmp/build.log" >&2
        exit 1
    fi
    binary=$tmp/bin/$TOOL
    asan=1
fi
[[ -x $binary ]] || { print -u2 "error: $binary is not an executable"; exit 2 }
binary=${binary:A}
# Sanitizer reports and fatal signals exit $ASAN_EXIT through _exit: no crash, no dialog.
export ASAN_OPTIONS="abort_on_error=0:halt_on_error=1:exitcode=$ASAN_EXIT:detect_leaks=0:handle_abort=1:handle_sigill=1:handle_sigtrap=1:handle_sigfpe=1:handle_segv=1:handle_sigbus=1"

# Devices by kind, from the binary under test (uids). The defaults come first, so that if a case
# ever did reach a device it would be the one already in use, not e.g. an iPhone's microphone.
picks=$("$binary" list -j 2>/dev/null | python3 -I -c '
import json, sys
d = json.load(sys.stdin)["devices"]
d.sort(key=lambda x: not (x["default_input"] or x["default_output"]))
first = lambda f: next((x["uid"] for x in d if f(x)), "")
print(first(lambda x: x["inputs"] > 0))
print(first(lambda x: x["inputs"] > 0 and x["outputs"] == 0))
print(first(lambda x: x["outputs"] > 0 and x["inputs"] == 0))' 2>/dev/null)
lines=("${(@f)picks}")
IN=${lines[1]-} INONLY=${lines[2]-} OUTONLY=${lines[3]-}

# Fixtures.
w=$tmp/work
mkdir -p "$w/rodir" "$w/dir.wav"
print -n "not a recording" > "$w/exists.wav"
print -n "read-only bytes" > "$w/readonly.wav"; chmod 444 "$w/readonly.wav"
chmod 555 "$w/rodir"
ln -s "$w/nowhere/target.wav" "$w/dangling.wav"
print "plain text, not audio" > "$w/text.wav"
printf 'RIFF\044\000\000\000WAVEfmt ' > "$w/truncated.wav"

# Name, size, mode, mtime and content of everything under the work dir (links themselves, not targets).
snapshot() {
    (cd "$w" && find . -mindepth 1 | LC_ALL=C sort | while IFS= read -r f; do
        if [[ -f $f && ! -L $f ]]; then print -r -- "$(stat -f '%N %z %p %Fm' "$f") $(md5 -q "$f")"
        else stat -f '%N %z %p %Fm' "$f"; fi
    done)
}
# A JSON string literal.
jstr() {
    local s=${1//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\t'/\\t}
    print -rn -- "\"$s\""
}
reports() { print -l ~/Library/Logs/DiagnosticReports/$TOOL-*.ips(N) }

passed=0 failed=0 skipped=0
typeset -a rows
reports_before=$(reports)

# check <case> <wanted exit codes> <needs> -- <args...>
check() {
    local name=$1 want=$2 need=$3; shift 4
    if [[ $need != - && -z ${(P)need} ]]; then
        (( skipped++ ))
        rows+=("{\"case\":$(jstr "$name"),\"ok\":null,\"exit\":null,\"want\":\"$want\",\"why\":\"no device of that kind\"}")
        (( json )) || print "skip  $name (no device of that kind)"
        return
    fi
    local before=$(snapshot) code why=""
    (cd "$w" && exec "$binary" "$@" -j) > "$tmp/out" 2> "$tmp/err" &
    local pid=$! ticks=0
    while kill -0 $pid 2>/dev/null && (( ticks < 300 )); do sleep 0.1; (( ticks++ )); done
    if kill -0 $pid 2>/dev/null; then
        kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; code=124; why="timed out after 30 s"
    else
        wait $pid; code=$?
    fi
    if [[ -z $why ]]; then
        if (( code == ASAN_EXIT )); then why="AddressSanitizer caught a memory error or fatal signal (see -v)"
        elif (( code >= 128 )); then why="crashed: signal $(( code - 128 ))"
        elif [[ " $want " != *" $code "* ]]; then why="exit $code, want $want"
        elif ! python3 -I -c '
import json, sys
lines = [l for l in open(sys.argv[1]) if l.strip()]
o = json.loads(lines[0]) if len(lines) == 1 else {}
sys.exit(0 if o.get("success") is False and o.get("error") else 1)' "$tmp/out" 2>/dev/null; then
            why="stdout isn't one JSON error object"
        elif [[ $(snapshot) != "$before" ]]; then why="files changed (created, modified or removed)"
        fi
    fi
    if (( verbose )); then
        print -u2 -- "--- $name: $TOOL $* -j  (exit $code)"
        sed 's/^/    out: /' "$tmp/out" >&2
        sed 's/^/    err: /' "$tmp/err" >&2
    fi
    if [[ -z $why ]]; then
        (( passed++ ))
        rows+=("{\"case\":$(jstr "$name"),\"ok\":true,\"exit\":$code,\"want\":\"$want\",\"why\":null}")
        (( json )) || print "ok    $name (exit $code)"
    else
        (( failed++ ))
        rows+=("{\"case\":$(jstr "$name"),\"ok\":false,\"exit\":$code,\"want\":\"$want\",\"why\":$(jstr "$why")}")
        (( json )) || print "FAIL  $name: $why"
    fi
}

long=$(printf 'a%.0s' {1..300}).wav
check "record: --out exists, no --force"          "3" IN -- record --from "$IN" -s 1 -o "$w/exists.wav"
check "record: --out read-only, --force"          "1" IN -- record --from "$IN" -s 1 -o "$w/readonly.wav" --force
check "record: --out is a dangling symlink"       "1" IN -- record --from "$IN" -s 1 -o "$w/dangling.wav"
check "record: --out in a missing directory"      "1" IN -- record --from "$IN" -s 1 -o "$w/missing/x.wav"
check "record: --out in a read-only directory"    "1" IN -- record --from "$IN" -s 1 -o "$w/rodir/x.wav"
check "record: --out is a directory"              "1" IN -- record --from "$IN" -s 1 -o "$w/dir.wav"
check "record: --out name too long"               "1" IN -- record --from "$IN" -s 1 -o "$w/$long"
check "record: --out isn't .wav"                  "2" IN -- record --from "$IN" -s 1 -o "$w/x.mp3"
check "record: channel the device lacks"          "3" IN -- record --from "$IN" -c 99 -s 1 -o "$w/x.wav"
check "record: device without inputs"             "3" OUTONLY -- record --from "$OUTONLY" -s 1 -o "$w/x.wav"
check "level: device without inputs"              "3" OUTONLY -- level --from "$OUTONLY" -s 1
check "play: missing file"                        "1" - -- play "$w/none.wav"
check "play: a directory"                         "1" - -- play "$w/dir.wav"
check "play: not audio"                           "1" - -- play "$w/text.wav"
check "play: truncated WAV"                       "1" - -- play "$w/truncated.wav"
check "play: device without outputs"              "3" INONLY -- play "$w/text.wav" --to "$INONLY"
check "target: no such device"                    "1" - -- level --from no-such-device-xyz -s 1
check "target: index out of range"                "1" - -- level --from 999 -s 1

new_reports=()
for r in ${(f)"$(reports)"}; do [[ $'\n'$reports_before$'\n' == *$'\n'$r$'\n'* ]] || new_reports+=("$r"); done
if (( ${#new_reports} )); then
    (( failed++ ))
    (( json )) || print "FAIL  new crash report(s): ${(j:, :)new_reports}"
fi

if (( json )); then
    local_reports=()
    for r in $new_reports; do local_reports+=("$(jstr "$r")"); done
    print -r -- "{\"binary\":$(jstr "$binary"),\"asan\":$( (( asan )) && print true || print false),\"passed\":$passed,\"failed\":$failed,\"skipped\":$skipped,\"cases\":[${(j:,:)rows}],\"crash_reports\":[${(j:,:)local_reports}]}"
else
    print "$passed passed, $failed failed, $skipped skipped ($([[ $asan == 1 ]] && print "AddressSanitizer build" || print "$binary"))"
fi
(( failed == 0 ))
