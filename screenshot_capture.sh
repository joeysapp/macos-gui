#!/bin/zsh

# screenshot
#
# A small macOS screenshot utility with:
#   - app/window targeting
#   - display targeting
#   - window shadow control
#   - interactive fallback
#   - sensible output naming
#
# Requires: macOS, zsh, osascript/JXA, screencapture

set -e

NAME="screenshot"
VERSION="0.1.0"

usage() {
    cat <<EOF
Usage:
  $NAME [options]

Capture:
  -a, --app APP       Capture the front window of APP
  -w, --window NAME   Capture a window by title
  -D, --display N     Capture display N
  -s, --screen        Capture the entire main screen
  -S, --space         Capture the current Space
  -i, --interactive   Select a window interactively

Output:
  -o, --output PATH   Output file
  -d, --directory DIR Output directory
  -n, --no-shadow     Don't include window shadow
  -f, --format TYPE   png, jpg, tiff, pdf

Options:
  -h, --help          Show this help
  -v, --version       Show version
  -q, --quiet         Suppress status output
  -p, --print-path    Print resulting file path

Examples:
  $NAME
  $NAME -s
  $NAME -a Safari
  $NAME -a "Google Chrome" -o ~/Desktop/chrome.png
  $NAME -w "Terminal" -o ~/Desktop/terminal.png
  $NAME -D 2
  $NAME -i -o ~/Desktop/window.png
  $NAME -a Finder -n

If no capture target is specified, interactive window selection is used.

EOF
}

die() {
    print -u2 "error: $*"
    exit 1
}

log() {
    (( QUIET )) || print "$@"
}

timestamp() {
    date '+%Y-%m-%d_%H-%M-%S'
}

# ---------------------------------------------------------------------------
# Window lookup
# ---------------------------------------------------------------------------

# Ask CoreGraphics for the actual CGWindowID.
#
# This is the useful bit: System Events knows about applications/windows,
# but screencapture wants a CGWindowID. JXA gives us a relatively painless
# bridge to CoreGraphics without compiling anything.
#
# Match order:
#   1. exact owner + window title
#   2. exact owner
#   3. case-insensitive owner
#
# For --app, the largest/most prominent window is preferred.

window_id_for_app() {
    local app="$1"
    app="${app%.app}"

    osascript -l JavaScript - "$app" <<'JXA'
ObjC.import('CoreGraphics')

const args = ObjC.deepUnwrap(
    $.NSProcessInfo.processInfo.arguments
)

const app = String(args[args.length - 1]).toLowerCase()

const options =
    $.kCGWindowListOptionOnScreenOnly |
    $.kCGWindowListExcludeDesktopElements

const info = ObjC.castRefToObject($.CGWindowListCopyWindowInfo(
    options,
    $.kCGNullWindowID
))

const count = info.count
const matches = []

for (let i = 0; i < count; i++) {
    const w = info.objectAtIndex(i)

    const owner = ObjC.unwrap(w.objectForKey('kCGWindowOwnerName') || '')
        .toLowerCase()

    const layer = Number(
        ObjC.unwrap(w.objectForKey('kCGWindowLayer') || 0)
    )

    const alpha = Number(
        ObjC.unwrap(w.objectForKey('kCGWindowAlpha') || 0)
    )

    const bounds = w.objectForKey('kCGWindowBounds')

    let width = 0
    let height = 0

    if (bounds) {
        width = Number(ObjC.unwrap(bounds.objectForKey('Width') || 0))
        height = Number(ObjC.unwrap(bounds.objectForKey('Height') || 0))
    }

    if (
        owner.includes(app) &&
        layer === 0 &&
        alpha > 0 &&
        width > 100 &&
        height > 100
    ) {
        const id = ObjC.unwrap(
            w.objectForKey('kCGWindowNumber')
        )

        matches.push({
            id: Number(id),
            area: width * height
        })
    }
}

matches.sort((a, b) => b.area - a.area)

if (matches.length) {
    matches[0].id
} else {
    ""
}
JXA
}

window_id_for_title() {
    local title="$1"

    osascript -l JavaScript - "$title" <<'JXA'
ObjC.import('CoreGraphics')

const args = ObjC.deepUnwrap(
    $.NSProcessInfo.processInfo.arguments
)

const title = String(args[args.length - 1]).toLowerCase()

const options =
    $.kCGWindowListOptionOnScreenOnly |
    $.kCGWindowListExcludeDesktopElements

const info = ObjC.castRefToObject($.CGWindowListCopyWindowInfo(
    options,
    $.kCGNullWindowID
))

const windows = ObjC.deepUnwrap(info)

const matches = windows.filter(w =>
    String(w.kCGWindowName || '').toLowerCase().includes(title) &&
    Number(w.kCGWindowLayer || 0) === 0 &&
    Number(w.kCGWindowAlpha || 0) > 0
)

if (matches.length) {
    matches[0].kCGWindowNumber
} else {
    ""
}
JXA
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT=""
DIRECTORY="$HOME/Desktop"
FORMAT="png"

make_output_path() {
    if [[ -n "$OUTPUT" ]]; then
        print -r -- "$OUTPUT"
        return
    fi

    print -r -- "$DIRECTORY/screenshot_$(timestamp).$FORMAT"
}

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

capture() {
    local output="$1"
    shift

    mkdir -p "$(dirname "$output")"

    local -a args
    args=(-x)

    # -o = don't include window shadow
    (( NO_SHADOW )) && args+=(-o)

    # Format
    case "$FORMAT" in
        png)  args+=(-t png) ;;
        jpg)  args+=(-t jpg) ;;
        jpeg) args+=(-t jpg) ;;
        tiff) args+=(-t tiff) ;;
        pdf)  args+=(-t pdf) ;;
        *) die "unsupported format: $FORMAT" ;;
    esac

    args+=("$@")
    args+=("$output")

    log "📸  $output"

    local has_l=0
    for arg in "${args[@]}"; do
        if [[ "$arg" == "-l" ]]; then
            has_l=1
            break
        fi
    done

    if (( has_l )); then
        # Workaround for macOS bug where screencapture -l hangs after writing
        screencapture "${args[@]}" &
        local pid=$!
        
        local attempts=100
        local last_size=0
        local stable_count=0
        
        while (( attempts-- > 0 )); do
            if [[ -f "$output" ]]; then
                local current_size=$(stat -f "%z" "$output" 2>/dev/null || echo 0)
                if (( current_size > 0 && current_size == last_size )); then
                    (( stable_count++ ))
                    if (( stable_count >= 3 )); then
                        break
                    fi
                else
                    last_size=$current_size
                    stable_count=0
                fi
            fi
            sleep 0.1
        done
        
        kill $pid 2>/dev/null || true
        wait $pid 2>/dev/null || true
    else
        screencapture "${args[@]}"
    fi

    [[ -f "$output" ]] || die "screenshot failed (could not find window?)"

    if (( PRINT_PATH )); then
        print -r -- "$output"
    fi
}

capture_app() {
    local app="$1"
    local id

    log "🔎  Looking for $app..."

    id="$(window_id_for_app "$app" | tail -1)"

    [[ -n "$id" ]] ||
        die "couldn't find a visible window for '$app'"

    log "🪟  $app → window $id"

    capture "$OUTPUT_PATH" -l "$id"
}

capture_window() {
    local title="$1"
    local id

    log "🔎  Looking for window '$title'..."

    id="$(window_id_for_title "$title" | tail -1)"

    [[ -n "$id" ]] ||
        die "couldn't find window '$title'"

    log "🪟  '$title' → window $id"

    capture "$OUTPUT_PATH" -l "$id"
}

capture_interactive() {
    log "🖱   Select a window..."

    capture "$OUTPUT_PATH" -w
}

capture_screen() {
    capture "$OUTPUT_PATH" -m
}

capture_display() {
    local display="$1"

    [[ "$display" =~ '^[0-9]+$' ]] ||
        die "display must be a number"

    capture "$OUTPUT_PATH" -D "$display"
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

QUIET=0
PRINT_PATH=0
NO_SHADOW=0

TARGET=""
TARGET_VALUE=""

while (( $# )); do
    case "$1" in
        -h|--help)
            usage
            exit
            ;;

        -v|--version)
            print "$NAME $VERSION"
            exit
            ;;

        -q|--quiet)
            QUIET=1
            shift
            ;;

        -p|--print-path)
            PRINT_PATH=1
            shift
            ;;

        -a|--app)
            (( $# >= 2 )) || die "--app requires an application name"
            TARGET="app"
            TARGET_VALUE="$2"
            shift 2
            ;;

        -w|--window)
            (( $# >= 2 )) || die "--window requires a window title"
            TARGET="window"
            TARGET_VALUE="$2"
            shift 2
            ;;

        -D|--display)
            (( $# >= 2 )) || die "--display requires a display number"
            TARGET="display"
            TARGET_VALUE="$2"
            shift 2
            ;;

        -s|--screen)
            TARGET="screen"
            shift
            ;;

        -S|--space)
            TARGET="space"
            shift
            ;;

        -i|--interactive)
            TARGET="interactive"
            shift
            ;;

        -n|--no-shadow)
            NO_SHADOW=1
            shift
            ;;

        -o|--output)
            (( $# >= 2 )) || die "--output requires a path"
            OUTPUT="$2"
            shift 2
            ;;

        -d|--directory)
            (( $# >= 2 )) || die "--directory requires a path"
            DIRECTORY="$2"
            shift 2
            ;;

        -f|--format)
            (( $# >= 2 )) || die "--format requires a format"
            FORMAT="${2:l}"
            shift 2
            ;;

        --)
            shift
            break
            ;;

        -*)
            die "unknown option: $1"

            ;;

        *)
            die "unexpected argument: $1"
            ;;
    esac
done

OUTPUT_PATH="$(make_output_path)"

case "$TARGET" in
    app)
        capture_app "$TARGET_VALUE"
        ;;

    window)
        capture_window "$TARGET_VALUE"
        ;;

    display)
        capture_display "$TARGET_VALUE"
        ;;

    screen)
        capture_screen
        ;;

    interactive)
        capture_interactive
        ;;

    space)
        # macOS screencapture captures the currently visible Space.
        capture_screen
        ;;

    "")
        capture_interactive
        ;;

    *)
        die "invalid target"
        ;;
esac
