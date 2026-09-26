#!/bin/zsh

set -e

NAME="wallpaper"

usage() {
    cat <<EOF
Usage:
  $NAME <image>
  $NAME set <image>
  $NAME random <directory>
  $NAME rotate <directory> [interval]
  $NAME spaces <image|directory>

Options:
  -h, --help       Show this help
  -v, --version    Show version
  -n, --dry-run    Print what would happen without changing anything

Examples:
  $NAME ~/Pictures/foo.jpg
  $NAME random ~/Pictures/Wallpapers
  $NAME rotate ~/Pictures/Wallpapers 30m
  $NAME spaces ~/Pictures/Wallpapers

Intervals:
  30s   30 seconds
  5m    5 minutes
  1h    1 hour
  1d    1 day
EOF
}

die() {
    print -u2 "error: $*"
    exit 1
}

# Convert e.g. 30s / 5m / 1h / 1d to seconds.
parse_interval() {
    local value="$1"

    if [[ "$value" =~ '^([0-9]+)(s|m|h|d)$' ]]; then
        local number="${match[1]}"
        local unit="${match[2]}"

        case "$unit" in
            s) echo "$number" ;;
            m) echo "$((number * 60))" ;;
            h) echo "$((number * 3600))" ;;
            d) echo "$((number * 86400))" ;;
        esac
    else
        die "invalid interval '$value' (try 30s, 5m, 1h, or 1d)"
    fi
}

set_wallpaper() {
    local image="$1"

    [[ -f "$image" ]] || die "not a file: $image"

    image="$(realpath "$image")"

    print "🖼  $image"

    if (( DRY_RUN )); then
        return
    fi

    osascript <<EOF
tell application "System Events"
    tell every desktop
        set picture to POSIX file "$image"
    end tell
end tell
EOF
}

random_image() {
    local directory="$1"

    [[ -d "$directory" ]] || die "not a directory: $directory"

    local -a images
    images=(
        "$directory"/*.jpg(N)
        "$directory"/*.jpeg(N)
        "$directory"/*.png(N)
        "$directory"/*.heic(N)
        "$directory"/*.webp(N)
    )

    (( ${#images} )) || die "no images found in $directory"

    # zsh's RANDOM is sufficient for wallpaper selection.
    set_wallpaper "${images[$((RANDOM % ${#images} + 1))]}"
}

# Rotations

rotate_native() {
    local directory="$1"
    local interval="$2"

    [[ -d "$directory" ]] || die "not a directory: $directory"

    local seconds
    seconds="$(parse_interval "$interval")"

    print "🖼  Wallpaper rotation"
    print "📁  $directory"
    print "⏱  every $interval"
    print "🎲  random order"

    (( DRY_RUN )) && return

    directory="$(realpath "$directory")"

    osascript <<EOF
tell application "System Events"
    tell every desktop
        set pictures folder to POSIX file "$directory"
        set random order to true
        set change interval to $seconds
        set picture rotation to 1
    end tell
end tell
EOF
}

random_forever() {
    local directory="$1"
    local interval="$2"

    local seconds
    seconds="$(parse_interval "$interval")"

    [[ "$seconds" -gt 0 ]] || die "interval must be greater than zero"

    print "🎲 Rotating wallpapers from:"
    print "   $directory"
    print "⏱  every $interval"
    print
    print "Press Ctrl-C to stop."

    while true; do
        random_image "$directory"
        sleep "$seconds"
    done
}

set_spaces() {
    local source="$1"

    if [[ -d "$source" ]]; then
        local -a images
        images=(
            "$source"/*.jpg(N)
            "$source"/*.jpeg(N)
            "$source"/*.png(N)
            "$source"/*.heic(N)
            "$source"/*.webp(N)
        )

        (( ${#images} )) || die "no images found in $source"

        local desktop_count
        desktop_count="$(
            osascript -e '
                tell application "System Events"
                    count of desktops
                end tell
            '
        )"

        print "🖥  Found $desktop_count desktops"

        local i=1
        for image in "${images[@]}"; do
            (( i > desktop_count )) && break

            image="$(realpath "$image")"

            print "   Space $i ← $image"

            if (( ! DRY_RUN )); then
                osascript -e "
                    tell application \"System Events\"
                        set picture of desktop $i to POSIX file \"$image\"
                    end tell
                "
            fi

            (( i++ ))
         done
    else
        # Same image everywhere.
        set_wallpaper "$source"
    fi
}

DRY_RUN=0

# Global options
while (( $# )); do
    case "$1" in
        -h|--help)
            usage
            exit
            ;;
        -v|--version)
            print "$NAME 0.1.0"
            exit
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

(( $# )) || {
    usage
    exit 1
}

command="$1"
shift

case "$command" in
    set)
        (( $# == 1 )) || die "set requires an image"
        set_wallpaper "$1"
        ;;

    random)
        (( $# == 1 )) || die "random requires a directory"
        random_image "$1"
        ;;

    rotate)
        (( $# >= 1 && $# <= 2 )) || die "rotate requires <directory> [interval]"
        rotate_native "$1" "${2:-30m}"
        ;;

    spaces)
        (( $# == 1 )) || die "spaces requires an image or directory"
        set_spaces "$1"
        ;;

    *)
        # Friendly shorthand:
        # wallpaper foo.jpg
        # wallpaper ~/Pictures/Wallpapers
        if [[ -f "$command" ]]; then
            set_wallpaper "$command"
        elif [[ -d "$command" ]]; then
            random_image "$command"
        else
            die "unknown command or path: $command"
        fi
        ;;
esac
