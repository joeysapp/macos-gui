raise_app() {
    local app="$1"
    app="${app%.app}"

    osascript - "$app" <<'APPLESCRIPT'
on run argv
    set appName to item 1 of argv

    tell application "System Events"
        if not (exists process appName) then
            return false
        end if

        tell process appName
            set frontmost to true

            if (count of windows) > 0 then
                try
                    perform action "AXRaise" of window 1
                end try
            end if
        end tell
    end tell

    return true
end run
APPLESCRIPT
}


capture_app() {
    local app="$1"
    app="${app%.app}"

    log "🔎  Looking for $app..."

    raise_app "$app" >/dev/null
    sleep 0.15

    local id
    id="$(window_id_for_app "$app" | tail -1)"

    [[ -n "$id" ]] ||
        die "couldn't find a visible window for '$app'"

    log "🪟  $app → window $id"

    capture "$OUTPUT_PATH" -l "$id"
}

capture_app "Spotify.app"
