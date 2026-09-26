#!/bin/zsh

set -e

NAME="mirror.sh"

usage() {
    cat <<HELP
Usage:
  $NAME start <id|index>
  $NAME stop [id|index]
  $NAME list
  $NAME status

Options:
  -h, --help       Show this help
  -v, --version    Show version

Examples:
  $NAME list
  $NAME start 4D:42:49:32:5D:B5
  $NAME start 1
  $NAME stop
HELP
}

die() {
    print -u2 "error: $*"
    exit 1
}

list_displays() {
    print "🔍 Scanning for AirPlay displays (this takes a couple seconds)..."
    
    local tmpfile
    tmpfile=$(mktemp)
    
    sh -c "dns-sd -B _airplay._tcp > \"$tmpfile\" & sleep 2; kill -9 \$! 2>/dev/null"
    
    local -a names
    while read -r line; do
        if [[ "$line" == *"_airplay._tcp."* && "$line" != *"STARTING"* ]]; then
            local name="${line#* _airplay._tcp. *}"
            name="${name#"${name%%[![:space:]]*}"}"
            if [[ -n "$name" ]]; then
                names+=("$name")
            fi
        fi
    done < "$tmpfile"
    rm -f "$tmpfile"
    
    local -A unique_names
    for n in "${names[@]}"; do
        unique_names[$n]=1
    done
    
    if (( ${#unique_names[@]} == 0 )); then
        print "No AirPlay displays found on the network."
        return
    fi
    
    print "🖥  Available Displays:"
    local i=1
    for name in ${(k)unique_names}; do
        local detail_tmp
        detail_tmp=$(mktemp)
        
        sh -c "dns-sd -L \"$name\" _airplay._tcp > \"$detail_tmp\" & sleep 1; kill -9 \$! 2>/dev/null"
        
        local deviceid=""
        local psi=""
        if grep -q "deviceid=" "$detail_tmp"; then
            deviceid=$(grep -o "deviceid=[^ ]*" "$detail_tmp" | head -1 | cut -d= -f2)
        fi
        if grep -q "psi=" "$detail_tmp"; then
            psi=$(grep -o "psi=[^ ]*" "$detail_tmp" | head -1 | cut -d= -f2)
        fi
        rm -f "$detail_tmp"
        
        local id_to_show="$deviceid"
        [[ -n "$psi" ]] && id_to_show="$psi"
        
        print "  [$i] $name"
        [[ -n "$id_to_show" ]] && print "      ID: $id_to_show"
        
        ((i++))
    done
}

toggle_mirroring() {
    local action="$1"
    local target="$2"
    
    osascript - "$action" "$target" <<'APPLESCRIPT'
on run argv
    set theAction to item 1 of argv
    set theTarget to item 2 of argv
    
    tell application "System Events"
        tell process "ControlCenter"
            try
                set menuBarItem to first menu bar item of menu bar 1 whose description is "Screen Mirroring"
            on error
                return "Error: Screen Mirroring menu bar item not found in Control Center."
            end try
            
            click menuBarItem
            
            set maxLoops to 50
            set foundCheckboxes to false
            
            repeat with i from 1 to maxLoops
                try
                    set scrollArea to scroll area 1 of group 1 of window 1
                    set currentCheckboxes to checkboxes of group 1 of scrollArea
                    if (count of currentCheckboxes) > 0 then
                        set foundCheckboxes to true
                        set targetElem to missing value
                        
                        if theAction is "stop" and theTarget is "" then
                            repeat with elem in currentCheckboxes
                                if value of elem is 1 then
                                    set targetElem to elem
                                    exit repeat
                                end if
                            end repeat
                        else
                            try
                                set targetIndex to theTarget as integer
                                if targetIndex > 0 and targetIndex ≤ (count of currentCheckboxes) then
                                    set targetElem to item targetIndex of currentCheckboxes
                                end if
                            end try
                            
                            if targetElem is missing value then
                                repeat with elem in currentCheckboxes
                                    set theID to value of attribute "AXIdentifier" of elem
                                    if theID as string contains theTarget then
                                        set targetElem to elem
                                        exit repeat
                                    end if
                                end repeat
                            end if
                        end if
                        
                        if targetElem is not missing value then
                            set isChecked to (value of targetElem) is 1
                            set elemID to value of attribute "AXIdentifier" of targetElem
                            
                            if theAction is "start" then
                                if isChecked then
                                    set resultMsg to "Already mirroring to " & elemID
                                else
                                    click targetElem
                                    set resultMsg to "Started mirroring to " & elemID
                                end if
                            else if theAction is "stop" then
                                if isChecked then
                                    click targetElem
                                    set resultMsg to "Stopped mirroring to " & elemID
                                else
                                    set resultMsg to "Not currently mirroring to " & elemID
                                end if
                            end if
                            
                            delay 0.2
                            try
                                key code 53
                            end try
                            return resultMsg
                        end if
                    end if
                end try
                delay 0.1
            end repeat
            
            try
                key code 53
            end try
            
            if not foundCheckboxes then
                return "Error: Could not find displays list. Is Wi-Fi/Bluetooth on, or are there no displays nearby?"
            else
                return "Error: Could not find display with index or ID: " & theTarget
            end if
            
        end tell
    end tell
end run
APPLESCRIPT
}

status_displays() {
    osascript <<'APPLESCRIPT'
    tell application "System Events"
        tell process "ControlCenter"
            try
                click (first menu bar item of menu bar 1 whose description is "Screen Mirroring")
                delay 0.5
                set checkBoxes to checkboxes of group 1 of scroll area 1 of group 1 of window 1
                set activeIDs to ""
                repeat with elem in checkBoxes
                    if value of elem is 1 then
                        set activeIDs to activeIDs & (value of attribute "AXIdentifier" of elem) & "\n"
                    end if
                end repeat
                key code 53
                
                if activeIDs is "" then
                    return "No active screen mirroring."
                else
                    return "Active mirroring sessions:\n" & activeIDs
                end if
            on error
                try
                    key code 53
                end try
                return "Error checking status."
            end try
        end tell
    end tell
APPLESCRIPT
}

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
    list)
        list_displays
        ;;
    start)
        (( $# >= 1 )) || die "start requires an ID or index"
        toggle_mirroring "start" "$1"
        ;;
    stop)
        toggle_mirroring "stop" "${1:-}"
        ;;
    status)
        status_displays
        ;;
    *)
        die "unknown command: $command"
        ;;
esac
