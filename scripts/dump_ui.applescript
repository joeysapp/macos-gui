tell application "System Events"
    tell process "ControlCenter"
        click (first menu bar item of menu bar 1 whose description is "Screen Mirroring")
        delay 2.0
        set ui_dump to ""
        try
            set ui_dump to ui_dump & "Window: " & name of window 1 & "\n"
            set allElements to entire contents of window 1
            repeat with elem in allElements
                try
                    set ui_dump to ui_dump & (class of elem) & " | name: " & (name of elem) & " | desc: " & (description of elem) & "\n"
                end try
            end repeat
        on error errMsg
            set ui_dump to "Error dumping UI: " & errMsg
        end try
        key code 53
        return ui_dump
    end tell
end tell
