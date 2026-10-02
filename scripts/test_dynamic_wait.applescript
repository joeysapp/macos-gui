tell application "System Events"
    tell process "ControlCenter"
        click (first menu bar item of menu bar 1 whose description is "Screen Mirroring")
        
        set maxLoops to 50
        set foundCount to 0
        set elapsed to 0
        
        repeat with i from 1 to maxLoops
            try
                set foundCount to count of checkboxes of group 1 of scroll area 1 of group 1 of window 1
                if foundCount > 0 then
                    set elapsed to i
                    exit repeat
                end if
            end try
            delay 0.1
        end repeat
        
        key code 53
        return "Found " & foundCount & " checkboxes after " & elapsed & " loops."
    end tell
end tell
