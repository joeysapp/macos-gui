tell application "System Events"
    tell process "ControlCenter"
        click (first menu bar item of menu bar 1 whose description is "Screen Mirroring")
        delay 1.0
        
        set checkBoxes to checkboxes of group 1 of scroll area 1 of group 1 of window 1
        
        -- Now wait 1 second
        delay 1.0
        
        -- And try to count them
        set c to count of checkBoxes
        
        key code 53
        return "Count: " & c
    end tell
end tell
