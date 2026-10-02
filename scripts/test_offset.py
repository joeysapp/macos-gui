import sys
script = """on run argv
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
            
            -- Dynamically wait for checkboxes to appear (up to 5 seconds)
            set maxLoops to 50
            set checkBoxes to {}
            set foundCheckboxes to false
            
            repeat with i from 1 to maxLoops
                try
                    set scrollArea to scroll area 1 of group 1 of window 1
                    set checkBoxes to checkboxes of group 1 of scrollArea
                    if (count of checkBoxes) > 0 then
                        set foundCheckboxes to true
                        exit repeat
                    end if
                end try
                delay 0.1
            end repeat
            
            if not foundCheckboxes then
                try
                    key code 53
                end try
                return "Error: Could not find displays list. Is Wi-Fi/Bluetooth on, or are there no displays nearby?"
            end if
            
            set targetElem to missing value
            
            if theAction is "stop" and theTarget is "" then
                repeat with elem in checkBoxes
                    if value of elem is 1 then
                        set targetElem to elem
                        exit repeat
                    end if
                end repeat
            else
                try
                    set targetIndex to theTarget as integer
                    if targetIndex > 0 and targetIndex ≤ (count of checkBoxes) then
                        set targetElem to item targetIndex of checkBoxes
                    end if
                end try
                
                if targetElem is missing value then
                    repeat with elem in checkBoxes
                        set theID to value of attribute "AXIdentifier" of elem
                        if theID as string contains theTarget then
                            set targetElem to elem
                            exit repeat
                        end if
                    end repeat
                end if
            end if
            
            if targetElem is missing value then
                key code 53
                if theAction is "stop" and theTarget is "" then
                    return "No active screen mirroring session found to stop."
                else
                    return "Error: Could not find display with index or ID: " & theTarget
                end if
            end if
            
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
            key code 53
            return resultMsg
            
        end tell
    end tell
end run"""
print(script[550:650])
