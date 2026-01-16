# macos-gui.py
General-purpose macOS GUI automation library using OCR and fuzzy searching. 
I know I need to separate this out from one file someday.

# Requirements
`brew install tesseract numpy pillow` 

# Components:
- Screen - Capture screenshots (Screen.capture(), Screen.size())
- OCR - Text detection via Tesseract (find_text(), find_exact(), find_fuzzy(), read_all())
- Templates - Visual element matching (match(), find_color_region())
- Mouse - Control via cliclick (move(), click(), drag(), scroll())
- Keyboard - Control via AppleScript (type(), press(), hotkey())
- GUI - High-level operations (click_on_text(), find_text(), wait_for_text(), type_in_field())

Usage:
```py
from macos-gui import GUI, Mouse, Keyboard, Screen

gui = GUI()
gui.click_on_text("Submit")           # Find and click text
gui.wait_for_text("Success", timeout=10)  # Wait for text
Keyboard.hotkey('cmd', 'c')           # Copy
Mouse.move(100, 200)                  # Move mouse smoothly
```

Interactive REPL:
```py
 $ python3 macos-gui.py repl
```
