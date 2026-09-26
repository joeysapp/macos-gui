# macos-utils
Here are a handful of useful scripts I've made for stuff

## raise window
Until I can shout "Alexa raise Spotify to the top" I have to use this.

## airplay
Sometimes I lose Bluetooth connection, but most of the time when I do it's a result of something I did.

## screenshot
The builtin screenshot tool was missing the fuzzy-finding by title I think? I felt justified in this.

## automation
### Components
- Screen - Capture screenshots (Screen.capture(), Screen.size())
- OCR - Text detection via Tesseract (find_text(), find_exact(), find_fuzzy(), read_all())
- Templates - Visual element matching (match(), find_color_region())
- Mouse - Control via cliclick (move(), click(), drag(), scroll())
- Keyboard - Control via AppleScript (type(), press(), hotkey())
- GUI - High-level operations (click_on_text(), find_text(), wait_for_text(), type_in_field())

### Installation
I should just make a requirements.txt, but 
```zsh
brew install tesseract numpy pillow
```

### Usage
```py
from macos-gui import GUI, Mouse, Keyboard, Screen

gui = GUI()
gui.click_on_text("Submit")           # Find and click text
gui.wait_for_text("Success", timeout=10)  # Wait for text
Keyboard.hotkey('cmd', 'c')           # Copy
Mouse.move(100, 200)                  # Move mouse smoothly
```

### Interactive REPL
```py
 $ python3 macos-gui.py repl
```
