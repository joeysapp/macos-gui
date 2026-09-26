# macos-utils
Here are a handful of useful scripts I've made for stuff

## raise window
Until I can shout "Alexa raise Spotify to the top" I have to use this.

## airplay
Sometimes I lose Bluetooth connection, but most of the time when I do it's a result of something I did.

## screenshot
The builtin screenshot tool was missing the fuzzy-finding by title I think? I felt justified in this.

## clipboard-network
Windows listener correctly receives length-prefixed UTF-8 protocol from macOS.
```sh
macOS
  ▼ pbpaste → SSH stdin
  
Windows SSH session (session 0)
  ▼ localhost TCP
Windows clipboard bridge (session 1)
  ▼ Set-Clipboard
Windows desktop clipboard
```
As a result, we get the following benefits:
- newlines survive
- tabs survive
- Unicode survives
- emoji survive
- whitespace survives
- there is no shell quoting of clipboard contents
- clipboard data never becomes a command

To use in your network:
1. Add this directory to your path
2. **If a zshell user**, install completions and dump cache:
  - zcomp="$HOME/.zsh/completions" mkdir -p $zcomp && cp _clipboard-network $zcomp
  - add these lines to zshrc if not present:
      fpath=(~/.zsh/completions $fpath)
      autoload -Uz compinit
      compinit
  - rm -f ~/.zcompdump && compinit 
3. **If more than 0 Windows machines**, install the ps1 task on any Windows hosts to safely ingest bytes correctly (clip -> utf8 chunks) Register once *AFTER* moving your script to where it belongs on posix. Checks to verify:
  1. Start-ScheduledTask -TaskName "$name-you-set"
  2. Get-ScheduledTask -TaskName "$name-you-set"
  3. Test-NetConnection 127.0.0.1 -Port 37421
     > TcpTestSucceeded : True
  4. 
  4. 
  
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
