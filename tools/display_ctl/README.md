# `display_ctl` - Native macOS Display Management CLI

A lightweight, high-performance native Swift CLI utility for managing macOS displays using Quartz CoreGraphics and AppKit APIs. Works seamlessly with both built-in, external, and AirPlay displays.

## Features

- **Mirror / Extend Controls**: Instantly mirror any secondary display to your main display or set it to an extended desktop without clicking through System Settings.
- **Resolution Management**: Inspect supported resolutions and refresh rates, and switch resolutions directly from terminal.
- **Display Inspection**: Query display hardware IDs, localized screen names, logical scaling, physical backing pixels, and current mirror master.
- **Fast Execution**: Built with native Quartz Display Services C/Swift APIs executing in < 10ms.

## Usage

```bash
# List all connected displays and their current mirror states
display_ctl list
display_ctl list -j

# Mirror external or AirPlay display to main display
display_ctl mirror
display_ctl mirror "LG" 0

# Set external or AirPlay display to extended desktop (independent)
display_ctl extend
display_ctl extend "LG"

# Toggle between mirrored and extended
display_ctl toggle

# View supported resolutions for a display
display_ctl resolutions "LG"

# Set resolution
display_ctl resolution "LG" 1920x1080 60
display_ctl resolution 2560x1440
```
