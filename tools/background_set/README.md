# background_set

A fast, native macOS desktop wallpaper management tool written in Swift using Cocoa/AppKit `NSWorkspace`.

## Features
- **Display Targeting**: Target all connected displays (default) or individual screens by index (`0`, `1`), display ID (`1`, `23091`), or name (`built-in`, `lg`).
- **Scaling Modes**:
  - `fill`: Aspect fill (zoom/crop to cover screen without distortion) [default]
  - `fit`: Aspect fit (entire image visible, bordered with matte background color)
  - `stretch`: Stretch axes independently to fill screen
  - `center`: 1:1 original pixel size centered on screen
- **Custom Matte & Solid Background Color**:
  - Accepts hex codes (`#1e1e1e`, `#ffffff`, `#333`) and named colors (`black`, `white`, `gray`, `red`, etc.).
  - Can set solid background colors directly without needing an image (`background_set color "#0d1117"`).
- **Multi-Display & Spaces**:
  - `spaces`: Assign distinct images across connected monitors.
  - Granular control over multi-monitor configurations.
- **Random & Rotation**:
  - `random <dir>`: Select a random wallpaper from a folder.
  - `rotate <dir> [interval]`: Cycle wallpapers automatically at intervals (e.g. `30s`, `5m`, `1h`).
- **Display & Status Inspection**:
  - `list` / `status`: View all detected displays, IDs, resolutions, current image, mode, and fill color.
  - `get`: Query current wallpaper path.
- **Zsh Autocompletion**: Bundled `_background_set` completion file.

## Build
Compile directly with Apple Swift:
```sh
swiftc -O background_set.swift -o background_set
```

## Usage Examples

### Basic Wallpaper Setting
```sh
# Set on all displays (default)
background_set ~/Pictures/Wallpapers/mountain.jpg

# Set using explicit subcommand
background_set set ~/Pictures/Wallpapers/mountain.jpg
```

### Modes & Background Colors
```sh
# Fit image with dark matte background border
background_set ~/Pictures/art.png --mode fit --color "#121212"

# Center small pixel art at 1:1 scale on navy background
background_set ~/Pictures/retro.png --mode center --color "#0b1d3a"

# Set a solid dark color wallpaper across all screens
background_set color "#0d1117"
```

### Display Targeting
```sh
# List all connected displays and current settings
background_set list

# Apply only to primary display (index 0)
background_set ~/Pictures/code.png --display 0

# Apply to external monitor by name match
background_set ~/Pictures/wide.jpg --display "lg" --mode fill
```

### Rotation & Multiple Displays
```sh
# Random wallpaper from folder
background_set random ~/Pictures/Wallpapers

# Continuous rotation every 15 minutes
background_set rotate ~/Pictures/Wallpapers 15m --mode fill

# Distribute images across monitors
background_set spaces ~/Pictures/Monitors/
```
