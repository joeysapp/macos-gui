# macOS GUI Tools Suite

A collection of lightweight, high-performance CLI tools for macOS display management, screen mirroring, automation, and system interaction.

Each utility follows the Unix philosophy—modular, independent single-purpose tools with structured JSON output (`-j`, `--json`), terse documentation, and Zsh autocompletions.

---

## Suite Components

### 1. `airplay-tool` (aliases: `airplay`, `airplay_connect`)
High-performance macOS AirPlay Screen Mirroring CLI.
- **Fast Concurrent Discovery**: Resolves mDNS TXT records in parallel via Bonjour (`_airplay._tcp`), cutting discovery time to ~1 second.
- **Smart Toggle (`airplay-tool toggle`)**: Disconnects active sessions or dynamically reconnects to the last-used host with configurable network discovery wait (`--timeout 8`).
- **Flexible Target Resolution**: Connect by 1-based index (`1`, `2`), friendly alias (`tv`), name substring (`LG`, `torus`), MAC address, or UUID.
- **Friendly Aliases**: Persistent shorthand nicknames (`airplay-tool alias set tv "LG"`).
- **Auto-Reconnect Watchdog (`airplay-tool watch "LG"`)**: Keeps flaky connections alive (e.g. LG OLED TVs).
- **Diagnostics & Logs (`airplay-tool logs`)**: Queries macOS Unified Log for socket teardowns, timeout reasons, and connection metrics.
- **JSON Output**: Full machine-readable `-j` / `--json` support across all subcommands.

### 2. `background-set` (alias: `background_set`)
Native Swift wallpaper management tool using Cocoa/AppKit `NSWorkspace`.
- **Display Targeting**: Target all connected displays (default) or individual screens (`-d 0`, `-d "LG"`).
- **Scaling Modes**: `fill`, `fit`, `stretch`, `center`.
- **Custom Matte & Solid Colors**: Supports `#hex` codes and named colors with letterboxing or pure solid backgrounds (`background-set color "#0d1117"`).
- **Multi-Display & Spaces**: Distribute wallpapers across monitors (`spaces`).
- **Random & Rotation**: Built-in background rotation daemon (`rotate ~/Wallpapers 15m`).
- **JSON Output**: `background-set list -j` returns instant monitor metadata in 5ms.

### 3. `display_ctl`
Native Swift display arrangement and resolution utility built directly with Quartz CoreGraphics.
- **Mirror / Extend**: Toggle between mirrored and extended desktop modes (`display_ctl mirror`, `display_ctl extend`, `display_ctl toggle`).
- **Resolution Control**: Inspect and configure display modes, backing pixel dimensions, and refresh rates (`display_ctl resolution 1920x1080 60`).
- **Instant Query**: Native C CoreGraphics calls executing in <10ms.
- **JSON Output**: `display_ctl list -j` outputs clean display lists.

### 4. `clipboard-network`
Zero-quote length-prefixed clipboard sharing bridge between macOS (`pbpaste`) and Windows SSH environments.

### 5. `bluetooth-tool` (alias: `bluetooth`)
Native Swift Bluetooth CLI on IOBluetooth. Full reference: [`tools/bluetooth-tool/README.md`](tools/bluetooth-tool/README.md).
- **Devices**: `list` / `status` with stable 1-based indices; target by index, alias, name substring or address.
- **Read-back transactions**: `connect`, `disconnect`, `toggle`, `power on|off|toggle` act only when needed (`changed:false` otherwise) and report the state read back; no read-back in time = exit 4.
- **Keyboard guard**: `power off` refuses (exit 3) while a Bluetooth keyboard, mouse or trackpad is connected, naming it, unless `--force`.
- **Waiting & events**: `wait <target> connected|disconnected -t 30`, `watch` (one line per connect/disconnect, JSON Lines with `-j`).
- **Logs**: `logs --last 2h --grep text` / `logs -f` over `bluetoothd` + `com.apple.bluetooth`.
- **Aliases**: `alias set receiver dh590` stores the device address in `~/.config/macos-gui/bluetooth_aliases.json`.
- **JSON**: `-j` on every command, a fixed contract (consumed by Topic). Exit codes: 0 ok, 1 failed, 2 usage, 3 refused, 4 timed out / unconfirmed.
- **Permission**: embeds an Info.plist with `NSBluetoothAlwaysUsageDescription`; a parent app (e.g. TopicGet.app) needs the key too.

---

## Installation

Install all utilities and their Zsh completions via [`scripts/install.sh`](file:///Users/zooey/etc/git/macos-gui/scripts/install.sh):

```bash
./scripts/install.sh
```

To preview without modifying anything:
```bash
./scripts/install.sh --dry-run
```

It won't install while any of the tools is running (a `watch`, a wallpaper `rotate`, a call from topic-get): it lists those processes and exits 3, because a Swift rebuild rewrites the binary in place and can kill whatever runs it. `--dry-run` says it would refuse; `--force` installs anyway.

Ensure the following is in your `~/.zshrc`:
```zsh
export PATH="$HOME/.local/bin:$PATH"
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit && compinit
```

---

## Tool Quick Reference

```bash
# AirPlay
airplay-tool list -j                       # List nearby devices in JSON
airplay-tool toggle                        # Smart toggle: disconnect or reconnect last host
airplay-tool start tv --mode extend        # Connect using friendly alias
airplay-tool status                        # Check active session & last host
airplay-tool stop                          # Clean disconnect
airplay-tool watch "LG"                    # Keepalive watchdog
airplay-tool logs --last 2h -j             # Unified Log diagnostics

# Wallpapers
background-set list -j                     # List display wallpaper states
background-set ~/Wallpapers/art.png        # Set wallpaper across screens
background-set color "#0d1117"             # Set solid matte background
background-set rotate ~/Wallpapers 15m     # Rotate wallpapers

# Display Arrangement
display_ctl list -j                        # Inspect displays
display_ctl mirror                         # Mirror secondary display
display_ctl extend                         # Extend secondary display
display_ctl toggle                         # Toggle mirroring mode
display_ctl resolutions "LG"               # List available display modes

# Bluetooth
bluetooth-tool list -j                     # Paired devices (index, address, kind, connected)
bluetooth-tool status                      # Power, connected devices, power-off guard verdict
bluetooth-tool disconnect receiver         # Read-back disconnect (alias, index, name or address)
bluetooth-tool wait "MAGIC KEEB" connected -t 30   # Block until it connects (exit 4 on timeout)
bluetooth-tool power off                   # Refuses while a BT keyboard/mouse/trackpad is connected
bluetooth-tool watch -j                    # JSON Lines per connect/disconnect
bluetooth-tool logs --last 30m --grep disconnect   # Unified log (bluetoothd)
```
