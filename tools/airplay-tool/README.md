# `airplay-tool` - macOS AirPlay Screen Mirroring CLI

A high-performance command-line utility for managing AirPlay screen mirroring sessions on macOS.

Also available via symlinks: `airplay`, `airplay_connect`.

## Key Features

- **Fast Concurrent Discovery**: Resolves network TXT records in parallel via Bonjour (`_airplay._tcp`), cutting scan times down to ~1 second.
- **Smart Toggle (`airplay-tool toggle`)**:
  - Automatically disconnects if a screen mirroring session is currently active.
  - Automatically searches for and reconnects to the last-used target host with configurable network discovery wait (`--timeout 8`).
- **Flexible Target Addressing**:
  - Stable 1-based indices (`airplay-tool start 1`)
  - User-defined shorthand aliases (`airplay-tool start tv`)
  - Substring friendly names (`airplay-tool start "LG"`, `airplay-tool start "torus"`)
  - Hardware device MAC (`4D:42:49:32:5D:B5`) or host UUID (`6B0E5414-F927-4F27-AEDB-DBDBD37A4493`)
- **JSON Support (`-j`, `--json`)**: Full machine-readable structured JSON output across all commands for scripting, piping, and automation.
- **State & Alias Management**:
  - Persistent state in `~/.local/state/macos-gui/airplay_state.json` tracks last connected host and mode.
  - Alias management in `~/.config/macos-gui/airplay_aliases.json` for custom friendly shortcuts (`alias set tv "LG"`).
- **Native Display Mode Integration**: Automatically configures `mirror` or `extend` upon connection via `display_ctl`.
- **Auto-Reconnect Watchdog (`watch`)**: Keepalive daemon that detects unexpected drops (common on LG OLED TVs) and reconnects immediately.
- **Unified Log Diagnostics (`logs`)**: Inspects recent macOS Unified Log events (`AirPlayXPCHelper`, `ControlCenter`) for disconnect reasons, socket errors, and performance metrics.
- **Zsh Autocompletion**: Full completion for all commands, flags, aliases, and dynamic target recommendations.

## Usage

```bash
# List available AirPlay devices
airplay-tool list
airplay-tool list -j

# Smart toggle (disconnects if active, or connects to last host)
airplay-tool toggle
airplay-tool toggle "LG" --timeout 10

# Connect to target (by index, name, alias, or ID)
airplay-tool start 1
airplay-tool start "LG" --mode extend
airplay-tool start tv

# Check connection status
airplay-tool status
airplay-tool status -j

# Disconnect
airplay-tool stop

# Manage friendly aliases
airplay-tool alias list
airplay-tool alias set tv "LG"
airplay-tool alias remove tv

# Auto-reconnect watchdog daemon
airplay-tool watch "LG" --mode extend

# Inspect recent AirPlay teardown & disconnect logs
airplay-tool logs --last 2h
airplay-tool logs --last 30m -j
```

## Options & Flags

| Flag | Description |
| --- | --- |
| `-j, --json` | Output machine-readable JSON format |
| `-v, --verbose` | Enable detailed diagnostic and timing output |
| `-m, --mode <mirror\|extend>` | Display arrangement mode (via `display_ctl`) |
| `-t, --timeout <seconds>` | Network discovery / wait timeout (default: 1.0s, toggle: 8.0s) |
| `--last <interval>` | Unified log search duration (e.g. `30m`, `1h`, `2h`, `24h`) |
| `-f, --filter <text>` | Filter log messages by substring |
| `-h, --help` | Display usage instructions |
| `--version` | Display version information |
