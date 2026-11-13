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
- **JSON Support (`-j`, `--json`)**: Full machine-readable structured JSON output across all commands (and line-delimited JSON Lines for `watch` and `logs --follow`) for scripting, piping, and automation.
- **State & Alias Management**:
  - Persistent state in `~/.local/state/macos-gui/airplay_state.json` tracks last connected host and mode.
  - Alias management in `~/.config/macos-gui/airplay_aliases.json` for custom friendly shortcuts (`alias set tv "LG"`).
- **Native Display Mode Integration**: Automatically configures `mirror` or `extend` upon connection via `display_ctl`.
- **Auto-Reconnect Watchdog (`watch`)**:
  - Keepalive daemon that detects unexpected drops (common on LG OLED TVs) and reconnects immediately.
  - Configurable polling interval via `-i, --interval` (e.g. `60000ms`, `60s`, `1m`; default: `60000ms` / `60s`).
  - Target is optional: automatically defaults to the currently active session or last connected host.
  - Streams JSON Lines events with `-j`.
- **State Waiter (`wait`)**:
  - Blocks until target reaches `connected`, `disconnected`, or `online` state (`-t timeout`, `-i interval`).
  - Exits 0 on match, or 4 on timeout.
- **Unified Log Diagnostics & Streaming (`logs`)**:
  - Queries recent macOS Unified Log events (`AirPlayXPCHelper`, `ControlCenter`) with `--last`, `--limit`, and `--grep`.
  - Live log streaming with `-f, --follow` (streaming JSON Lines with `-j`).
- **Zsh Autocompletion**: Full completion for all commands, flags, dynamic targets, aliases, and modes.
- **Regression Suite**: Automated regression tests in `check.sh` testing all error paths, argument parsing, JSON contracts, and exit codes.

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
airplay-tool connect tv
airplay-tool start tv --force              # Disconnects any active display first

# Check connection status
airplay-tool status
airplay-tool status -j

# Disconnect
airplay-tool stop
airplay-tool disconnect

# Wait for a device state in scripts (exits 0 on state match, 4 on timeout)
airplay-tool wait tv connected --timeout 15
airplay-tool wait "LG" online --timeout 10
airplay-tool wait tv disconnected --timeout 5 -j

# Manage friendly aliases
airplay-tool alias list
airplay-tool alias set tv "LG"
airplay-tool alias remove tv

# Auto-reconnect watchdog daemon (default interval: 60000ms / 60s)
airplay-tool watch                         # Auto-detects active or last host
airplay-tool watch tv --interval 60000ms
airplay-tool watch "LG" --interval 30s --mode extend
airplay-tool watch tv -i 60000ms -j        # Stream JSON Lines events

# Inspect recent AirPlay teardown & disconnect logs
airplay-tool logs --last 2h
airplay-tool logs --last 30m --limit 50 --grep "disconnect" -j

# Live stream AirPlay Unified Log events
airplay-tool logs --follow
airplay-tool logs -f -j
```

## Options & Flags

| Flag | Description |
| --- | --- |
| `-j, --json` | Output machine-readable JSON (JSON Lines for `watch` and `logs -f`) |
| `-v, --verbose` | Enable detailed diagnostic, timing, and polling output |
| `-q, --quiet` | Suppress non-essential output |
| `-i, --interval <duration>` | Check interval for `watch` and `wait` (e.g. `60000ms`, `60s`, `1m`; default `60000ms` for watch) |
| `-t, --timeout <seconds>` | Wait/discovery timeout in seconds (default: 1.0s, toggle: 8.0s, wait: 30.0s) |
| `-m, --mode <mirror\|extend>` | Display arrangement mode (via `display_ctl`) |
| `--force` | Override active sessions or bypass safety checks |
| `--last <interval>` | Unified log search duration (e.g. `30m`, `1h`, `2h`, `24h`; default: `2h`) |
| `--limit <count>` | Maximum log lines to return (default: `200`, `0` for unlimited) |
| `--grep, --filter <pattern>` | Filter log messages by substring |
| `-f, --follow` | Stream live Unified Log events in real time |
| `-h, --help` | Display usage instructions (also `airplay-tool help <command>`) |
| `-V, --version` | Display version information |

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Operation failed (device not found, system error) |
| `2` | Usage error (invalid arguments or flags) |
| `3` | Refused (safety guard prevented action without `--force`) |
| `4` | Unconfirmed / timed out (wait timeout or toggle timeout) |
