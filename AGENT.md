# AGENT.md - Architecture, Conventions, and Contributor Guidelines

This document serves as the primary operational manual, design philosophy, and ruleset for agents and contributors working on the `macos-gui` repository. It consolidates the foundational rules originally established in `BUILD.md` along with subsequent architectural standards and tool additions.

---

## 1. Core Philosophy & Design Tenets

The suite provides lightweight, native, scriptable CLI utilities for macOS system management, screen mirroring, audio/display hardware routing, and workspace automation.

- **Modular Unix Philosophy**: Every tool is an independent, single-purpose executable (e.g. `airplay-tool`, `bluetooth-tool`, `audio-tool`, `background_set`, `display_ctl`). Never create a monolithic wrapper that forces tools into an all-in-one executable.
- **Read-Back Transactions**: Any command modifying system hardware or preferences must follow a read-back transaction: read current state, apply changes only when they differ, read the state back, and report what was actually read from the system—never blindly echo what was requested. If the state never reads back within the timeout, exit with code 4.
- **Shell-Hardened & Stream-Ready**: All tools must handle redirection, subshell execution, and UNIX piping cleanly. Streaming commands (`watch`, `logs --follow`, `level --watch`) must configure stdout to be line-buffered so downstream consumers receive live events without buffering delay.

---

## 2. CLI Command & Option Standards

Every tool and subcommand must strictly adhere to the following contract:

### Required Standard Flags
- **Help (`-h`, `--help`, `help`)**:
  - Global help must display an overview, available subcommands, options, exit codes, and examples.
  - Subcommands must support contextual help (e.g. `airplay-tool watch --help` or `airplay-tool help watch`).
  - Descriptions must be terse, direct, and explain the exact expected arguments.
- **JSON Output (`-j`, `--json`)**:
  - Full machine-readable structured JSON format on stdout across every command.
  - On error with `-j`: output a JSON error object (`{"error": "<msg>", "code": <code>, "success": false}`) and exit with `<code>`.
  - For streaming commands (`watch`, `logs --follow`): emit line-delimited JSON Lines (NDJSON).
- **Diagnostics (`-v`, `--verbose`)**:
  - Emits API timings, internal system queries, and detailed debug messages on stderr (leaving stdout clean for piping).
- **Quiet Mode (`-q`, `--quiet`)**:
  - Suppresses non-essential status messages for terse scripting.
- **Version (`-V`, `--version`, `version`)**:
  - Terse version string (`<tool-name> <version>`).

### Standard Exit Codes
All tools across the repository standardize on these exit codes:

| Code | Meaning | Example |
| --- | --- | --- |
| `0` | **Success / OK** | Command executed and verified |
| `1` | **Failed** | Runtime failure, device not found, system error |
| `2` | **Usage Error** | Missing required arguments, invalid flag, bad format |
| `3` | **Refused** | Safety guard tripped without `--force` (e.g. powering off BT with keyboard connected) |
| `4` | **Unconfirmed / Timed Out** | State never read back, wait condition timed out, toggle host discovery timed out |
| `128+N` | **Signal Termination** | Process terminated by signal `N` (SIGINT = 130) |

---

## 3. Zsh Autocompletions

- Every command must ship with a corresponding Zsh completion file prefixed with an underscore (e.g. `tools/airplay-tool/_airplay-tool`).
- Completions must support all subcommands, global flags, and command-specific options.
- Dynamic completion should be implemented wherever possible:
  - Aliases dynamically read from `~/.config/macos-gui/*_aliases.json`.
  - Discovered / cached devices dynamically read from `~/.cache/macos-gui/` or `tool list -j`.
- Symlinks to completions are installed into `~/.zsh/completions` via `scripts/install.sh`.

---

## 4. State & Configuration Paths

Persistent data follows XDG specifications and macOS conventions, honoring environment variables:

- **State Directory (`$XDG_STATE_HOME/macos-gui` or `~/.local/state/macos-gui`)**:
  - Tracks last-connected hosts, active modes, and runtime state (e.g. `airplay_state.json`).
- **Config Directory (`$XDG_CONFIG_HOME/macos-gui` or `~/.config/macos-gui`)**:
  - User-defined shorthand aliases (e.g. `airplay_aliases.json`, `bluetooth_aliases.json`, `audio_aliases.json`).
- **Cache Directory (`$XDG_CACHE_HOME/macos-gui` or `~/.cache/macos-gui`)**:
  - Transient discovery cache (e.g. `airplay_cache.json`).

---

## 5. Tool Suite Overview

### 1. `airplay-tool` (aliases: `airplay`, `airplay_connect`)
High-performance macOS AirPlay Screen Mirroring CLI.
- **Discovery**: Concurrent Bonjour (`_airplay._tcp`) TXT record resolution (~1 second).
- **Targeting**: 1-based index (`1`), alias (`tv`), substring (`LG`), MAC address, or UUID (`psi`).
- **Toggle**: Disconnects if active; otherwise reconnects to specified/last target with network discovery wait.
- **Watchdog (`watch`)**: Keepalive daemon that detects session drops and reconnects immediately. Polling interval configurable via `-i, --interval` (e.g. `60000ms`, `60s`, `1m`; default: `60000ms` / `60s`). Auto-detects active or last session if target omitted. Streams JSON Lines with `-j`.
- **Wait (`wait <target> [state]`)**: Blocks until target is `connected`, `disconnected`, or `online` (`-t timeout`, `-i interval`). Exits 0 on match, 4 on timeout.
- **Logs (`logs`)**: Unified Log viewer for `AirPlayXPCHelper` and `ControlCenter` events with `--last`, `--limit`, `--grep`, and live streaming with `-f, --follow`.
- **Display Modes**: Automatically invokes `display_ctl` upon connection for `--mode mirror` or `--mode extend`.
- **Regression Suite**: `tools/airplay-tool/check.sh`.

### 2. `bluetooth-tool` (alias: `bluetooth`)
Native Swift CLI built directly on `IOBluetooth`.
- **Transactions**: Read-back transactions for `power on|off|toggle`, `connect`, `disconnect`, `toggle`.
- **Safety Guards**: `power off` refuses (exit 3) while a Bluetooth keyboard, mouse, or trackpad is connected unless `--force`.
- **Wait & Watch**: `wait <target> connected|disconnected -t 30`; `watch` streams connect/disconnect events (JSON Lines with `-j`).
- **Logs**: Unified Log diagnostics over `bluetoothd` and `com.apple.bluetooth` (`--last`, `--grep`, `-f`).
- **Aliases**: `alias set <name> <target>` in `bluetooth_aliases.json`.
- **Permissions**: Embeds `Info.plist` with `NSBluetoothAlwaysUsageDescription`.

### 3. `audio-tool` (alias: `audio`)
Native Swift CLI built on `CoreAudio`.
- **Device Management**: Full hardware enumeration (including USB audio interfaces like Focusrite Scarlett).
- **Transactions**: Set and read-back default input/output/system device, volume, and mute.
- **Play & Record**: Direct IOProc audio playback without touching system defaults; WAV recording that cleanly finalizes even on interruption; level metering (`level --watch`).
- **Safety**: Safe resource deallocation preventing SIGABRT on failure paths.
- **Regression Suite**: `tools/audio-tool/check.sh` running 18 failure paths under AddressSanitizer.
- **Permissions**: Embeds `Info.plist` with `NSMicrophoneUsageDescription`.

### 4. `display-tool` (source: `tools/display_ctl`)
Native Swift display arrangement utility via Quartz CoreGraphics.
- **Modes**: Instant query (<10ms), desktop `mirror`, `extend`, `toggle`.
- **Resolution**: Query and set resolution and refresh rates.

### 5. `clipboard-tool` (source: `clipboard-network`)
Zero-quote length-prefixed clipboard sharing bridge between macOS (`pbpaste`) and Windows SSH environments.

### 6. `background-tool` (aliases: `background_set`, `background-set`)
Native Swift macOS wallpaper management and dynamic HEIC generator.
- **Dynamic HEIC Generation (`dynamic`)**: Ingests frame directory or file sequence and outputs compliant dynamic wallpapers with Apple XMP metadata (`apple_desktop:solar`, `apple_desktop:h24`, `apple_desktop:apr`).
- **Display UUID Targeting**: Targets displays by UUID, prefix, 0-based index, display ID, or name substring.
- **Transactions**: Read-back verification confirming applied wallpaper with window server within 2-second timeout (exit 4 on unconfirmed).
- **Regression Suite**: `tools/background-tool/check.sh`.

### 7. `hot-corner-tool`
Native Swift CLI for triggering macOS system workspace actions.
- **Commands**: `desktop` (Show Desktop), `mission-control` (Mission Control), `app-windows` (App Exposé), and `notification-center` (Notification Center sidebar).
- **One-Shot Actions**: Direct stateless execution via `CoreDockSendNotification` and system ControlCenter bridge.
- **Regression Suite**: `tools/hot-corner-tool/check.sh`.

---

## 6. Installation & Safety Rules

- The 7 standardized suite tools (`airplay-tool`, `audio-tool`, `background-tool`, `bluetooth-tool`, `display-tool`, `hot-corner-tool`, `clipboard-tool`) and their completions are installed via [`scripts/install.sh`](file:///Users/zooey/etc/git/macos-gui/scripts/install.sh).
- **Clean Installation**: `install.sh` automatically cleans up obsolete legacy definitions (`airplay`, `airplay_connect`, `audio`, `bluetooth`, `background_set`, `display_ctl`, `clipboard-network`).
- **In-Use Guard**: `install.sh` refuses (exit 3) if any tool in the suite is actively running (e.g. background watch daemon or audio stream) to prevent in-place binary overwrite corruption during Swift compilation. Use `--force` to bypass when intended.
- **Dry Run**: `scripts/install.sh --dry-run` previews all symlink, cleanup, and compilation actions without modifying disk.

---

## 7. Git & Development Rules

- **Commit Messages**:
  - Must be **single-line** and **strictly under 120 characters**.
  - Must serve as clear, concise at-a-glance headers.
  - Detailed design rationales and rules belong in the documentation (`README.md`, `AGENT.md`), not multiline commit messages.
- **Author Identity**:
  - Commits must be authored by the configured user:
    - Name: `Joey Sapp`
    - Email: `sapp.joey@gmail.com`
    - Matches `@joeysapp` on GitHub and the local git configuration.
- **Branching & Worktrees**:
  - Builds and feature work should take place in dedicated working branches or worktrees to prevent workspace distraction and conflicts.
- **Testing & Verification**:
  - Never commit regressions without verifying against existing test suites (`tools/airplay-tool/check.sh`, `tools/audio-tool/check.sh`).

---

## 8. Future Roadmap

- **macOS Notification Alerts**: Lightweight CLI tool to dismiss persistent top-right notification banners and linger alerts.
- **Time Machine CLI**: Audit backup health, snapshot dates, and drive/iCloud capacity.
- **Network / Bluetooth Peripheral Monitoring**: Monitor legacy network shares, mounted iOS devices, and camera streaming bridges.
- **macOS System Defaults**: Record and apply fundamental OS settings across new machines.
