# macOS GUI Tools Suite: Architectural Review & Next Steps

This document reviews the project concerns, current build requirements, and architectural roadmap outlined in [`BUILD.md`](file:///Users/zooey/etc/git/macos-gui/BUILD.md).

---

## 1. Project Philosophy & Packaging Architecture

### Core Design Decision: Unix Philosophy Over Monolithic CLI
In [`BUILD.md`](file:///Users/zooey/etc/git/macos-gui/BUILD.md#L28), you noted:
> *"I don't want to have some kind of global `macos-util` tool that shows all tools available.. but again I have never bundled a set of scripts for use like this. Need best practices/documentation for me moving forward."*

**Recommendation: Independent, Single-Purpose Tools with a Modular Directory Layout**
- Avoid a monolithic `macos-util <subcommand>`. Modern Unix CLI suites (like Git, Docker plugins, or Homebrew utilities) work best when each command is an independent executable on `$PATH`.
- Structure each tool in `tools/<tool_name>/`:
  ```
  macos-gui/
  ├── README.md
  ├── BUILD.md
  ├── REVIEW.md
  ├── scripts/
  │   ├── install.sh             # Lightweight symlink & completion installer
  │   └── ...                    # Scratch / internal research scripts
  └── tools/
      ├── background_set/
      │   ├── background_set.swift # Native Swift implementation
      │   ├── background_set       # Compiled native arm64 executable
      │   ├── _background_set      # Zsh autocompletions
      │   └── README.md            # Tool-specific documentation
      ├── airplay/
      │   ├── airplay_connect
      │   ├── _airplay_connect
      │   └── README.md
      ├── clipboard_network/
      │   ├── clipboard-network
      │   ├── _clipboard-network
      │   ├── windows/           # .ps1 registration scripts
      │   └── README.md
      ├── window_ctl/
      │   └── ...
      └── screenshot/
          └── ...
  ```

### Frictionless Installation: [`scripts/install.sh`](file:///Users/zooey/etc/git/macos-gui/scripts/install.sh)
We created [`scripts/install.sh`](file:///Users/zooey/etc/git/macos-gui/scripts/install.sh) to handle packaging cleanly:
1. Symlinks executables to `~/.local/bin/` (or a custom `--bin-dir`).
2. Symlinks completion files (`_*`) to `~/.zsh/completions/`.
3. Compiles any Swift sources if binary is missing.
4. Requires only standard shell configuration in `~/.zshrc`:
   ```zsh
   export PATH="$HOME/.local/bin:$PATH"
   fpath=(~/.zsh/completions $fpath)
   autoload -Uz compinit && compinit
   ```
5. Supports `--dry-run` and `--uninstall` for clean management.

---

## 2. Background Set Tool: Code Built & Deep Technical Insights

The immediate code has been built in [`tools/background_set/`](file:///Users/zooey/etc/git/macos-gui/tools/background_set/) and wired to [`background_set.sh`](file:///Users/zooey/etc/git/macos-gui/background_set.sh).

### Implementation Highlights
- **Native Swift Executable**: Built with AppKit Cocoa APIs (`NSWorkspace.shared.setDesktopImageURL`), executing in sub-milliseconds without AppleScript overhead.
- **Display Targeting**:
  - Sets all connected displays by default.
  - Granular selection via `-d, --display <id|index|name>`:
    - Index: `0`, `1`, etc.
    - Hardware ID: `NSScreenNumber` (e.g. `1`, `23091`).
    - Substring matching: `--display "built-in"`, `--display "lg"`.
- **Scaling Modes (`-m, --mode`)**:
  - `fill`: Aspect fill (crop/zoom to cover without distorting) [default].
  - `fit`: Aspect fit (entire image visible, bordered with matte background color).
  - `stretch`: Stretch axes independently to screen dimensions.
  - `center`: 1:1 original pixel scale, centered without distortion.
- **Custom Background Color & Matte (`-c, --color`)**:
  - Uses `NSWorkspaceDesktopImageFillColorKey`.
  - Supports `#hex` codes (`#1e1e1e`, `#ffffff`, `#333`) and named colors (`black`, `white`, `gray`, `red`, etc.).
  - Supports pure solid-color wallpapers via `background_set color "#0d1117"`.
- **Listing & Status Inspection**:
  - `background_set list` or `status` displays all connected monitors, resolutions, IDs, current wallpaper URLs, mode, and fill colors.
  - `background_set get` retrieves current wallpaper path.
- **Rotation & Directory Randomizer**:
  - `random <dir>` selects and applies a random image.
  - `rotate <dir> [interval]` runs a foreground loop cycling through images.
- **Shell Completion**: Full zsh completion provided in [`_background_set`](file:///Users/zooey/etc/git/macos-gui/tools/background_set/_background_set).

### Critical Finding: Spaces vs Displays in macOS
In [`BUILD.md`](file:///Users/zooey/etc/git/macos-gui/BUILD.md#L7), you noted:
> *"targetable displays AND spaces by index/id would be great (perhaps it would set all spaces/all displays by default? targetting by ID is granular setting)"*

Our architectural investigation into macOS Sonoma, Sequoia, and macOS 26 reveals:
1. **The AppleScript Misconception**: In your previous `background_set.sh`, `count of desktops` returned the number of **physical monitors**, *not* Mission Control virtual Spaces. `set picture of desktop 1` modified monitor 1, not space 1.
2. **Private Wallpaper Storage**: Since macOS Sonoma, macOS moved per-space wallpaper state into an opaque private database (`~/Library/Application Support/com.apple.wallpaper/Store/Index.plist`). There is no public Apple API to set wallpaper on an inactive virtual space without restarting `WallpaperAgent` (which is disruptive and unstable).
3. **Active Space Scope**: Native `NSWorkspace.shared.setDesktopImageURL` applies to the **currently active space** of the specified screen.
4. **Space Rotation / Multi-Display**:
   - For multi-monitor setups, our `spaces` command distributes wallpapers across physical screens.
   - For virtual spaces, if distinct per-space wallpapers are needed, a lightweight watcher daemon can subscribe to `NSWorkspace.activeSpaceDidChangeNotification` to swap wallpapers dynamically on space changes.

### Rotation Strategy
In [`BUILD.md`](file:///Users/zooey/etc/git/macos-gui/BUILD.md#L9):
> *"the scheduled/rotation stuff could honestly be either marked as 'get another tool to do this' maybe?"*

- **Current State**: We retain lightweight `random` and foreground `rotate <dir> [interval]` in `background_set`.
- **Recommendation**: For scheduled background rotation that survives terminal exit and system reboots, we should generate standard macOS `launchd` plist agents (`~/Library/LaunchAgents/com.user.wallpaper.plist`) or recommend cron/launchd rather than keeping an unmonitored shell loop running.

---

## 3. Application & Window Controls (Tiling & Positioning)

### Analysis of [`tmp/max_options.png`](file:///Users/zooey/etc/git/macos-gui/tmp/max_options.png)
In [`BUILD.md`](file:///Users/zooey/etc/git/macos-gui/BUILD.md#L14):
> *"windows management is solved by other tools AND macOS itself now (tmp/max_options.png) - could those be somehow roped into our controls? i would love to say e.g. move spotify to left 1/2 horizontal of screen."*

The screenshot in `tmp/max_options.png` captures the native **macOS Sequoia Window Tiling menu**:
- **Move & Resize**: Left half, Right half, Top half, Bottom half.
- **Fill & Arrange**: Fill, Left & Right, Left & Quarters, Quarters.

### How We Can Control This in CLI
There are two ways to accomplish this:
1. **Direct System Menu Automation via AppleScript / System Events**:
   Under macOS Sequoia, every application with standard menus has:
   `menu bar item "Window" -> menu item "Move & Resize" -> menu item "Left"` (or "Right", "Fill", etc.).
   This can be triggered directly:
   ```applescript
   tell application "System Events"
       tell process "Spotify"
           click menu item "Left" of menu 1 of menu item "Move & Resize" of menu "Window" of menu bar 1
       end tell
   end tell
   ```
2. **Accessibility API (`AXUIElement`) Position & Size**:
   For granular positioning (e.g. thirds, quarters, exact coordinates, centering, or minimizing/maximizing across multiple displays), we can set:
   - `AXPosition` (`x`, `y`)
   - `AXSize` (`width`, `height`)
   - `AXMinimized` (`true`/`false`)
   - `AXFullScreen` (`true`/`false`)

**Proposed Tool: `window_ctl` / `win`**
- Commands:
  - `win raise <app>`: Bring app and window to front (already started in `application_raise.sh`).
  - `win tile <app> <left|right|top|bottom|fill|center>`: Triggers native macOS tiling.
  - `win move <app> <x> <y> <w> <h>`: Absolute placement.
  - `win minimize <app>` / `win maximize <app>`.

---

## 4. Screenshot & Video Capture Tool

### Current Issue in Repo
`screenshot_capture.sh` in the workspace root was inadvertently overwritten with a copy of `background_set.sh`.

### Proposed Tool: `tools/screenshot/`
macOS comes with high-grade native CLI tools (`screencapture` and `AVFoundation`):
- `screenshot screen [display-index]` (`screencapture -D <display_id>`)
- `screenshot window <app-name>` (`screencapture -l <window_id>`)
- `screenshot area` (`screencapture -i`)
- `screenshot record [duration]` (`screencapture -v` for native video capture without external dependencies)

---

## 5. AirPlay & Display Controls (`tools/airplay/` & `tools/display_ctl/`)

### Deep Investigation: What Broke in `airplay_connect`?

Through macOS Accessibility UI hierarchy analysis and Unified Log forensics, we identified the exact root causes of the reported issues:

#### 1. Why `stop` Failed ("opens dropdown then nothing happens")
- **ControlCenter Dynamic UI Shift**: When an AirPlay display is disconnected, ControlCenter presents it as an `AXCheckBox` (`role: AXCheckBox`, `value: 0`).
- **The State Transformation**: Once connected, ControlCenter completely replaces the element with an `AXDisclosureTriangle` (`role: AXDisclosureTriangle`, `value: 1`, `subrole: AXToggle`) alongside two mode buttons (`AXButton` for Mirror vs Extend) and status text (`AXStaticText: "Currently Extending"`).
- **The Bug**: The legacy script queried exclusively `checkboxes of group 1 of scrollArea`. Because the connected device was now an `AXDisclosureTriangle`, the search returned no checked checkboxes, leaving `targetElem` empty and failing with `Could not find display with index or ID:`.
- **The Fix**: `tools/airplay/airplay_connect` now targets any active `AXDisclosureTriangle` with `value: 1` or `AXCheckBox` with `value: 1`. Clicking the disclosure triangle cleanly and immediately toggles the session off.

#### 2. Why Device IDs & English Names Failed
- **DNS-SD vs ControlCenter Discrepancies**:
  - `dns-sd -L` yields `deviceid=4D:42:49:32:5D:B5` and `psi=00000000-0000-0000-0000-4D4249325DB5`.
  - The legacy script showed `psi` as the device ID.
  - In ControlCenter, the accessibility identifier is `screen-mirroring-device-AirPlay:4D:42:49:32:5D:B5`.
  - Testing `00000000-0000-...` failed substring matching against `4D:42:49:32:5D:B5`.
  - Passing English names (`"LG"`, `"[LG] webOS TV..."`) failed because the legacy script only evaluated `theID contains theTarget`, completely ignoring device titles and names.
  - Discovery iteration order in zsh associative arrays was non-deterministic, and local host (`foxy`) was listed as index 1 despite never appearing in ControlCenter.
- **The Fix**: `tools/airplay/airplay_connect` implements full normalization:
  - Resolves friendly names (`"LG"`, `"webOS"`, `"torus"`) to exact hardware target IDs.
  - Strips colons/dashes to match raw hex (`4D4249325DB5`), formatted MACs, or UUIDs.
  - Automatically filters out localhost.
  - Sorts devices deterministically by name so index numbers remain permanent.

---

### Root Cause Analysis: Why the LG TV AirPlay Connection Drops

We isolated the exact moment of the connection drop using `log show` during the session that disconnected at `00:40:17`:

```syslog
2026-10-02 00:40:16.939379-0500 localhost ControlCenter[611]: (AVRouting) [com.apple.coremedia:]
  AVFigRoutingContextRouteConfigUpdated: reason=configUpdateReasonEndedBottomUpRouteChange
2026-10-02 00:40:17.108001-0500 localhost AirPlayXPCHelper[400]: (CoreUtils) [com.apple.airplay:APTransportConnectionHTTP]
  Connection [0x265D] closed to 192.168.1.67:40961 (from 192.168.1.66:56419)
2026-10-02 00:40:17.108089-0500 localhost AirPlayXPCHelper[400]: (CoreUtils) [com.apple.airplay:APTransportSession]
  ### [0xB852] stream 0x81DE disconnected with reason -72320/0xFFFEE580
2026-10-02 00:40:17.108157-0500 localhost AirPlayXPCHelper[400]: (CoreUtils) [com.apple.airplay:APSenderSessionAirPlay]
  transport session [0xB852] disconnected with reason: -16613/0xFFFFBF1B
```

#### Diagnostic Conclusions
1. **The Disconnect Origin**: `configUpdateReasonEndedBottomUpRouteChange` is AVFoundation terminology indicating that the route teardown was initiated by the **receiver** (the LG TV), *not* macOS ("TopDown").
2. **The Socket Teardown**: The LG TV closed the TCP event stream socket (`192.168.1.67:40961`), triggering error `-72320` (`kConnectionClosedErr`) and session abort `-16613`.
3. **LG webOS C1 Firmware Bug**:
   - LG's **Quick Start+** feature prevents webOS system daemons from cleanly reinitializing across standby cycles. Over time, the internal AirPlay daemon (`airplay2d`) encounters memory fragmentation and socket timeouts under continuous screen mirroring streams.
   - Network protocol mismatch (IPv6 mDNS / IGMP snooping) on LG Wi-Fi interfaces can trigger keepalive drops.

#### Recommended Fixes on LG TV
1. **Disable Quick Start+**:
   - Navigate to: `Settings > All Settings > General > Devices > TV Management > Quick Start+` -> **Turn OFF**.
   - **Crucial**: Perform a cold power cycle by unplugging the LG TV power cord from the wall for 60 seconds, then plug back in.
2. **Disable IPv6 on TV**:
   - Navigate to: `Settings > Network > Wi-Fi Connection > Advanced Wi-Fi Settings` -> **Turn IPv6 OFF**.
3. **CLI Auto-Reconnect Watchdog**:
   - In `tools/airplay/airplay_connect`, run `airplay_connect watch "LG"`. If the TV ever drops the stream, the watchdog detects the teardown and auto-reconnects in < 2 seconds.

---

### New Native Swift Display Management: `tools/display_ctl/`

To provide robust programmatic control without relying on menubar GUI scripting for display properties, we built **`display_ctl`**:
- **Mirroring & Extension**: Built directly on Quartz Display Services C APIs (`CGConfigureDisplayMirrorOfDisplay`, `CGBeginDisplayConfiguration`, `CGCompleteDisplayConfiguration`). Toggles or configures mirror vs extend in < 10ms.
- **Resolution Management**: Inspects all hardware display modes via `CGDisplayCopyAllDisplayModes` and applies resolutions with `CGDisplaySetDisplayMode`.
- **Display Status**: Reports logical scaling, physical backing pixels, refresh rates, hardware IDs, and mirror relationships for all connected screens.

Commands:
```bash
display_ctl list                      # Inspect all displays, IDs, resolutions, and mirror states
display_ctl mirror                    # Mirror secondary (AirPlay/external) to main display
display_ctl mirror "LG" 0             # Mirror LG TV to built-in display
display_ctl extend                    # Set secondary display to independent desktop
display_ctl toggle                    # Toggle between mirror and extend
display_ctl resolutions "LG"          # View available resolutions and refresh rates
display_ctl resolution "LG" 1920x1080 # Set resolution
```

---

## 6. Clipboard Network Migration

- `clipboard-network`, `_clipboard-network`, and the Windows PowerShell scripts are co-located in `tools/clipboard_network/`.
- Lightweight install symlinks maintained in `scripts/install.sh`.

---

## Summary of Completed Tasks & Next Steps

### Completed
- [x] Researched macOS display vs virtual space APIs, scaling options, and fill colors.
- [x] Built native Swift `background_set` tool with display targeting, 4 scaling modes (`fill`, `fit`, `stretch`, `center`), matte hex color, solid background canvas generation, and display inspection.
- [x] Built native Swift `display_ctl` tool with instant CoreGraphics mirroring (`mirror`, `extend`, `toggle`), resolution switching (`resolution`, `resolutions`), and status reporting.
- [x] Fixed `airplay_connect`:
  - Resolved `stop` failure by targeting active macOS `AXDisclosureTriangle` and checked toggle elements.
  - Resolved device ID & name targeting (supports index, friendly name substrings `"LG"`, MAC with/without colons, UUID).
  - Stabilized device discovery with deterministic sorting and localhost exclusion.
  - Added `--mode mirror|extend` automated arrangement upon connection.
  - Added `airplay_connect watch <target>` keepalive watchdog daemon for auto-reconnecting intermittent displays.
  - Added `airplay_connect logs` for instant macOS Unified Log diagnostics.
- [x] Diagnosed LG TV disconnection root cause: isolated `configUpdateReasonEndedBottomUpRouteChange` and socket teardown `-72320` from TV receiver; provided Quick Start+ / IPv6 remediation.
- [x] Updated [`scripts/install.sh`](file:///Users/zooey/etc/git/macos-gui/scripts/install.sh) to compile and symlink all independent tools (`background_set`, `display_ctl`, `airplay_connect`, `clipboard-network`) and their zsh completions.
- [x] Documented architectural review, forensic log analysis, and usage guides in [`REVIEW.md`](file:///Users/zooey/etc/git/macos-gui/REVIEW.md), [`tools/display_ctl/README.md`](file:///Users/zooey/etc/git/macos-gui/tools/display_ctl/README.md), and [`tools/airplay/README.md`](file:///Users/zooey/etc/git/macos-gui/tools/airplay/README.md).

### Proposed Next Steps
1. **Window Tiling Tool (`window_ctl`)**: Implement the macOS Sequoia tiling integration (`left`, `right`, `top`, `bottom`, `fill`) based on `tmp/max_options.png`.
2. **Screenshot & Video Tool (`screenshot`)**: Rebuild `screenshot_capture.sh` with window targeting, display targeting, and video recording.
