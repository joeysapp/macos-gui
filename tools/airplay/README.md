# `airplay_connect` - macOS AirPlay Screen Mirroring CLI

A robust command-line utility for managing AirPlay screen mirroring sessions on macOS.

## Key Features

- **Deterministic Device Matching**:
  - Target by 1-based index (`airplay_connect start 1`)
  - Target by friendly name substring (`airplay_connect start "LG"`, `airplay_connect start "torus"`)
  - Target by MAC address (`4D:42:49:32:5D:B5` or `4D4249325DB5`)
  - Target by UUID / `psi` (`00000000-0000-0000-0000-4D4249325DB5`)
- **Reliable Disconnect**:
  - `airplay_connect stop` with no arguments detects and cleanly stops any active screen mirroring or extending session (handles both macOS `AXDisclosureTriangle` and `AXCheckBox` states).
- **Mode Integration**:
  - Specify `--mode mirror` or `--mode extend` during `start` to automatically configure display arrangement upon connection via native CoreGraphics (`display_ctl`).
- **Auto-Reconnect Watchdog**:
  - `airplay_connect watch "LG"` continuously monitors the connection and automatically reconnects if an external device (such as an LG webOS TV) unexpectedly drops.
- **Diagnostics & Unified Log Inspection**:
  - `airplay_connect logs` inspects recent `AirPlayXPCHelper` and `ControlCenter` logs for socket teardowns, timeout reasons, and connection metrics.

## Usage

```bash
# List available AirPlay devices with stable indices
airplay_connect list

# Connect to device (by index, friendly name, or ID)
airplay_connect start 1
airplay_connect start "LG"
airplay_connect start "LG" --mode extend

# Check connection status
airplay_connect status

# Disconnect active session
airplay_connect stop

# Keepalive watchdog (automatically reconnects if dropped)
airplay_connect watch "LG"

# View recent disconnect logs
airplay_connect logs --last 2h
```
