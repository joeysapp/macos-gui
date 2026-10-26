# `bluetooth-tool` - native macOS Bluetooth CLI

Swift + IOBluetooth. Lists paired devices, connects/disconnects them, reads and sets controller
power, waits for and watches connection changes, and reads the Bluetooth unified log.
Also installed as `bluetooth`.

Every verb that changes something is a **read-back transaction**: read the state; if it already
says what was asked, send nothing (`changed:false`); otherwise act, read back, and report `now`
from the read-back. A state that doesn't read back within the timeout exits 4, never 0.

## Commands

| Command | Does |
| --- | --- |
| `list` | Paired devices, sorted by name then address, with 1-based indices |
| `status` | Power and connected devices (human output also shows the power-off guard) |
| `power [on\|off\|toggle]` | No argument: print `on`/`off`. With one: set it, read it back |
| `connect <target>` | `openConnection()`, read back until connected |
| `disconnect <target>` | `closeConnection()`, read back until disconnected |
| `toggle <target>` | Connect if it reads disconnected, else disconnect |
| `wait <target> <connected\|disconnected>` | Block until the device reads that state |
| `watch [target]` | One line per connect/disconnect until Ctrl-C (exit 0) |
| `logs` | Unified log: process `bluetoothd` + subsystem `com.apple.bluetooth` |
| `alias [list]` | Aliases and the device each resolves to |
| `alias set <alias> <target>` | Store target's address under alias, read it back |
| `alias rm <alias>` | Remove (also `remove`); absent = `changed:false`, exit 0 |
| `help [command [sub]]` | Same as `<command> -h` / `--help` / `<command> help` |

## Options

| Option | Commands | Meaning |
| --- | --- | --- |
| `-j, --json` | all | JSON on stdout; `watch` and `logs -f` print JSON Lines |
| `-v, --verbose` | all | API calls, return codes, timings, permission state on stderr |
| `-h, --help` | all | Help for the command (`help` as first argument works too) |
| `-V, --version` | top level | `bluetooth-tool 0.1.0` (`-j`: `{"name":...,"version":...}`) |
| `-t, --timeout <s>` | power, connect, disconnect, toggle, wait | Read-back / wait limit, default 10 |
| `--force` | power | Turn off even while a Bluetooth keyboard/mouse/trackpad is connected |
| `--last <window>` | logs | `<n>[m\|h\|d]` or `boot`, default `2h` |
| `-f, --follow` | logs | Stream new entries until Ctrl-C; not with `--last`/`--limit` |
| `--grep <text>` | logs | Only messages containing text, case-insensitive |
| `-n, --limit <n>` | logs | Newest N entries, default 200, `0` = all |

Short flags cluster (`-jv`), values take `-t 5`, `-t5`, `--timeout 5` or `--timeout=5`, and `--`
ends options.

## Targets

| Form | Example | Notes |
| --- | --- | --- |
| 1-based index | `3` | From `list`; shifts when devices pair/unpair |
| alias | `receiver` | `alias set receiver 3` |
| name substring | `dh590` | Case-insensitive; exact name wins; several matches = error listing them |
| address | `94:08:53:f1:06:55`, `94-08-53-f1-06-55`, `940853f10655` | Full address, any case |

Only paired devices resolve. Script against the address or an alias; the index is positional.

## Exit codes

`0` ok, `1` failed (incl. unknown or ambiguous target), `2` usage, `3` refused by a safety guard,
`4` timed out / unconfirmed. Results go to stdout; narration and warnings go to stderr. Color only
when stdout is a TTY and `NO_COLOR` is unset; no emoji.

## JSON (`-j`)

Device:
```json
{"index":1,"name":"STR-DH590","address":"94:08:53:f1:06:55","kind":"Speaker","paired":true,"connected":false,"alias":"receiver"}
```
`address` is lowercase, colon-separated, zero-padded. `alias` is `null` when unset. `name` is
trimmed (falls back to the address). `kind` is one of: `Keyboard`, `Mouse`, `Trackpad`, `Gamepad`,
`Remote`, `Headphones`, `Headset`, `Speaker`, `Microphone`, `Video Display`, `Camera`, `Phone`,
`Computer`, `Network`, `Printer`, `Wearable`, `Unknown`. It comes from the class of device; BLE
devices report class 0, so theirs comes from macOS's own record (`system_profiler
SPBluetoothDataType`, one call per run, ~0.2 s), else `Unknown`.

| Command | Output |
| --- | --- |
| `list` | `{"power":"on","devices":[Device...]}` |
| `status` | `{"power":"on","connected":[Device...]}` |
| `power` | `{"power":"on"}` |
| `power on\|off\|toggle` | `{"success":true,"was":"on","now":"off","changed":true}` |
| `connect`/`disconnect`/`toggle` | `{"success":true,"action":"disconnect","changed":true,"was":"connected","now":"disconnected","device":Device,"ms":412}` |
| `wait` | `{"success":true,"state":"disconnected","device":Device,"waited_ms":1200}` |
| `watch` | JSON Lines: `{"at":"2026-10-26T08:49:36.661Z","event":"connected","device":Device}` |
| `logs` | `{"entries":[{"time":"2026-10-26T08:49:36.661Z","process":"bluetoothd","message":"..."}]}` |
| `logs -f` | JSON Lines, one entry object each |
| `alias list` | `{"aliases":[{"alias":"receiver","target":"94:08:53:f1:06:55","device":Device\|null}]}` |
| `alias set` | `{"success":true,"changed":true,"alias":"receiver","was":null,"now":"94:08:53:f1:06:55","device":Device}` |
| `alias rm` | `{"success":true,"changed":true,"alias":"receiver","was":"94:08:53:f1:06:55","now":null}` |
| any failure | `{"success":false,"error":"<why>"}`, plus `"device"` when one was resolved |

`toggle`'s `action` is what it did (`connect` or `disconnect`). `ms` is the whole transaction.
Times are RFC3339 UTC with milliseconds. All JSON is one line per document.

## Behaviour notes

- **Power-off guard.** `power off` (and `power toggle` while on) refuses with exit 3, naming the
  device, while a Bluetooth keyboard, mouse or trackpad is connected. It checks connected
  IOBluetooth devices by kind *and* the IOKit HID registry (Bluetooth / BLE transports with
  keyboard, pointer or touchpad usages), so a BLE keyboard that IOBluetooth reports with class 0 or
  doesn't mark connected still blocks. `--force` overrides with a stderr warning. `status` shows
  the guard's current verdict.
- **Power set is a private API.** `IOBluetoothPreferenceSetControllerPowerState` /
  `...GetControllerPowerState` (as blueutil), declared with `@_silgen_name`; both are exported by
  IOBluetooth on macOS 26 (arm64). The read-back uses the private getter; if macOS ignores the
  setter, it exits 4 with that explanation. If the public `IOBluetoothHostController` disagrees
  after the getter confirms, stderr says so. **Setting power has not been run on hardware yet**:
  this machine's keyboards are on Bluetooth, so only the read path is verified.
- **connect** pages Classic (BR/EDR) devices with `openConnection()`; BLE devices normally
  reconnect on their own when woken, and connect may fail for them. Connection state is
  IOBluetooth's `isConnected()`; for audio devices that is the baseband link, not the audio profile.
- **watch** uses IOBluetooth connect and per-device disconnect notifications, plus a 1 s poll as a
  backstop; a line prints only when a device's read state changes (no duplicates). Polling is
  silent in the unified log. Ctrl-C, SIGTERM or SIGHUP exit 0.
- **logs** runs `/usr/bin/log show|stream --style ndjson` (absolute path: zsh has a `log` builtin).
  bluetoothd logs ~1000 lines/min, hence the default limit. bluetooth-tool's own IOBluetooth chatter
  (~45 lines per run from the CoreBluetooth shim) is excluded. `--grep` is applied in the predicate.
- **Aliases** store the device *address* (`~/.config/macos-gui/bluetooth_aliases.json`,
  `{"receiver": "94:08:53:f1:06:55"}`), so renames and index shifts don't break them. Hand-edited
  name values also resolve. Alias names: letters, digits, `.`, `_`, `-`; not all digits, not an
  address. A malformed alias file is ignored with a warning, and `alias set`/`rm` refuse to
  overwrite it. No other state is kept.

## Permission (TCC)

On macOS 26 IOBluetooth is a shim over CoreBluetooth and asks TCC (Bluetooth) on first use in each
process. The grant belongs to the **responsible process**: run from a terminal, that is the terminal
app; run by an app (TopicGet.app), it is that app.

- The binary embeds an Info.plist (`-sectcreate __TEXT __info_plist`, see `Info.plist`) with
  `NSBluetoothAlwaysUsageDescription`, so it can get its own prompt when it is its own responsible
  process (e.g. launched by launchd).
- **A parent .app that runs `bluetooth-tool` (TopicGet.app) still needs
  `NSBluetoothAlwaysUsageDescription` in its own Info.plist and the user's Bluetooth grant**;
  otherwise TCC denies (or kills) the child. Grant in System Settings > Privacy & Security >
  Bluetooth.
- If IOBluetooth returns no device list, or none while access reads denied, the tool exits 1 with
  that explanation instead of printing an empty list. `-v` prints the permission state.

## Build

`scripts/install.sh` compiles it when the source or `Info.plist` is newer than the binary:

```sh
swiftc -O tools/bluetooth-tool/bluetooth-tool.swift -o tools/bluetooth-tool/bluetooth-tool \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker tools/bluetooth-tool/Info.plist
```

Each rebuild changes the ad-hoc signature, so a TCC grant made to the binary itself may need
re-granting.

## Examples

```sh
bluetooth-tool list
bluetooth-tool list -j | jq -r '.devices[] | select(.kind=="Keyboard") | .address'
bluetooth-tool alias set receiver dh590
bluetooth-tool disconnect receiver -j
bluetooth-tool wait "MAGIC KEEB" connected -t 30 && echo typing
bluetooth-tool power off              # exit 3 while a BT keyboard/mouse/trackpad is connected
bluetooth-tool power toggle --force
bluetooth-tool watch -j | while read -r ev; do echo "$ev" | jq -r .event; done
bluetooth-tool logs --last 30m --grep disconnect
bluetooth-tool logs -f -j
```
