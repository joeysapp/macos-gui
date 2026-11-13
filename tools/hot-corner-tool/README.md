# hot-corner-tool

A high-performance, native Swift CLI for triggering macOS system workspace actions traditionally mapped to Hot Corners.

## Actions

- **`desktop`** (`show-desktop`, `desk`): Show Desktop (moves windows aside to expose desktop icons and files).
- **`mission-control`** (`mc`): Mission Control (overview of all open application windows and Desktop Spaces).
- **`app-windows`** (`windows`, `expose`): Application Windows (App Exposé for the frontmost application).
- **`notification-center`** (`notifications`, `nc`): Notification Center (toggles macOS notification feed and widget panel).

## Usage

```bash
hot-corner-tool <command> [options]
```

### Commands

```bash
hot-corner-tool desktop
hot-corner-tool mission-control
hot-corner-tool app-windows
hot-corner-tool notification-center
```

### Options

| Option | Shorthand | Description |
| --- | --- | --- |
| `--json` | `-j` | Machine-readable structured JSON format |
| `--verbose` | `-v` | Diagnostic logs and internal framework details on stderr |
| `--quiet` | `-q` | Suppress non-essential output for terse scripting |
| `--version` | `-V` | Print tool version string |
| `--help` | `-h` | Display help text and subcommand options |

### JSON Output

```json
{
  "action" : "desktop",
  "success" : true,
  "timestamp" : 1794550918.7276869
}
```

On error:
```json
{
  "code" : 2,
  "error" : "unknown command 'xyz'",
  "success" : false
}
```

### Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success / action triggered |
| `1` | Runtime failure (e.g. permission error) |
| `2` | Usage error (unknown command or option) |
