# background-tool

A high-performance native macOS CLI utility for setting desktop wallpapers, solid colors, rotating directories, multi-display space configurations, and generating dynamic HEIC wallpapers (solar, time-based, and light/dark appearance).

## Features

- **Dynamic HEIC Generation (`dynamic`)**:
  - Ingests image directory or list of frames and compiles native multi-image `.heic` wallpapers with Apple XMP metadata.
  - Supports three Apple schedule standards:
    - **`solar`**: Sun altitude (`a`) and azimuth (`z`) calculations matching macOS solar tracking.
    - **`time` (`h24`)**: Clock-based 24-hour day fraction schedule (`t = 0.0 .. 1.0`).
    - **`appearance` (`apr`)**: Light/Dark mode transitions with dedicated `l` and `d` appearance previews.
- **Display UUID Targeting**:
  - Target displays by persistent hardware UUID (e.g. `37D8832A-2D66-02CA-B9F7-8F30A301B230`), prefix, 0-based index (`0`, `1`), display ID (`1`, `87`), or localized name (`built-in`, `lg`).
- **Read-Back Transactions**:
  - Modifying desktop wallpapers initiates an active verification transaction against macOS window server preferences, confirming the applied path before reporting success.
- **Standard Flag & JSON Language**:
  - Full support for `--json` (`-j`), `--verbose` (`-v`), `--quiet` (`-q`), `--dry-run` (`-n`), and standardized exit codes.

## Commands

```bash
background-tool <image> [options]
background-tool set <image> [options]
background-tool dynamic <directory|images...> [options]
background-tool color <hex|name> [options]
background-tool random <directory> [options]
background-tool rotate <directory> [interval] [options]
background-tool spaces <directory|images...> [options]
background-tool list [options]
background-tool get [options]
```

### Dynamic Wallpaper Creation

```bash
# Ingest 16 frames from directory with solar schedule and apply to display 0
background-tool dynamic ~/Pictures/MojaveSolar/ --schedule solar --apply -d 0

# Create Light/Dark dynamic HEIC from two images
background-tool dynamic day.png night.png --schedule appearance --out ~/Pictures/day_night.heic

# 24-hour time-based dynamic wallpaper
background-tool dynamic ~/Pictures/Timelapse/ --schedule time --apply -d 23C9D0D5-AAAD-48C2-99E8-4E5C460725D6
```

### Options

| Option | Shorthand | Description |
| --- | --- | --- |
| `--display` | `-d` | Target display (UUID, index, ID, name substring, or `all`) |
| `--mode` | `-m` | Scaling mode: `fill`, `fit`, `stretch`, `center` (default: `fill`) |
| `--color` | `-c` | Fill / matte border color (`#hex` or color name) |
| `--schedule` | `-s` | Dynamic schedule type: `solar`, `time`, `appearance` |
| `--out` | `-o` | Output file path for generated `.heic` file |
| `--apply` | | Apply generated dynamic wallpaper to target displays |
| `--light` | | Frame index or filename override for light appearance |
| `--dark` | | Frame index or filename override for dark appearance |
| `--dry-run` | `-n` | Simulate operations without modifying system settings |
| `--json` | `-j` | Machine-readable structured JSON format |
| `--verbose` | `-v` | Diagnostic logs and internal framework details on stderr |
| `--quiet` | `-q` | Suppress non-essential output for terse scripting |
| `--version` | `-V` | Print tool version string |
| `--help` | `-h` | Display help text and subcommand options |

### Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success / verified |
| `1` | Runtime failure (e.g. file error or decode failure) |
| `2` | Usage error (invalid flag, missing arguments) |
| `4` | Unconfirmed / read-back transaction timed out |
