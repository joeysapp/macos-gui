# `audio-tool` - native macOS audio CLI

Swift + CoreAudio. Lists every audio device (including USB interfaces `say -a '?'` leaves out),
shows and sets the default input/output/system output, reads and sets volume and mute, plays a
file on any device and channel, records and meters any device's input channels, and watches for
devices coming and going. Also installed as `audio`.

Every verb that changes something is a **read-back transaction**: read the state; if it already
says what was asked, send nothing (`changed:false`); otherwise act, read back, and report `now`
from the read-back. A state that doesn't read back within the timeout exits 4, never 0. A device
without the control asked for is exit 3 with the reason, never a fake success.

## Commands

| Command | Does |
| --- | --- |
| `list` | Every CoreAudio device, sorted by name then uid, with 1-based indices |
| `status` | The default input, output and system output, each with volume and mute |
| `default <input\|output\|system> [target]` | No target: print that default. With one: set it, read it back |
| `volume <target> [0-100]` | No value: print the output volume. With one: set it, read it back |
| `mute <target> [on\|off\|toggle]` | No value: print the output mute. With one: set it, read it back |
| `play <file>` | Play wav/aiff/caf/m4a/mp3 on one device (default: the default output), blocking |
| `record --from <target>` | Record input channels to a WAV for `-s` seconds or `--until-silence` |
| `level --from <target>` | Peak and RMS dBFS per input channel over a span, or every 100 ms (`--watch`) |
| `watch` | One line per device added/removed and per default change, until Ctrl-C (exit 0) |
| `alias [list]` | Aliases and the device each resolves to |
| `alias set <alias> <target>` | Store target's uid under alias, read it back |
| `alias rm <alias>` | Remove (also `remove`); absent = `changed:false`, exit 0 |
| `help [command [sub]]` | Same as `<command> -h` / `--help` / `<command> help` |

## Options

| Option | Commands | Meaning |
| --- | --- | --- |
| `-j, --json` | all | JSON on stdout; `watch` and `level --watch` print JSON Lines |
| `-v, --verbose` | all | CoreAudio calls, status codes, stream formats, latencies, timings on stderr |
| `-h, --help` | all | Help for the command (`help` as first argument works too) |
| `-V, --version` | top level | `audio-tool 0.1.0` (`-j`: `{"name":...,"version":...}`) |
| `-t, --timeout <s>` | default, volume, mute | Read-back limit, default 5 |
| `--input` | volume, mute | The input side instead of the output |
| `--to <target>` | play | Output device (default: the default output) |
| `-c, --channels <list>` | play, record, level | 1-based: `1`, `1,2`, `3-4`, `L` (=1), `R` (=2), `LR` (=1,2) |
| `--gain <dB>` | play | -120..24, default 0; samples clip at full scale (stderr says how many) |
| `--from <target>` | record, level | Input device (required) |
| `-s, --seconds <s>` | record, level | record: exact length. level: span (default 2), or the `--watch` limit |
| `--until-silence <s>` | record | Stop after this long below `--threshold`, once something louder was heard |
| `--max <s>` | record | Cap for `--until-silence`, default 60 |
| `--threshold <dBFS>` | record | Silence level for `--until-silence`, -120..0, default -45 |
| `--rate <Hz>` | record | Resample to this rate (8000..384000) |
| `--float` | record | 32-bit float instead of 24-bit PCM |
| `-o, --out <file.wav>` | record | Output path (default `./record-YYYYMMDD-HHMMSS.wav`) |
| `--force` | record | Overwrite `--out` if it exists (otherwise exit 3) |
| `--watch` | level | One line per 100 ms until Ctrl-C (exit 0) |

Short flags cluster (`-jv`), values take `-s 2`, `-s2`, `--seconds 2` or `--seconds=2`, `--`
ends options. `--gain`/`--threshold` accept a `dB` suffix (`--gain=-6dB`); negative values work as
the next argument (`--gain -6`). Defaults per command: play `--channels` 1,2 for mono/stereo files
(else 1..N), record 1, level all.

## Targets

| Form | Example | Notes |
| --- | --- | --- |
| 1-based index | `4` | From `list`; shifts when devices come and go |
| alias | `scarlett` | `alias set scarlett "Scarlett 2i2"` |
| uid | `BuiltInSpeakerDevice` | `kAudioDevicePropertyDeviceUID`, exact (any case): **the stable key** |
| name substring | `speakers` | Case-, diacritic-insensitive, `’` = `'`; exact name wins |

Several name matches narrow to the devices with the channels the command needs (outputs for play,
`default output`, volume; inputs for record, level, `default input`, `--input`), so `MacBook`
means the speakers to `play` and the microphone to `record`. Still several: exit 2 listing them
(`"matches"` in JSON). No match or a bad index: exit 1 naming every device (`"devices"` in JSON).
Script against the uid or an alias; the index is positional and the `id` (AudioObjectID) changes
when a device is replugged or re-enumerated, and across reboots.

## Exit codes

`0` ok, `1` failed (incl. no such target), `2` usage (incl. ambiguous target), `3` refused (the
device has no such control, channel or side; it says it can't be the default; `--out` exists),
`4` timed out / not confirmed by read-back (`record --until-silence` that heard nothing).
`play`, `record` and `level` cut short by a signal print their result and exit 128+signal (130
Ctrl-C, 143 SIGTERM); `watch` and `level --watch` exit 0 on Ctrl-C. Results go to stdout; narration
and warnings go to stderr. Color only when stdout is a TTY, `NO_COLOR` is unset and `TERM` isn't
`dumb`; no emoji.

## JSON (`-j`)

Device (from `list`, `status`, `default`, `alias`):
```json
{"index":4,"id":87,"uid":"AppleUSBAudioEngine:Focusrite:Scarlett 2i2 4th Gen:S26M909368233E:1,2",
 "name":"Scarlett 2i2 4th Gen","manufacturer":"Focusrite","transport":"usb","inputs":4,"outputs":2,
 "input_channels":["Input 1","Input 2","Loopback 1","Loopback 2"],"output_channels":["Output 1","Output 2"],
 "rate":48000,"default_input":false,"default_output":true,"default_system_output":true,"alive":true,"alias":null}
```
- `id` is the AudioObjectID: **volatile**. `uid` is the stable key.
- `transport` is one of `usb`, `builtin`, `bluetooth` (incl. LE), `continuity` (iPhone microphone),
  `aggregate`, `virtual`, `hdmi`, `displayport`, `airplay`, `thunderbolt`, `pci`, `firewire`, `avb`,
  `unknown`, from `kAudioDevicePropertyTransportType`.
- `inputs`/`outputs` are channel counts from the stream configuration. `input_channels` /
  `output_channels` are `kAudioObjectPropertyElementName` per channel, as the device gives them
  (built-in devices name theirs `"1"`, `"2"`; `""` for one it leaves unnamed), `[]` if it names none.
- `rate` is the nominal sample rate. `alias` is `null` when unset.

Stub (play, record, level, volume, mute, default set): `{"uid":"BuiltInSpeakerDevice","name":"MacBook Pro Speakers"}`.

| Command | Output |
| --- | --- |
| `list` | `{"devices":[Device...]}` |
| `status` | `{"input":{"device":Device,"volume":50,"mute":false},"output":{...},"system":{...}}`; `null` where none |
| `default <role>` | `{"role":"output","device":Device}` (`device` null when there is none) |
| `default <role> <target>` | `{"success":true,"role":"output","changed":true,"was":Stub,"now":Stub,"ms":4}` |
| `volume <target>` | `{"device":Stub,"scope":"output","volume":69}` |
| `volume <target> <n>` | `{"success":true,"device":Stub,"scope":"output","changed":true,"was":69,"now":40,"ms":3}` |
| `mute <target>` | `{"device":Stub,"scope":"output","mute":false}` |
| `mute <target> <state>` | `{"success":true,"device":Stub,"scope":"output","changed":true,"was":false,"now":true,"ms":2}` |
| `play` | `{"success":true,"played":"/abs/tone.wav","to":Stub,"channels":[1],"seconds":0.5,"ms":680,"stopped":"end"}` |
| `record` | `{"success":true,"file":"/abs/rec.wav","from":Stub,"channels":[3,4],"rate":48000,"seconds":2.5,"peak_dbfs":-12.17,"rms_dbfs":-28.99,"clipped":false,"format":"s24","stopped":"seconds","dropped_frames":0,"ms":2574}` |
| `level` | `{"from":Stub,"rate":48000,"seconds":1,"channels":[{"channel":1,"peak_dbfs":-54.42,"rms_dbfs":-66.31,"clipped":false}],"stopped":"seconds"}` |
| `level --watch` | JSON Lines: `{"at":"2026-11-01T16:19:21.973Z","channels":[...]}` |
| `watch` | JSON Lines: `{"event":"added","uid":"...","name":"...","at":"<RFC3339 UTC>"}`; default changes add `"role"` (`input`/`output`/`system`), with `uid`/`name` null when there is none |
| `alias list` | `{"aliases":[{"alias":"scarlett","target":"<uid>","device":Device\|null}]}` |
| `alias set` | `{"success":true,"changed":true,"alias":"scarlett","was":null,"now":"<uid>","device":Device}` |
| `alias rm` | `{"success":true,"changed":true,"alias":"scarlett","was":"<uid>","now":null}` |
| any failure | `{"success":false,"error":"<why>"}`, plus `"device":Stub` when one was resolved, `"matches"` / `"devices"` for target errors |

- `seconds` (play) is the audio actually played; (record) the file's length; `ms` is wall time.
- `stopped`: play `end` | `signal`; record `seconds` | `silence` | `max` | `signal` | `error`; level `seconds` | `signal`.
- dBFS values have 2 decimals; silence (-inf) is `null`. `clipped` = a sample at -0.01 dBFS or above.
- `format` is `s24` or `f32`. `dropped_frames` > 0 means the recording has gaps (we fell behind, or
  the device skipped sample time).
- A record failure after the file exists (heard nothing by `--max`, device gone, disk full) prints the
  same object with `"success":false` and `"error"`, and keeps the file.
- Times are RFC3339 UTC with milliseconds. All JSON is one line per document.

## Behaviour notes

- **play** reads the file with AVAudioFile, resamples it to the device's nominal rate with
  AVAudioConverter (high quality), and streams it through a 2 s lock-free ring into a raw IOProc on
  that device (`AudioDeviceCreateIOProcIDWithBlock`), so it mixes with other apps and never touches
  the defaults, the device's rate, buffer size or volume. The IOProc's input streams are switched off
  (`kAudioDevicePropertyIOProcStreamUsage`, read back), so playing to an interface doesn't open its
  microphones. Channel mapping: a mono file goes to every listed channel; one listed channel gets the
  average of the file's channels; otherwise file channel i goes to listed channel i (extra file
  channels are dropped, extra listed channels get silence, with a warning). It returns after the last
  frame has left the device (device + stream latency, safety offset and one buffer). Ctrl-C/SIGTERM
  fade out over 10 ms first. It warns (stderr) when the device is muted or at volume 0.
- **record** gathers the listed input channels in the IOProc into a 10 s ring; the main thread
  measures, optionally resamples and writes. WAV: 24-bit PCM (`--float`: IEEE float with a `fact`
  chunk), WAVE_FORMAT_EXTENSIBLE above 2 channels, header sizes rewritten every 0.5 s and at the end,
  so even a killed process leaves a playable file. `--seconds` is counted in frames (exact length).
  `--until-silence`: 50 ms blocks; a block is loud when any channel's RMS reaches `--threshold`; after
  the first loud block, that many seconds of quiet blocks in a row stop it (the trailing quiet is
  kept). Levels in the result are measured on the captured audio, before `--rate`. Refuses lengths
  that would pass the 4 GB WAV limit. A device that stops delivering audio for 2 s (unplugged) ends
  the recording with an error and keeps the file. The capture starts before the file is created, so
  a capture that can't start leaves no empty file. A second Ctrl-C exits at once; the file is still
  valid up to the last header rewrite.
- **level** uses the same capture; `--watch` windows are 100 ms of audio each.
- **volume** uses the control macOS's own volume slider uses (`kAudioHardwareServiceDeviceProperty_
  VirtualMainVolume` via AudioObject), else the main element's `VolumeScalar`, else every channel's.
  A read-back within 1 of the value asked counts (hardware steps); `now` is what reads back.
  **mute** is the main element's `Mute`, else every channel's (muted = all read muted). Many USB
  interfaces (the Scarlett 2i2 4th Gen among them) expose neither: exit 3.
- **default** refuses (exit 3) a device without channels on that side or whose
  `kAudioDevicePropertyDeviceCanBeDefault(System)Device` is 0. macOS itself moves the defaults when
  devices come and go (`coreaudiod`'s preferred-device list); `watch` shows it.
- **watch** registers HAL listeners on the device list and the three defaults, plus a 1 s poll as a
  backstop; devices are keyed by uid, and a line prints only when what reads back changed.
- **Not run on hardware yet:** the *set* paths of `default`, `volume` and `mute` (setting, or the
  no-op `changed:false` path). Only reads and refusals were exercised, so as not to disturb a live
  setup; `watch` events (needs a device or default change) also haven't fired in a test.
- **Aliases** store the device *uid* (`~/.config/macos-gui/audio_aliases.json`,
  `{"scarlett": "AppleUSBAudioEngine:..."}`), so renames, replugs and index shifts don't break them.
  Hand-edited name values also resolve. Alias names: letters, digits, `.`, `_`, `-`; not all digits,
  not a present device's uid. A malformed alias file is ignored with a warning, and `alias set`/`rm`
  refuse to overwrite it. No other state is kept.

## Permission (TCC)

Microphone access (record, level) belongs to the **responsible process**: run from a terminal, that
is the terminal app (or `tmux`, if its server was started outside a terminal); run by an app
(TopicGet.app), it is that app. macOS hands a process without the grant silence, not an error, so:

- `record`/`level` check `AVCaptureDevice.authorizationStatus(for: .audio)` first: denied or
  restricted is exit 1 with that explanation; not determined asks (the prompt names the responsible
  app) and waits up to 60 s. `-v` prints the state.
- A capture in which every sample is exactly 0 warns on stderr: normal for an idle loopback or
  digital input, otherwise a sign the microphone is being withheld.
- The binary embeds an Info.plist (`-sectcreate __TEXT __info_plist`, see `Info.plist`) with
  `NSMicrophoneUsageDescription`, so it can get its own prompt when it is its own responsible process
  (e.g. launched by launchd). **A parent .app that runs `audio-tool record`/`level` (TopicGet.app)
  needs `NSMicrophoneUsageDescription` in its own Info.plist and the user's Microphone grant.**
- `play`, `list`, `status`, `default`, `volume`, `mute` and `watch` need no permission.

## Build

`scripts/install.sh` compiles it when the source or `Info.plist` is newer than the binary:

```sh
swiftc -O tools/audio-tool/audio-tool.swift -o tools/audio-tool/audio-tool \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker tools/audio-tool/Info.plist
```

Each rebuild changes the ad-hoc signature, so a TCC grant made to the binary itself may need
re-granting. Uses the Synchronization module (macOS 15+).

## Examples

```sh
audio-tool list
audio-tool list -j | jq -r '.devices[] | select(.transport=="usb") | .uid'
audio-tool alias set scarlett "Scarlett 2i2"
audio-tool play ding.wav --to scarlett --channels 1 --gain -6
audio-tool play speech.m4a --to BuiltInSpeakerDevice -j
audio-tool record --from scarlett --channels 1 --until-silence 1.5 -o phrase.wav -j
audio-tool record --from scarlett -c 3,4 -s 10 --rate 44100 -o loopback.wav
audio-tool level --from scarlett --watch
audio-tool level --from scarlett -c 1 -s 1 -j | jq '.channels[0].peak_dbfs'
audio-tool volume speakers 30
audio-tool mute BuiltInMicrophoneDevice toggle --input
audio-tool default output speakers
audio-tool watch -j | while read -r ev; do echo "$ev" | jq -r .event; done
```
