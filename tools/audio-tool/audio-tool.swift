// audio-tool - native macOS audio CLI built on CoreAudio.
//
// Commands: list, status, default, volume, mute, play, record, level, watch, alias.
// -j/--json output is a contract other programs parse (see README.md). Every verb that
// changes something is a read-back transaction: read the state, act only if it differs,
// read it back, and report what was READ, never what was requested. play, record and level
// run a raw IOProc on the device you name, so they never touch the system defaults.

import Foundation
import CoreAudio
import AudioToolbox
import AVFoundation
import Synchronization

let VERSION = "0.1.0"
let TOOL_NAME = "audio-tool"

enum ExitCode {
    static let ok: Int32 = 0
    static let failed: Int32 = 1
    static let usage: Int32 = 2         // also: an ambiguous target
    static let refused: Int32 = 3       // the device can't (no such control or channel), or a guard said no
    static let unconfirmed: Int32 = 4   // timed out, or the state never read back
    static func signal(_ s: Int32) -> Int32 { 128 + s }   // play/record/level cut short: 130 SIGINT, 143 SIGTERM
}

// MARK: - Output

var jsonMode = false
var verboseMode = false
var helpTopic = ""   // command being parsed, for "Run '... --help'" hints
let env = ProcessInfo.processInfo.environment
let colorOK = isatty(STDOUT_FILENO) == 1 && env["NO_COLOR"] == nil && env["TERM"] != "dumb"

/// Ordered JSON value, so output keys appear in contract order.
enum JSON {
    case str(String), int(Int), num(Double), bool(Bool), null, arr([JSON]), obj([(String, JSON)])

    var text: String {
        switch self {
        case .str(let s): return JSON.quote(s)
        case .int(let i): return String(i)
        case .num(let d): return JSON.number(d)
        case .bool(let b): return b ? "true" : "false"
        case .null: return "null"
        case .arr(let a): return "[" + a.map(\.text).joined(separator: ",") + "]"
        case .obj(let o): return "{" + o.map { JSON.quote($0.0) + ":" + $0.1.text }.joined(separator: ",") + "}"
        }
    }

    /// Whole numbers print without a fraction (48000); -inf and NaN print null.
    static func number(_ d: Double) -> String {
        guard d.isFinite else { return "null" }
        if d == d.rounded() && abs(d) < 1e15 { return String(Int(d)) }
        return String(d)
    }

    static func quote(_ s: String) -> String {
        var out = "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if u.value < 0x20 || u.value == 0x7F || u.value == 0x2028 || u.value == 0x2029 {
                    out += String(format: "\\u%04x", u.value)
                } else {
                    out.unicodeScalars.append(u)
                }
            }
        }
        return out + "\""
    }
}

func emit(_ j: JSON) { print(j.text) }
func say(_ s: String) { print(s) }
func warn(_ s: String) { fputs(s + "\n", stderr) }   // stdio: a closed stderr (2>&-) is ignored, not fatal
func narrate(_ s: String) { if !jsonMode || verboseMode { warn(s) } }
func vlog(_ s: String) { if verboseMode { warn("\(TOOL_NAME): \(s)") } }
func paint(_ s: String, _ sgr: String) -> String { colorOK ? "\u{1B}[\(sgr)m\(s)\u{1B}[0m" : s }
func pad(_ s: String, _ w: Int, right: Bool = false) -> String {
    let n = s.count
    if n >= w { return s }
    let fill = String(repeating: " ", count: w - n)
    return right ? fill + s : s + fill
}
func ms(since t: Date) -> Int { Int((Date().timeIntervalSince(t) * 1000).rounded()) }
func secs(_ x: Double) -> String { x == x.rounded() ? String(Int(x)) : String(format: "%.1f", x) }
func roundTo(_ x: Double, _ places: Int) -> Double { let m = pow(10.0, Double(places)); return (x * m).rounded() / m }
func secondsJSON(_ x: Double) -> JSON { .num(roundTo(x, 3)) }
func dbfs(_ amplitude: Double) -> Double { amplitude > 0 ? 20 * log10(amplitude) : -Double.infinity }
func dbJSON(_ x: Double) -> JSON { x.isFinite ? .num(roundTo(x, 2)) : .null }
func dbText(_ x: Double) -> String { x.isFinite ? String(format: "%.1f", x) : "-inf" }
func kHz(_ r: Double) -> String { r == 0 ? "?" : (r / 1000 == (r / 1000).rounded() ? "\(Int(r / 1000)) kHz" : String(format: "%g kHz", r / 1000)) }

/// Run (newest first) by fail() before it exits: undo what a command has set up so far, such as
/// a recording file that no audio will reach.
var failCleanups: [() -> Void] = []

func fail(_ code: Int32, _ msg: String, device: Dev? = nil, extra: [(String, JSON)] = []) -> Never {
    let cleanups = failCleanups
    failCleanups = []
    cleanups.reversed().forEach { $0() }
    if jsonMode {
        var o: [(String, JSON)] = [("success", .bool(false)), ("error", .str(msg))]
        if let d = device { o.append(("device", d.stub)) }
        emit(.obj(o + extra))
    } else {
        warn("error: \(msg)")
    }
    exit(code)
}

func usageError(_ msg: String) -> Never {
    if !jsonMode {
        warn("error: \(msg)")
        warn("Run '\(TOOL_NAME) \(helpTopic.isEmpty ? "" : helpTopic + " ")--help' for usage.")
        exit(ExitCode.usage)
    }
    fail(ExitCode.usage, msg)
}

// MARK: - CoreAudio property access

let systemObject = AudioObjectID(kAudioObjectSystemObject)
let noObject: AudioObjectID = 0   // kAudioObjectUnknown

func paddr(_ selector: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
           _ element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
}

func caHas(_ obj: AudioObjectID, _ a: AudioObjectPropertyAddress) -> Bool {
    var a = a
    return AudioObjectHasProperty(obj, &a)
}

func caSettable(_ obj: AudioObjectID, _ a: AudioObjectPropertyAddress) -> Bool {
    var a = a
    var b: DarwinBoolean = false
    return AudioObjectIsPropertySettable(obj, &a, &b) == noErr && b.boolValue
}

func caGet<T: BitwiseCopyable>(_ obj: AudioObjectID, _ a: AudioObjectPropertyAddress, _ initial: T) -> T? {
    var a = a
    var v = initial
    var size = UInt32(MemoryLayout<T>.size)
    return AudioObjectGetPropertyData(obj, &a, 0, nil, &size, &v) == noErr ? v : nil
}

func caSet<T: BitwiseCopyable>(_ obj: AudioObjectID, _ a: AudioObjectPropertyAddress, _ value: T) -> OSStatus {
    var a = a
    var v = value
    return AudioObjectSetPropertyData(obj, &a, 0, nil, UInt32(MemoryLayout<T>.size), &v)
}

func caString(_ obj: AudioObjectID, _ a: AudioObjectPropertyAddress) -> String? {
    var a = a
    var ref: Unmanaged<CFString>? = nil
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let st = withUnsafeMutablePointer(to: &ref) { AudioObjectGetPropertyData(obj, &a, 0, nil, &size, $0) }
    guard st == noErr, let r = ref else { return nil }
    return r.takeRetainedValue() as String
}

func caIDs(_ obj: AudioObjectID, _ a: AudioObjectPropertyAddress) -> [AudioObjectID]? {
    var a = a
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(obj, &a, 0, nil, &size) == noErr else { return nil }
    let n = Int(size) / MemoryLayout<AudioObjectID>.size
    if n == 0 { return [] }
    var ids = [AudioObjectID](repeating: 0, count: n)
    guard AudioObjectGetPropertyData(obj, &a, 0, nil, &size, &ids) == noErr else { return nil }
    return Array(ids.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))
}

/// Channels per stream (one AudioBuffer each in an IOProc), from the stream configuration.
func streamChannels(_ dev: AudioObjectID, _ scope: Scope) -> [Int] {
    var a = paddr(kAudioDevicePropertyStreamConfiguration, scope.ca)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(dev, &a, 0, nil, &size) == noErr, size > 0 else { return [] }
    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(dev, &a, 0, nil, &size, raw) == noErr else { return [] }
    return UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self)).map { Int($0.mNumberChannels) }
}

func fourCC(_ s: String) -> UInt32 { s.utf8.reduce(0) { ($0 << 8) | UInt32($1) } }

func statusText(_ st: OSStatus) -> String {
    let u = UInt32(bitPattern: st)
    let bytes = [24, 16, 8, 0].map { UInt8((u >> UInt32($0)) & 0xFF) }
    if bytes.allSatisfy({ $0 >= 0x20 && $0 < 0x7F }) { return "'\(String(decoding: bytes, as: UTF8.self))' (\(st))" }
    return String(st)
}

enum Scope {
    case input, output
    var ca: AudioObjectPropertyScope { self == .input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput }
    var word: String { self == .input ? "input" : "output" }
}

enum Role: String, CaseIterable {
    case input, output, system
    var selector: AudioObjectPropertySelector {
        switch self {
        case .input: return kAudioHardwarePropertyDefaultInputDevice
        case .output: return kAudioHardwarePropertyDefaultOutputDevice
        case .system: return kAudioHardwarePropertyDefaultSystemOutputDevice
        }
    }
    var scope: Scope { self == .input ? .input : .output }
    var title: String { self == .input ? "Input" : self == .output ? "Output" : "System" }
    func id(_ d: Defaults) -> AudioObjectID { self == .input ? d.input : self == .output ? d.output : d.system }
}

struct Defaults { let input: AudioObjectID; let output: AudioObjectID; let system: AudioObjectID }

func readDefaults() -> Defaults {
    func get(_ r: Role) -> AudioObjectID { caGet(systemObject, paddr(r.selector), noObject) ?? noObject }
    return Defaults(input: get(.input), output: get(.output), system: get(.system))
}

// MARK: - Devices

struct Dev {
    var index = 0
    let id: AudioObjectID
    let uid: String
    let name: String
    let manufacturer: String
    let transport: String
    let inputs: Int
    let outputs: Int
    let inputNames: [String]
    let outputNames: [String]
    let rate: Double
    let isDefaultInput: Bool
    let isDefaultOutput: Bool
    let isDefaultSystem: Bool
    let alive: Bool
    var alias: String? = nil

    var label: String { "\(name) (\(uid))" }
    var stub: JSON { .obj([("uid", .str(uid)), ("name", .str(name))]) }
    var listed: JSON { .obj([("index", .int(index)), ("uid", .str(uid)), ("name", .str(name))]) }
    func channels(_ s: Scope) -> Int { s == .input ? inputs : outputs }
    var json: JSON {
        .obj([("index", .int(index)), ("id", .int(Int(id))), ("uid", .str(uid)), ("name", .str(name)),
              ("manufacturer", .str(manufacturer)), ("transport", .str(transport)),
              ("inputs", .int(inputs)), ("outputs", .int(outputs)),
              ("input_channels", .arr(inputNames.map { .str($0) })),
              ("output_channels", .arr(outputNames.map { .str($0) })),
              ("rate", .num(rate)), ("default_input", .bool(isDefaultInput)),
              ("default_output", .bool(isDefaultOutput)), ("default_system_output", .bool(isDefaultSystem)),
              ("alive", .bool(alive)), ("alias", alias.map { .str($0) } ?? .null)])
    }
}

/// kAudioDevicePropertyTransportType as one lowercase word.
func transportWord(_ t: UInt32) -> String {
    switch t {
    case fourCC("bltn"): return "builtin"
    case fourCC("usb "): return "usb"
    case fourCC("blue"), fourCC("blea"): return "bluetooth"
    case fourCC("ccwd"), fourCC("ccwl"), fourCC("cont"): return "continuity"
    case fourCC("grup"), fourCC("fgrp"): return "aggregate"
    case fourCC("virt"): return "virtual"
    case fourCC("hdmi"): return "hdmi"
    case fourCC("dprt"): return "displayport"
    case fourCC("airp"): return "airplay"
    case fourCC("thun"): return "thunderbolt"
    case fourCC("pci "): return "pci"
    case fourCC("1394"): return "firewire"
    case fourCC("eavb"): return "avb"
    default: return "unknown"
    }
}

/// kAudioObjectPropertyElementName per channel; [] when the device names none of them.
func channelNames(_ id: AudioObjectID, _ scope: Scope, _ count: Int) -> [String] {
    guard count > 0 else { return [] }
    let names = (1...count).map { (caString(id, paddr(kAudioObjectPropertyElementName, scope.ca, UInt32($0))) ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines) }
    return names.allSatisfy(\.isEmpty) ? [] : names
}

func readDev(_ id: AudioObjectID, _ d: Defaults) -> Dev {
    let uid = caString(id, paddr(kAudioDevicePropertyDeviceUID)) ?? ""
    let n = (caString(id, paddr(kAudioObjectPropertyName)) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    let ins = streamChannels(id, .input).reduce(0, +)
    let outs = streamChannels(id, .output).reduce(0, +)
    return Dev(id: id, uid: uid, name: n.isEmpty ? (uid.isEmpty ? "device \(id)" : uid) : n,
               manufacturer: (caString(id, paddr(kAudioObjectPropertyManufacturer)) ?? "").trimmingCharacters(in: .whitespaces),
               transport: transportWord(caGet(id, paddr(kAudioDevicePropertyTransportType), UInt32(0)) ?? 0),
               inputs: ins, outputs: outs, inputNames: channelNames(id, .input, ins), outputNames: channelNames(id, .output, outs),
               rate: caGet(id, paddr(kAudioDevicePropertyNominalSampleRate), Float64(0)) ?? 0,
               isDefaultInput: id == d.input, isDefaultOutput: id == d.output, isDefaultSystem: id == d.system,
               alive: (caGet(id, paddr(kAudioDevicePropertyDeviceIsAlive), UInt32(0)) ?? 0) != 0)
}

/// Case-, diacritic- and width-insensitive, with typographic quotes folded (Joe’s matches Joe's).
func fold(_ s: String) -> String {
    var t = s
    for (a, b) in [("\u{2018}", "'"), ("\u{2019}", "'"), ("\u{201C}", "\""), ("\u{201D}", "\"")] {
        t = t.replacingOccurrences(of: a, with: b)
    }
    return t.folding(options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive], locale: nil)
}

// MARK: - Aliases (~/.config/macos-gui/audio_aliases.json: {"alias": "<device uid>"})

let home = env["HOME"] ?? NSHomeDirectory()
let aliasPath = home + "/.config/macos-gui/audio_aliases.json"

enum AliasFile { case ok([String: String]), broken(String) }

func readAliasFile() -> AliasFile {
    guard let data = FileManager.default.contents(atPath: aliasPath) else { return .ok([:]) }
    if String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return .ok([:]) }
    guard let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        return .broken("\(aliasPath) is not a JSON object")
    }
    var out: [String: String] = [:]
    for (k, v) in dict { if let s = v as? String { out[k.lowercased()] = s } }
    return .ok(out)
}

var aliasCache: [String: String]? = nil
func aliases() -> [String: String] {
    if let c = aliasCache { return c }
    switch readAliasFile() {
    case .ok(let a): aliasCache = a
    case .broken(let why): warn("warning: \(why); ignoring aliases"); aliasCache = [:]
    }
    return aliasCache!
}

func writeAliases(_ a: [String: String]) throws {
    let dir = (aliasPath as NSString).deletingLastPathComponent
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    let data = a.isEmpty ? Data("{}".utf8)
        : try JSONSerialization.data(withJSONObject: a, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try (data + Data("\n".utf8)).write(to: URL(fileURLWithPath: aliasPath), options: .atomic)
    aliasCache = nil
}

// MARK: - Snapshot & target resolution

struct Snapshot { let devices: [Dev]; let defaults: Defaults }

/// Stored alias value: uid (exact, then any case), else exact name, else unique name substring.
func lookupStored(_ value: String, _ devs: [Dev]) -> Dev? {
    if let d = devs.first(where: { $0.uid == value }) ?? devs.first(where: { $0.uid.caseInsensitiveCompare(value) == .orderedSame }) {
        return d
    }
    let f = fold(value)
    let exact = devs.filter { fold($0.name) == f }
    if exact.count == 1 { return exact[0] }
    let hits = devs.filter { fold($0.name).contains(f) }
    return hits.count == 1 ? hits[0] : nil
}

func buildDevices(_ ids: [AudioObjectID], _ d: Defaults) -> [Dev] {
    var devs = ids.map { readDev($0, d) }
    devs.sort { (fold($0.name), $0.uid) < (fold($1.name), $1.uid) }
    for i in devs.indices { devs[i].index = i + 1 }
    for (name, value) in aliases().sorted(by: { $0.key < $1.key }) {
        if let d = lookupStored(value, devs), let i = devs.firstIndex(where: { $0.id == d.id }), devs[i].alias == nil {
            devs[i].alias = name
        }
    }
    return devs
}

func takeSnapshot() -> Snapshot {
    let t0 = Date()
    guard let ids = caIDs(systemObject, paddr(kAudioHardwarePropertyDevices)) else {
        fail(ExitCode.failed, "CoreAudio returned no device list (is coreaudiod running?)")
    }
    let d = readDefaults()
    let devs = buildDevices(ids, d)
    vlog("\(devs.count) devices in \(ms(since: t0)) ms; default input \(d.input), output \(d.output), system \(d.system)")
    return Snapshot(devices: devs, defaults: d)
}

/// Target = 1-based index | alias | uid (exact) | name (exact, then substring; case-insensitive).
/// Several name matches narrow to the devices that have `need` channels, if that leaves one.
func resolve(_ query: String, _ snap: Snapshot, need: Scope? = nil) -> Dev {
    let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
    if q.isEmpty { usageError("empty target") }
    let devs = snap.devices
    let there = devs.isEmpty ? "there are no audio devices"
        : "devices: " + devs.map { "\($0.index) \($0.name)" }.joined(separator: ", ")
    let listed: [(String, JSON)] = [("devices", .arr(devs.map(\.listed)))]
    if q.allSatisfy({ $0.isASCII && $0.isNumber }) {
        guard let n = Int(q), n >= 1, n <= devs.count else {
            fail(ExitCode.failed, "no device at index \(q) (\(there))", extra: listed)
        }
        return devs[n - 1]
    }
    if let v = aliases()[q.lowercased()] {
        if let d = lookupStored(v, devs) { return d }
        fail(ExitCode.failed, "alias '\(q.lowercased())' points to '\(v)', which matches no device (\(there))", extra: listed)
    }
    if let d = devs.first(where: { $0.uid == q }) ?? devs.first(where: { $0.uid.caseInsensitiveCompare(q) == .orderedSame }) {
        return d
    }
    let f = fold(q)
    let exact = devs.filter { fold($0.name) == f }
    if exact.count == 1 { return exact[0] }
    var hits = exact.isEmpty ? devs.filter { fold($0.name).contains(f) } : exact
    if hits.count > 1, let need {
        let able = hits.filter { $0.channels(need) > 0 }
        if able.count == 1 { return able[0] }
        if able.count > 1 { hits = able }
    }
    if hits.count == 1 { return hits[0] }
    if hits.isEmpty { fail(ExitCode.failed, "no device matches '\(q)' (\(there))", extra: listed) }
    fail(ExitCode.usage, "ambiguous target '\(q)' matches \(hits.count) devices: "
         + hits.map { "\($0.index) \($0.label)" }.joined(separator: "; ") + "; use the index, the uid or an alias",
         extra: [("matches", .arr(hits.map(\.listed)))])
}

func defaultOf(_ role: Role, _ snap: Snapshot) -> Dev? { snap.devices.first { $0.id == role.id(snap.defaults) } }

// MARK: - Volume & mute controls

enum ControlKind { case virtualMain, main, channels }
struct Control { let kind: ControlKind; let elements: [AudioObjectPropertyElement]; let settable: Bool }

/// The control macOS's own volume slider uses (virtual main volume), else the main element, else per channel.
func volumeControl(_ id: AudioObjectID, _ s: Scope, channels: Int) -> Control? {
    let v = paddr(kAudioHardwareServiceDeviceProperty_VirtualMainVolume, s.ca)
    if caHas(id, v) { return Control(kind: .virtualMain, elements: [kAudioObjectPropertyElementMain], settable: caSettable(id, v)) }
    let m = paddr(kAudioDevicePropertyVolumeScalar, s.ca)
    if caHas(id, m) { return Control(kind: .main, elements: [kAudioObjectPropertyElementMain], settable: caSettable(id, m)) }
    guard channels > 0 else { return nil }
    let els = (1...channels).map { UInt32($0) }.filter { caHas(id, paddr(kAudioDevicePropertyVolumeScalar, s.ca, $0)) }
    if els.isEmpty { return nil }
    return Control(kind: .channels, elements: els,
                   settable: els.allSatisfy { caSettable(id, paddr(kAudioDevicePropertyVolumeScalar, s.ca, $0)) })
}

func readVolume(_ id: AudioObjectID, _ s: Scope, _ c: Control) -> Float32? {
    if c.kind == .virtualMain { return caGet(id, paddr(kAudioHardwareServiceDeviceProperty_VirtualMainVolume, s.ca), Float32(0)) }
    let vals = c.elements.compactMap { caGet(id, paddr(kAudioDevicePropertyVolumeScalar, s.ca, $0), Float32(0)) }
    return vals.isEmpty ? nil : vals.reduce(0, +) / Float32(vals.count)
}

func writeVolume(_ id: AudioObjectID, _ s: Scope, _ c: Control, _ v: Float32) -> OSStatus {
    if c.kind == .virtualMain { return caSet(id, paddr(kAudioHardwareServiceDeviceProperty_VirtualMainVolume, s.ca), v) }
    for e in c.elements {
        let st = caSet(id, paddr(kAudioDevicePropertyVolumeScalar, s.ca, e), v)
        if st != noErr { return st }
    }
    return noErr
}

func muteControl(_ id: AudioObjectID, _ s: Scope, channels: Int) -> Control? {
    let m = paddr(kAudioDevicePropertyMute, s.ca)
    if caHas(id, m) { return Control(kind: .main, elements: [kAudioObjectPropertyElementMain], settable: caSettable(id, m)) }
    guard channels > 0 else { return nil }
    let els = (1...channels).map { UInt32($0) }.filter { caHas(id, paddr(kAudioDevicePropertyMute, s.ca, $0)) }
    if els.isEmpty { return nil }
    return Control(kind: .channels, elements: els,
                   settable: els.allSatisfy { caSettable(id, paddr(kAudioDevicePropertyMute, s.ca, $0)) })
}

/// Muted when every element of the control reads muted.
func readMute(_ id: AudioObjectID, _ s: Scope, _ c: Control) -> Bool? {
    let vals = c.elements.compactMap { caGet(id, paddr(kAudioDevicePropertyMute, s.ca, $0), UInt32(0)) }
    return vals.count == c.elements.count ? vals.allSatisfy { $0 != 0 } : nil
}

func writeMute(_ id: AudioObjectID, _ s: Scope, _ c: Control, _ on: Bool) -> OSStatus {
    for e in c.elements {
        let st = caSet(id, paddr(kAudioDevicePropertyMute, s.ca, e), UInt32(on ? 1 : 0))
        if st != noErr { return st }
    }
    return noErr
}

func pct(_ v: Float32) -> Int { Int((Double(v) * 100).rounded()) }

func controlText(_ c: Control) -> String {
    switch c.kind {
    case .virtualMain: return "virtual main volume"
    case .main: return "main element"
    case .channels: return "channels " + c.elements.map(String.init).joined(separator: ",")
    }
}

/// Volume (0-100) and mute of a device in a scope, nil where it has no such control.
func volumeAndMute(_ d: Dev, _ s: Scope) -> (Int?, Bool?) {
    let n = d.channels(s)
    let v = volumeControl(d.id, s, channels: n).flatMap { readVolume(d.id, s, $0) }.map(pct)
    let m = muteControl(d.id, s, channels: n).flatMap { readMute(d.id, s, $0) }
    return (v, m)
}

// MARK: - Waiting

/// Re-reads `check` until it holds or `timeout` passes.
func poll(timeout: TimeInterval, every: TimeInterval = 0.02, _ check: () -> Bool) -> Bool {
    let deadline = Date(timeIntervalSinceNow: timeout)
    while true {
        if check() { return true }
        let left = deadline.timeIntervalSinceNow
        if left <= 0 { return false }
        usleep(useconds_t(min(every, left) * 1_000_000))
    }
}

func timeoutValue(_ a: Args, default d: Double = 5) -> Double {
    guard let s = a.values["timeout"] else { return d }
    guard let v = Double(s), v > 0, v <= 86400 else { usageError("--timeout takes seconds (0 < s <= 86400), not '\(s)'") }
    return v
}

// MARK: - list, status

func printTable(_ devs: [Dev]) {
    let idxW = String(devs.map(\.index).max() ?? 1).count
    let nameW = max(4, devs.map { $0.name.count + ($0.alive ? 0 : 6) }.max() ?? 4)
    let trW = max(9, devs.map { $0.transport.count }.max() ?? 9)
    func defaults(_ d: Dev) -> String {
        let w = [d.isDefaultInput ? "input" : nil, d.isDefaultOutput ? "output" : nil, d.isDefaultSystem ? "system" : nil]
            .compactMap { $0 }.joined(separator: ",")
        return w.isEmpty ? "-" : w
    }
    let defW = max(7, devs.map { defaults($0).count }.max() ?? 7)
    let alW = max(5, devs.map { ($0.alias ?? "-").count }.max() ?? 5)
    let rateW = max(4, devs.map { kHz($0.rate).count }.max() ?? 4)
    say("  " + pad("#", idxW, right: true) + "  " + pad("NAME", nameW) + "  " + pad("TRANSPORT", trW) + "  "
        + pad("IN", 3, right: true) + "  " + pad("OUT", 3, right: true) + "  " + pad("RATE", rateW) + "  "
        + pad("DEFAULT", defW) + "  " + pad("ALIAS", alW) + "  UID")
    let indent = String(repeating: " ", count: idxW + 4)
    for d in devs {
        let def = defaults(d)
        say("  " + pad(String(d.index), idxW, right: true) + "  " + pad(d.name + (d.alive ? "" : " (dead)"), nameW) + "  "
            + pad(d.transport, trW) + "  " + pad(String(d.inputs), 3, right: true) + "  "
            + pad(String(d.outputs), 3, right: true) + "  " + pad(kHz(d.rate), rateW) + "  "
            + (def == "-" ? pad(def, defW) : paint(pad(def, defW), "32")) + "  " + pad(d.alias ?? "-", alW) + "  " + d.uid)
        // Channel names only when they say more than the channel number.
        for (word, names) in [("inputs: ", d.inputNames), ("outputs:", d.outputNames)] where !names.isEmpty {
            if names.enumerated().allSatisfy({ $0.element.isEmpty || $0.element == String($0.offset + 1) }) { continue }
            say(indent + word + " " + names.enumerated().map { "\($0.offset + 1) \($0.element.isEmpty ? "-" : $0.element)" }
                .joined(separator: ", "))
        }
    }
}

func cmdList(_ a: Args) {
    if let extra = a.pos.first { usageError("unexpected argument '\(extra)'") }
    let snap = takeSnapshot()
    if jsonMode {
        emit(.obj([("devices", .arr(snap.devices.map(\.json)))]))
        return
    }
    if snap.devices.isEmpty { say("No audio devices."); return }
    printTable(snap.devices)
}

func cmdStatus(_ a: Args) {
    if let extra = a.pos.first { usageError("unexpected argument '\(extra)'") }
    let snap = takeSnapshot()
    var parts: [(String, JSON)] = []
    var lines: [String] = []
    for role in Role.allCases {
        let d = defaultOf(role, snap)
        let (v, m) = d.map { volumeAndMute($0, role.scope) } ?? (nil, nil)
        parts.append((role.rawValue, .obj([("device", d?.json ?? .null), ("volume", v.map { .int($0) } ?? .null),
                                           ("mute", m.map { .bool($0) } ?? .null)])))
        let what = d.map { "\($0.name)  (\($0.uid))" } ?? "none"
        lines.append(pad(role.title + ":", 8) + " " + what + (d == nil ? "" : "  volume \(v.map(String.init) ?? "-")  mute "
            + (m.map { $0 ? paint("on", "33") : "off" } ?? "-")))
    }
    if jsonMode { emit(.obj(parts)) } else { lines.forEach(say) }
}

// MARK: - default

func cmdDefault(_ a: Args) {
    guard let first = a.pos.first?.lowercased() else { usageError("default needs a role: input, output or system") }
    guard let role = Role(rawValue: first) else { usageError("role must be input, output or system, not '\(a.pos[0])'") }
    if a.pos.count > 2 { usageError("unexpected argument '\(a.pos[2])'") }
    let timeout = timeoutValue(a)
    let t0 = Date()
    let snap = takeSnapshot()
    let was = defaultOf(role, snap)
    guard a.pos.count == 2 else {
        if jsonMode { emit(.obj([("role", .str(role.rawValue)), ("device", was?.json ?? .null)])) }
        else { say(was.map(\.label) ?? "none") }
        return
    }
    let dev = resolve(a.pos[1], snap, need: role.scope)
    func report(_ changed: Bool, _ now: Dev?) {
        if jsonMode {
            emit(.obj([("success", .bool(true)), ("role", .str(role.rawValue)), ("changed", .bool(changed)),
                       ("was", was?.stub ?? .null), ("now", now?.stub ?? .null), ("ms", .int(ms(since: t0)))]))
        } else if changed {
            say("Default \(role.rawValue): \(was?.name ?? "none") -> \(now?.label ?? "none") (\(ms(since: t0)) ms)")
        } else {
            say("Default \(role.rawValue) already \(dev.label); nothing sent")
        }
    }
    if was?.id == dev.id { report(false, dev); return }
    if dev.channels(role.scope) == 0 {
        fail(ExitCode.refused, "'\(dev.name)' has no \(role.scope.word)s, so it can't be the default \(role.rawValue)", device: dev)
    }
    let canSel = role == .system ? kAudioDevicePropertyDeviceCanBeDefaultSystemDevice : kAudioDevicePropertyDeviceCanBeDefaultDevice
    if let can = caGet(dev.id, paddr(canSel, role.scope.ca), UInt32(1)), can == 0 {
        fail(ExitCode.refused, "'\(dev.name)' says it can't be the default \(role.rawValue) device", device: dev)
    }
    narrate("Setting the default \(role.rawValue) to \(dev.label)...")
    let st = caSet(systemObject, paddr(role.selector), dev.id)
    vlog("AudioObjectSetPropertyData(default \(role.rawValue) = \(dev.id)) -> \(statusText(st))")
    if st != noErr { fail(ExitCode.failed, "setting the default \(role.rawValue) failed: \(statusText(st))", device: dev) }
    let ok = poll(timeout: timeout) { role.id(readDefaults()) == dev.id }
    let nowID = role.id(readDefaults())
    let now = buildDevices(caIDs(systemObject, paddr(kAudioHardwarePropertyDevices)) ?? [], readDefaults()).first { $0.id == nowID }
    guard ok else {
        fail(ExitCode.unconfirmed, "unconfirmed: asked for \(dev.name) but the default \(role.rawValue) still reads "
             + "\(now?.name ?? "none") after \(secs(timeout)) s", device: dev)
    }
    report(true, now)
}

// MARK: - volume, mute

struct ScopedTarget { let dev: Dev; let scope: Scope; let channels: Int }

func scopedTarget(_ verb: String, _ a: Args) -> ScopedTarget {
    guard let target = a.pos.first else { usageError("\(verb) needs a target (index, alias, uid or name)") }
    let scope: Scope = a.flags.contains("input") ? .input : .output
    let dev = resolve(target, takeSnapshot(), need: scope)
    let n = dev.channels(scope)
    if n == 0 {
        let hint = scope == .output && dev.inputs > 0 ? "; pass --input for its input \(verb)" : ""
        fail(ExitCode.refused, "'\(dev.name)' has no \(scope.word)s\(hint)", device: dev)
    }
    return ScopedTarget(dev: dev, scope: scope, channels: n)
}

func cmdVolume(_ a: Args) {
    if a.pos.count > 2 { usageError("unexpected argument '\(a.pos[2])'") }
    var want: Int? = nil
    if a.pos.count == 2 {
        let s = a.pos[1].hasSuffix("%") ? String(a.pos[1].dropLast()) : a.pos[1]
        guard let n = Int(s), (0...100).contains(n) else { usageError("volume takes a whole number 0-100, not '\(a.pos[1])'") }
        want = n
    }
    let timeout = timeoutValue(a)
    let t0 = Date()
    let t = scopedTarget("volume", a)
    let (dev, scope) = (t.dev, t.scope)
    guard let ctl = volumeControl(dev.id, scope, channels: t.channels) else {
        fail(ExitCode.refused, "'\(dev.name)' has no \(scope.word) volume control in CoreAudio", device: dev,
             extra: [("scope", .str(scope.word))])
    }
    guard let raw = readVolume(dev.id, scope, ctl) else {
        fail(ExitCode.failed, "can't read the \(scope.word) volume of '\(dev.name)'", device: dev)
    }
    let was = pct(raw)
    vlog("\(dev.name) \(scope.word) volume via \(controlText(ctl)): \(raw) (\(was)), settable: \(ctl.settable)")
    guard let want else {
        if jsonMode { emit(.obj([("device", dev.stub), ("scope", .str(scope.word)), ("volume", .int(was))])) }
        else { say(String(was)) }
        return
    }
    func report(_ changed: Bool, _ now: Int) {
        if jsonMode {
            emit(.obj([("success", .bool(true)), ("device", dev.stub), ("scope", .str(scope.word)), ("changed", .bool(changed)),
                       ("was", .int(was)), ("now", .int(now)), ("ms", .int(ms(since: t0)))]))
        } else if changed {
            say("\(dev.name) \(scope.word) volume \(was) -> \(now) (\(ms(since: t0)) ms)")
        } else {
            say("\(dev.name) \(scope.word) volume already \(was); nothing sent")
        }
    }
    if was == want { report(false, was); return }
    if !ctl.settable { fail(ExitCode.refused, "the \(scope.word) volume of '\(dev.name)' is read-only", device: dev) }
    let st = writeVolume(dev.id, scope, ctl, Float32(want) / 100)
    vlog("set \(controlText(ctl)) = \(Float32(want) / 100) -> \(statusText(st))")
    if st != noErr { fail(ExitCode.failed, "setting the \(scope.word) volume of '\(dev.name)' failed: \(statusText(st))", device: dev) }
    // Hardware steps can land a point off what was asked; within 1 counts, and `now` is what reads back.
    let ok = poll(timeout: timeout) { readVolume(dev.id, scope, ctl).map { abs(pct($0) - want) <= 1 } ?? false }
    let now = readVolume(dev.id, scope, ctl).map(pct) ?? -1
    guard ok else {
        fail(ExitCode.unconfirmed, "unconfirmed: asked for \(scope.word) volume \(want) on '\(dev.name)' but it reads "
             + "\(now) after \(secs(timeout)) s", device: dev)
    }
    report(true, now)
}

func cmdMute(_ a: Args) {
    if a.pos.count > 2 { usageError("unexpected argument '\(a.pos[2])'") }
    let arg = a.pos.count == 2 ? a.pos[1].lowercased() : nil
    if let arg, !["on", "off", "toggle"].contains(arg) { usageError("mute takes on, off or toggle, not '\(a.pos[1])'") }
    let timeout = timeoutValue(a)
    let t0 = Date()
    let t = scopedTarget("mute", a)
    let (dev, scope) = (t.dev, t.scope)
    guard let ctl = muteControl(dev.id, scope, channels: t.channels) else {
        fail(ExitCode.refused, "'\(dev.name)' has no \(scope.word) mute control in CoreAudio", device: dev,
             extra: [("scope", .str(scope.word))])
    }
    guard let was = readMute(dev.id, scope, ctl) else {
        fail(ExitCode.failed, "can't read the \(scope.word) mute of '\(dev.name)'", device: dev)
    }
    vlog("\(dev.name) \(scope.word) mute via \(controlText(ctl)): \(was), settable: \(ctl.settable)")
    func word(_ b: Bool) -> String { b ? "on" : "off" }
    guard let arg else {
        if jsonMode { emit(.obj([("device", dev.stub), ("scope", .str(scope.word)), ("mute", .bool(was))])) }
        else { say(word(was)) }
        return
    }
    let want = arg == "toggle" ? !was : arg == "on"
    func report(_ changed: Bool, _ now: Bool) {
        if jsonMode {
            emit(.obj([("success", .bool(true)), ("device", dev.stub), ("scope", .str(scope.word)), ("changed", .bool(changed)),
                       ("was", .bool(was)), ("now", .bool(now)), ("ms", .int(ms(since: t0)))]))
        } else if changed {
            say("\(dev.name) \(scope.word) mute \(word(was)) -> \(word(now)) (\(ms(since: t0)) ms)")
        } else {
            say("\(dev.name) \(scope.word) mute already \(word(was)); nothing sent")
        }
    }
    if was == want { report(false, was); return }
    if !ctl.settable { fail(ExitCode.refused, "the \(scope.word) mute of '\(dev.name)' is read-only", device: dev) }
    let st = writeMute(dev.id, scope, ctl, want)
    vlog("set mute \(controlText(ctl)) = \(want) -> \(statusText(st))")
    if st != noErr { fail(ExitCode.failed, "setting the \(scope.word) mute of '\(dev.name)' failed: \(statusText(st))", device: dev) }
    let ok = poll(timeout: timeout) { readMute(dev.id, scope, ctl) == want }
    let now = readMute(dev.id, scope, ctl) ?? was
    guard ok else {
        fail(ExitCode.unconfirmed, "unconfirmed: asked for \(scope.word) mute \(word(want)) on '\(dev.name)' but it reads "
             + "\(word(now)) after \(secs(timeout)) s", device: dev)
    }
    report(true, now)
}

// MARK: - Real-time plumbing: ring buffer, IOProc session, signals

/// Single-producer single-consumer ring of interleaved float frames. One side may be the
/// HAL's IO thread: no locks or allocation, just two monotonic frame counters.
final class Ring {
    let channels: Int
    let capacity: Int   // frames
    let data: UnsafeMutablePointer<Float>
    let head = Atomic<Int>(0)   // frames written (producer)
    let tail = Atomic<Int>(0)   // frames read (consumer)

    init(frames: Int, channels: Int) {
        self.channels = max(1, channels)
        capacity = max(1, frames)
        data = .allocate(capacity: capacity * self.channels)
        data.initialize(repeating: 0, count: capacity * self.channels)
    }
    deinit { data.deallocate() }

    var readable: Int { head.load(ordering: .acquiring) - tail.load(ordering: .acquiring) }
    var writable: Int { capacity - readable }

    func write(_ src: UnsafePointer<Float>, frames: Int) -> Int {
        let h = head.load(ordering: .relaxed)
        let n = min(frames, capacity - (h - tail.load(ordering: .acquiring)))
        if n <= 0 { return 0 }
        let start = h % capacity
        let first = min(n, capacity - start)
        (data + start * channels).update(from: src, count: first * channels)
        if n > first { data.update(from: src + first * channels, count: (n - first) * channels) }
        head.store(h + n, ordering: .releasing)
        return n
    }

    func read(into dst: UnsafeMutablePointer<Float>, frames: Int) -> Int {
        let t = tail.load(ordering: .relaxed)
        let n = min(frames, head.load(ordering: .acquiring) - t)
        if n <= 0 { return 0 }
        let start = t % capacity
        let first = min(n, capacity - start)
        dst.update(from: data + start * channels, count: first * channels)
        if n > first { (dst + first * channels).update(from: data, count: (n - first) * channels) }
        tail.store(t + n, ordering: .releasing)
        return n
    }
}

/// Where each requested device channel lives in the IOProc's AudioBufferList.
struct ChannelSlot { let buffer: Int; let offset: Int; let stride: Int }

func channelSlots(_ dev: AudioObjectID, _ s: Scope) -> [ChannelSlot] {
    var out: [ChannelSlot] = []
    for (b, n) in streamChannels(dev, s).enumerated() { for c in 0..<n { out.append(ChannelSlot(buffer: b, offset: c, stride: n)) } }
    return out
}

/// nil when every stream of the scope hands IOProcs interleaved 32-bit float (the HAL's virtual format).
func unsupportedFormat(_ dev: AudioObjectID, _ s: Scope) -> String? {
    guard let streams = caIDs(dev, paddr(kAudioDevicePropertyStreams, s.ca)) else { return "can't read its \(s.word) streams" }
    for st in streams {
        guard let f = caGet(st, paddr(kAudioStreamPropertyVirtualFormat), AudioStreamBasicDescription()) else {
            return "can't read the format of \(s.word) stream \(st)"
        }
        vlog("\(s.word) stream \(st): \(f.mChannelsPerFrame) ch, \(f.mBitsPerChannel)-bit, flags 0x\(String(f.mFormatFlags, radix: 16)), "
             + "\(f.mSampleRate) Hz")
        if f.mFormatID != kAudioFormatLinearPCM || f.mFormatFlags & kAudioFormatFlagIsFloat == 0 || f.mBitsPerChannel != 32
            || f.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0 {
            return "\(s.word) stream \(st) isn't interleaved 32-bit float (format \(f.mFormatID), flags \(f.mFormatFlags))"
        }
    }
    return nil
}

/// Turns one direction's streams on or off for one IOProc only (other clients are unaffected),
/// so play never opens the device's inputs (no microphone use) and record never mixes into its outputs.
func setStreamUsage(_ dev: AudioObjectID, _ proc: AudioDeviceIOProcID, _ s: Scope, on: Bool) -> Bool {
    let n = caIDs(dev, paddr(kAudioDevicePropertyStreams, s.ca))?.count ?? 0
    if n == 0 { return true }
    let size = 12 + 4 * n   // void* mIOProc; UInt32 mNumberStreams; UInt32 mStreamIsOn[n]
    let raw = UnsafeMutableRawPointer.allocate(byteCount: (size + 7) & ~7, alignment: 8)
    defer { raw.deallocate() }
    raw.initializeMemory(as: UInt8.self, repeating: 0, count: (size + 7) & ~7)
    raw.storeBytes(of: unsafeBitCast(proc, to: UnsafeRawPointer.self), as: UnsafeRawPointer.self)
    raw.storeBytes(of: UInt32(n), toByteOffset: 8, as: UInt32.self)
    var a = paddr(kAudioDevicePropertyIOProcStreamUsage, s.ca)
    var sz = UInt32(size)
    guard AudioObjectGetPropertyData(dev, &a, 0, nil, &sz, raw) == noErr else { return false }
    for i in 0..<n { raw.storeBytes(of: UInt32(on ? 1 : 0), toByteOffset: 12 + 4 * i, as: UInt32.self) }
    guard AudioObjectSetPropertyData(dev, &a, 0, nil, UInt32(size), raw) == noErr else { return false }
    for i in 0..<n { raw.storeBytes(of: UInt32(7), toByteOffset: 12 + 4 * i, as: UInt32.self) }
    sz = UInt32(size)
    guard AudioObjectGetPropertyData(dev, &a, 0, nil, &sz, raw) == noErr else { return false }
    return (0..<n).allSatisfy { (raw.load(fromByteOffset: 12 + 4 * $0, as: UInt32.self) != 0) == on }
}

/// One IOProc on one device. Never changes the device's rate, buffer size, defaults or hog mode.
final class IOSession {
    let device: AudioObjectID
    var proc: AudioDeviceIOProcID? = nil
    var running = false

    init(_ d: AudioObjectID) { device = d }

    /// nil on success, else why it couldn't start.
    func start(_ direction: Scope, _ block: @escaping AudioDeviceIOBlock) -> String? {
        var pid: AudioDeviceIOProcID? = nil
        let st = AudioDeviceCreateIOProcIDWithBlock(&pid, device, nil, block)
        guard st == noErr, let pid else { return "AudioDeviceCreateIOProcIDWithBlock failed: \(statusText(st))" }
        proc = pid
        let other: Scope = direction == .input ? .output : .input
        if !streamChannels(device, other).isEmpty {
            let ok = setStreamUsage(device, pid, other, on: false)
            vlog("\(other.word) streams off for this IOProc: " + (ok ? "read back" : "NOT confirmed (its buffers are zeroed instead)"))
        }
        let s2 = AudioDeviceStart(device, pid)
        vlog("AudioDeviceStart -> \(statusText(s2))")
        guard s2 == noErr else {
            AudioDeviceDestroyIOProcID(device, pid)
            proc = nil
            return "AudioDeviceStart failed: \(statusText(s2))"
        }
        running = true
        return nil
    }

    func stop() {
        guard let pid = proc else { return }
        if running { let st = AudioDeviceStop(device, pid); vlog("AudioDeviceStop -> \(statusText(st))"); running = false }
        let st = AudioDeviceDestroyIOProcID(device, pid)
        vlog("AudioDeviceDestroyIOProcID -> \(statusText(st))")
        proc = nil
    }
}

final class SignalBox { let caught = Atomic<Int32>(0) }
let signals = SignalBox()
var keepAlive: [AnyObject] = []

var signalsTrapped = false

/// SIGINT/SIGTERM/SIGHUP set a flag the command's loop checks; a second one exits at once.
/// Idempotent: two sources for one signal would read one Ctrl-C as the second.
func trapSignals() {
    if signalsTrapped { return }
    signalsTrapped = true
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        signal(sig, SIG_IGN)
        let s = DispatchSource.makeSignalSource(signal: sig, queue: .global())
        s.setEventHandler {
            if signals.caught.load(ordering: .acquiring) != 0 { exit(ExitCode.signal(sig)) }
            signals.caught.store(sig, ordering: .releasing)
        }
        s.resume()
        keepAlive.append(s)
    }
}

func caughtSignal() -> Int32 { signals.caught.load(ordering: .acquiring) }

func requireAlive(_ dev: Dev) {
    if !dev.alive { fail(ExitCode.failed, "'\(dev.name)' is not alive (unplugged or asleep?)", device: dev) }
    if let hog = caGet(dev.id, paddr(kAudioDevicePropertyHogMode), pid_t(-1)), hog != -1, hog != getpid() {
        fail(ExitCode.refused, "'\(dev.name)' is in exclusive (hog mode) use by pid \(hog)", device: dev)
    }
}

/// Frames between our last write and the speaker: device + stream latency, safety offset, one buffer.
func outputTail(_ dev: AudioObjectID) -> Int {
    let s = kAudioObjectPropertyScopeOutput
    var frames = Int(caGet(dev, paddr(kAudioDevicePropertyLatency, s), UInt32(0)) ?? 0)
    frames += Int(caGet(dev, paddr(kAudioDevicePropertySafetyOffset, s), UInt32(0)) ?? 0)
    frames += Int(caGet(dev, paddr(kAudioDevicePropertyBufferFrameSize), UInt32(512)) ?? 512)
    if let first = caIDs(dev, paddr(kAudioDevicePropertyStreams, s))?.first {
        frames += Int(caGet(first, paddr(kAudioStreamPropertyLatency), UInt32(0)) ?? 0)
    }
    return frames
}

// MARK: - Channel lists, numbers, paths

/// 1-based channel list: 1 | 1,2 | 3-4 | L | R | LR, each within the device's `count` channels.
func parseChannels(_ spec: String, count: Int, _ dev: Dev, _ s: Scope) -> [Int] {
    let t = spec.trimmingCharacters(in: .whitespaces).uppercased()
    var out: [Int] = []
    let bad = "--channels takes 1-based channel numbers like 1, 1,2, 3-4, L, R or LR, not '\(spec)'"
    switch t {
    case "L": out = [1]
    case "R": out = [2]
    case "LR": out = [1, 2]
    default:
        for part in t.split(separator: ",", omittingEmptySubsequences: false) {
            let p = part.trimmingCharacters(in: .whitespaces)
            if let dash = p.firstIndex(of: "-"), dash != p.startIndex {
                guard let lo = Int(p[..<dash]), let hi = Int(p[p.index(after: dash)...]), lo >= 1, hi >= lo, hi <= 1024 else {
                    usageError(bad)
                }
                out += Array(lo...hi)
            } else {
                guard let n = Int(p), n >= 1, n <= 1024 else { usageError(bad) }
                out.append(n)
            }
        }
    }
    if Set(out).count != out.count { usageError("--channels lists a channel twice: '\(spec)'") }
    if let over = out.first(where: { $0 > count }) {
        fail(ExitCode.refused, "'\(dev.name)' has \(count) \(s.word)\(count == 1 ? "" : "s"); there is no \(s.word) channel \(over)",
             device: dev)
    }
    return out
}

func number(_ a: Args, _ key: String, _ range: ClosedRange<Double>, unit: String, suffixes: [String] = []) -> Double? {
    guard var s = a.values[key]?.trimmingCharacters(in: .whitespaces) else { return nil }
    for suf in suffixes where s.lowercased().hasSuffix(suf) { s = String(s.dropLast(suf.count)); break }
    guard let v = Double(s), v.isFinite, range.contains(v) else {
        usageError("--\(key) takes \(unit) (\(JSON.number(range.lowerBound))..\(JSON.number(range.upperBound))), not '\(a.values[key]!)'")
    }
    return v
}

func positive(_ a: Args, _ key: String, max: Double = 86400) -> Double? {
    guard let s = a.values[key] else { return nil }
    guard let v = Double(s), v.isFinite, v > 0, v <= max else { usageError("--\(key) takes seconds (0 < s <= \(Int(max))), not '\(s)'") }
    return v
}

/// Absolute and lexically normalized (. and ..), never rewritten by what exists: Foundation's
/// standardizing drops /private only when the file is already there, so paths would differ run to run.
func absolutePath(_ p: String) -> String {
    let expanded = (p as NSString).expandingTildeInPath
    let full = expanded.hasPrefix("/") ? expanded : FileManager.default.currentDirectoryPath + "/" + expanded
    var parts: [Substring] = []
    for c in full.split(separator: "/") {
        if c == "." { continue }
        if c == ".." { if !parts.isEmpty { parts.removeLast() }; continue }
        parts.append(c)
    }
    return "/" + parts.joined(separator: "/")
}

/// Non-interleaved float32 at a rate; more than 2 channels needs a layout (discrete, in order).
func floatFormat(_ rate: Double, _ channels: Int, layout: AVAudioChannelLayout? = nil) -> AVAudioFormat? {
    if channels <= 2 { return AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: AVAudioChannelCount(channels), interleaved: false) }
    guard let l = layout ?? AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_DiscreteInOrder | UInt32(channels)) else { return nil }
    return AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, interleaved: false, channelLayout: l)
}

/// AVAudioFile errors carry a Core Audio OSStatus such as 'typ?'; say what it means.
func audioFileError(_ e: Error) -> String {
    let code = OSStatus(truncatingIfNeeded: (e as NSError).code)
    let meaning: [OSStatus: String] = [
        OSStatus(bitPattern: fourCC("typ?")): "not a file type Core Audio reads",
        OSStatus(bitPattern: fourCC("fmt?")): "an audio data format Core Audio can't decode",
        OSStatus(bitPattern: fourCC("dta?")): "the file is damaged or not what its extension says",
        OSStatus(bitPattern: fourCC("prm?")): "permission denied",
        OSStatus(bitPattern: fourCC("wht?")): "unspecified audio file error",
    ]
    if let m = meaning[code] { return "\(m), OSStatus \(statusText(code))" }
    return "\((e as NSError).localizedDescription), OSStatus \(statusText(code))"
}

// MARK: - play

/// The IO side of play: copy K interleaved channels from the ring into their device slots.
final class PlayIO {
    let ring: Ring
    let k: Int
    let slotBuffer: UnsafeMutablePointer<Int>
    let slotOffset: UnsafeMutablePointer<Int>
    let slotStride: UnsafeMutablePointer<Int>
    let dst: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>
    let scratch: UnsafeMutablePointer<Float>
    let scratchFrames = 4096
    let fadeFrames: Int
    var fadeLeft = -1                       // IO thread only
    let callbacks = Atomic<Int>(0)
    let played = Atomic<Int>(0)
    let underrun = Atomic<Int>(0)
    let sourceDone = Atomic<Bool>(false)
    let fadeRequested = Atomic<Bool>(false)
    let silenced = Atomic<Bool>(false)

    init(ring: Ring, slots: [ChannelSlot], fadeFrames: Int) {
        self.ring = ring
        k = slots.count
        self.fadeFrames = max(1, fadeFrames)
        slotBuffer = .allocate(capacity: k); slotOffset = .allocate(capacity: k); slotStride = .allocate(capacity: k)
        for (i, s) in slots.enumerated() { slotBuffer[i] = s.buffer; slotOffset[i] = s.offset; slotStride[i] = s.stride }
        dst = .allocate(capacity: k)
        dst.initialize(repeating: nil, count: k)
        scratch = .allocate(capacity: scratchFrames * k)
    }
    deinit { slotBuffer.deallocate(); slotOffset.deallocate(); slotStride.deallocate(); dst.deallocate(); scratch.deallocate() }

    func render(_ out: UnsafeMutablePointer<AudioBufferList>) {
        let abl = UnsafeMutableAudioBufferListPointer(out)
        var frames = 0
        for i in 0..<abl.count {
            guard let p = abl[i].mData else { continue }
            memset(p, 0, Int(abl[i].mDataByteSize))
            if frames == 0 && abl[i].mNumberChannels > 0 { frames = Int(abl[i].mDataByteSize) / (Int(abl[i].mNumberChannels) * 4) }
        }
        callbacks.wrappingAdd(1, ordering: .relaxed)
        if frames == 0 || silenced.load(ordering: .relaxed) { return }
        for c in 0..<k { dst[c] = slotBuffer[c] < abl.count ? abl[slotBuffer[c]].mData?.assumingMemoryBound(to: Float.self) : nil }
        let fading = fadeRequested.load(ordering: .acquiring)
        if fading && fadeLeft < 0 { fadeLeft = fadeFrames }
        var done = 0
        var dry = false
        while done < frames {
            let want = min(frames - done, scratchFrames)
            let got = ring.read(into: scratch, frames: want)
            for f in 0..<got {
                var g: Float = 1
                if fading { if fadeLeft > 0 { g = Float(fadeLeft) / Float(fadeFrames); fadeLeft -= 1 } else { g = 0 } }
                let row = scratch + f * k
                let at = done + f
                for c in 0..<k { if let d = dst[c] { d[at * slotStride[c] + slotOffset[c]] = row[c] * g } }
            }
            done += got
            if fading && fadeLeft == 0 { break }
            if got < want {
                dry = true
                if !sourceDone.load(ordering: .acquiring) { underrun.wrappingAdd(want - got, ordering: .relaxed) }
                break
            }
        }
        if fading && (fadeLeft == 0 || dry) { silenced.store(true, ordering: .releasing) }
        played.wrappingAdd(done, ordering: .releasing)
    }
}

/// File -> (resampled to the device rate) chunks of deinterleaved float, F file channels.
final class FileFeeder {
    let file: AVAudioFile
    let converter: AVAudioConverter?
    let readBuf: AVAudioPCMBuffer
    let outBuf: AVAudioPCMBuffer
    var fileDone = false
    var convDone = false
    var error: String? = nil

    init?(_ file: AVAudioFile, deviceRate: Double, chunk: Int) {
        self.file = file
        let inFmt = file.processingFormat
        guard let rb = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(chunk)) else { return nil }
        readBuf = rb
        if inFmt.sampleRate == deviceRate {
            converter = nil
            outBuf = rb
        } else {
            guard let outFmt = floatFormat(deviceRate, Int(inFmt.channelCount), layout: inFmt.channelLayout),
                  let conv = AVAudioConverter(from: inFmt, to: outFmt),
                  let ob = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: AVAudioFrameCount(chunk)) else { return nil }
            conv.sampleRateConverterQuality = AVAudioQuality.high.rawValue
            converter = conv
            outBuf = ob
        }
    }

    private func fill(_ buf: AVAudioPCMBuffer, _ n: AVAudioFrameCount) -> Bool {
        if fileDone || file.framePosition >= file.length { fileDone = true; return false }
        do { try file.read(into: buf, frameCount: min(n, buf.frameCapacity)) } catch {
            self.error = audioFileError(error)
            fileDone = true
            return false
        }
        if buf.frameLength == 0 { fileDone = true; return false }
        return true
    }

    /// Next chunk in outBuf; returns its frame count, 0 when the file is finished.
    func next() -> Int {
        guard let conv = converter else { return fill(outBuf, outBuf.frameCapacity) ? Int(outBuf.frameLength) : 0 }
        if convDone { return 0 }
        outBuf.frameLength = 0
        var err: NSError?
        let status = conv.convert(to: outBuf, error: &err) { [unowned self] want, inStatus in
            if self.fill(self.readBuf, want) { inStatus.pointee = .haveData; return self.readBuf }
            inStatus.pointee = .endOfStream
            return nil
        }
        if status == .error { error = err?.localizedDescription ?? "conversion failed"; convDone = true; return 0 }
        if status == .endOfStream { convDone = true }
        return Int(outBuf.frameLength)
    }
}

func cmdPlay(_ a: Args) {
    guard let fileArg = a.pos.first else { usageError("play needs a file") }
    if a.pos.count > 1 { usageError("unexpected argument '\(a.pos[1])'") }
    let gainDB = number(a, "gain", -120...24, unit: "dB", suffixes: ["dbfs", "db"]) ?? 0
    let t0 = Date()
    let path = absolutePath(fileArg)
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else { fail(ExitCode.failed, "no such file: \(path)") }
    if isDir.boolValue { fail(ExitCode.failed, "\(path) is a directory") }
    let snap = takeSnapshot()
    let dev: Dev
    if let t = a.values["to"] {
        dev = resolve(t, snap, need: .output)
    } else {
        guard let d = defaultOf(.output, snap) else { fail(ExitCode.failed, "there is no default output device; name one with --to") }
        dev = d
    }
    if dev.outputs == 0 { fail(ExitCode.refused, "'\(dev.name)' has no outputs", device: dev) }
    requireAlive(dev)
    let file: AVAudioFile
    do { file = try AVAudioFile(forReading: URL(fileURLWithPath: path)) } catch {
        fail(ExitCode.failed, "can't read \(path) as audio: \(audioFileError(error))")
    }
    let fileCh = Int(file.processingFormat.channelCount)
    let fileRate = file.processingFormat.sampleRate
    if file.length <= 0 || fileCh == 0 { fail(ExitCode.failed, "\(path) has no audio frames") }
    let chans = a.values["channels"].map { parseChannels($0, count: dev.outputs, dev, .output) }
        ?? (fileCh <= 2 ? Array(1...min(2, dev.outputs)) : Array(1...min(fileCh, dev.outputs)))
    let k = chans.count
    if fileCh > 1 && k > 1 && fileCh != k {
        warn("warning: \(fileCh)-channel file on \(k) channels: " + (fileCh > k ? "file channels \(k + 1)-\(fileCh) are dropped"
            : "channels \(chans[fileCh...].map(String.init).joined(separator: ",")) get silence"))
    }
    if let why = unsupportedFormat(dev.id, .output) { fail(ExitCode.failed, "can't play on '\(dev.name)': \(why)", device: dev) }
    let allSlots = channelSlots(dev.id, .output)
    let devRate = caGet(dev.id, paddr(kAudioDevicePropertyNominalSampleRate), Float64(0)) ?? dev.rate
    guard devRate > 0, allSlots.count >= dev.outputs else { fail(ExitCode.failed, "can't read the output layout of '\(dev.name)'", device: dev) }
    let chunk = 4096
    guard let feeder = FileFeeder(file, deviceRate: devRate, chunk: chunk) else {
        fail(ExitCode.failed, "can't convert \(path) from \(kHz(fileRate)) to \(kHz(devRate))")
    }
    let gain = Float(pow(10, gainDB / 20))
    let ring = Ring(frames: Int(devRate * 2), channels: k)
    let io = PlayIO(ring: ring, slots: chans.map { allSlots[$0 - 1] }, fadeFrames: Int(devRate * 0.01))
    let mixBuf = UnsafeMutablePointer<Float>.allocate(capacity: chunk * k)
    defer { mixBuf.deallocate() }
    var pending = 0, pendingAt = 0, clipped = 0, sourceFinished = false

    // File channels -> listed device channels: mono goes to every listed channel, one listed
    // channel gets the mix of every file channel, otherwise file channel i -> listed channel i.
    func produce() -> Bool {
        if pending == 0 {
            if sourceFinished { return false }
            let n = feeder.next()
            if n == 0 { sourceFinished = true; io.sourceDone.store(true, ordering: .releasing); return false }
            let src = feeder.outBuf.floatChannelData!
            let fc = Int(feeder.outBuf.format.channelCount)
            for f in 0..<n {
                for c in 0..<k {
                    var x: Float
                    if fc == 1 { x = src[0][f] }
                    else if k == 1 { var s: Float = 0; for i in 0..<fc { s += src[i][f] }; x = s / Float(fc) }
                    else { x = c < fc ? src[c][f] : 0 }
                    x *= gain
                    if x > 1 { x = 1; clipped += 1 } else if x < -1 { x = -1; clipped += 1 }
                    mixBuf[f * k + c] = x
                }
            }
            pending = n
            pendingAt = 0
        }
        let w = ring.write(mixBuf + pendingAt * k, frames: pending)
        pending -= w
        pendingAt += w
        return w > 0
    }

    let fileSeconds = Double(file.length) / fileRate
    vlog("file: \(fileCh) ch, \(fileRate) Hz, \(file.length) frames (\(fileSeconds) s); device rate \(devRate)"
         + (fileRate == devRate ? "" : " (AVAudioConverter)") + "; ring \(ring.capacity) frames; gain \(gainDB) dB")
    let (vol, muted) = volumeAndMute(dev, .output)
    if muted == true { warn("warning: '\(dev.name)' is muted") } else if vol == 0 { warn("warning: '\(dev.name)' volume is 0") }
    while produce() {}   // prefill: short files are entirely in the ring before the device starts
    trapSignals()
    let session = IOSession(dev.id)
    narrate("Playing \((path as NSString).lastPathComponent) (\(secs(fileSeconds)) s) on \(dev.name), channel"
            + (k == 1 ? " " : "s ") + chans.map(String.init).joined(separator: ",") + "...")
    if let why = session.start(.output, { _, _, _, out, _ in io.render(out) }) {
        fail(ExitCode.failed, "can't start '\(dev.name)': \(why)", device: dev)
    }
    var stopped = "end"
    var lastCalls = 0
    var lastProgress = Date()
    while true {
        let sig = caughtSignal()
        if sig != 0 {
            stopped = "signal"
            io.fadeRequested.store(true, ordering: .releasing)
            _ = poll(timeout: 0.25, every: 0.005) { io.silenced.load(ordering: .acquiring) }
            break
        }
        let progressed = produce()
        if sourceFinished && pending == 0 && ring.readable == 0 { break }
        let calls = io.callbacks.load(ordering: .relaxed)
        if calls != lastCalls { lastCalls = calls; lastProgress = Date() }
        else if Date().timeIntervalSince(lastProgress) > (calls == 0 ? 5 : 2) {
            session.stop()
            fail(ExitCode.failed, calls == 0 ? "'\(dev.name)' never started asking for audio"
                 : "'\(dev.name)' stopped asking for audio (unplugged?)", device: dev)
        }
        if !progressed { usleep(5_000) }
    }
    if stopped == "end" {
        // Let the last frames leave the device before stopping the IOProc.
        let tail = Double(outputTail(dev.id)) / devRate + 0.03
        vlog("drained; waiting \(Int(tail * 1000)) ms of output latency")
        usleep(useconds_t(tail * 1_000_000))
    }
    session.stop()
    let seconds = Double(io.played.load(ordering: .acquiring)) / devRate
    let under = io.underrun.load(ordering: .relaxed)
    vlog("\(io.callbacks.load(ordering: .relaxed)) IO cycles, \(io.played.load(ordering: .relaxed)) frames, \(under) underrun frames")
    if let e = feeder.error { warn("warning: reading \(path) stopped early: \(e)") }
    if under > 0 { warn("warning: \(under) frames of silence were inserted (the file couldn't be read fast enough)") }
    if clipped > 0 { warn("warning: \(clipped) samples were clipped at full scale (--gain \(JSON.number(gainDB)))") }
    if jsonMode {
        emit(.obj([("success", .bool(true)), ("played", .str(path)), ("to", dev.stub), ("channels", .arr(chans.map { .int($0) })),
                   ("seconds", secondsJSON(seconds)), ("ms", .int(ms(since: t0))), ("stopped", .str(stopped))]))
    } else {
        say("Played \(path) on \(dev.name) channel" + (k == 1 ? " " : "s ") + chans.map(String.init).joined(separator: ",")
            + ": \(String(format: "%.2f", seconds)) s" + (stopped == "signal" ? " (stopped by signal)" : ""))
    }
    if stopped == "signal" { exit(ExitCode.signal(caughtSignal())) }
}

// MARK: - Capture (record, level)

/// The IO side of record/level: gather the requested input channels into the ring.
final class CaptureIO {
    let ring: Ring
    let k: Int
    let slotBuffer: UnsafeMutablePointer<Int>
    let slotOffset: UnsafeMutablePointer<Int>
    let slotStride: UnsafeMutablePointer<Int>
    let src: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>
    let scratch: UnsafeMutablePointer<Float>
    let scratchFrames = 4096
    var nextSampleTime = -1.0               // IO thread only
    let callbacks = Atomic<Int>(0)
    let captured = Atomic<Int>(0)
    let dropped = Atomic<Int>(0)            // ring full: we fell behind
    let gaps = Atomic<Int>(0)               // the device skipped sample time (overload)

    init(ring: Ring, slots: [ChannelSlot]) {
        self.ring = ring
        k = slots.count
        slotBuffer = .allocate(capacity: k); slotOffset = .allocate(capacity: k); slotStride = .allocate(capacity: k)
        for (i, s) in slots.enumerated() { slotBuffer[i] = s.buffer; slotOffset[i] = s.offset; slotStride[i] = s.stride }
        src = .allocate(capacity: k)
        src.initialize(repeating: nil, count: k)
        scratch = .allocate(capacity: scratchFrames * k)
    }
    deinit { slotBuffer.deallocate(); slotOffset.deallocate(); slotStride.deallocate(); src.deallocate(); scratch.deallocate() }

    func capture(_ input: UnsafePointer<AudioBufferList>, _ time: UnsafePointer<AudioTimeStamp>,
                 _ output: UnsafeMutablePointer<AudioBufferList>) {
        let outs = UnsafeMutableAudioBufferListPointer(output)
        for i in 0..<outs.count { if let p = outs[i].mData { memset(p, 0, Int(outs[i].mDataByteSize)) } }
        callbacks.wrappingAdd(1, ordering: .relaxed)
        let ins = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        var frames = 0
        for c in 0..<k {
            let i = slotBuffer[c]
            if i < ins.count, let p = ins[i].mData, ins[i].mNumberChannels > 0 {
                src[c] = p.assumingMemoryBound(to: Float.self)
                if frames == 0 { frames = Int(ins[i].mDataByteSize) / (Int(ins[i].mNumberChannels) * 4) }
            } else {
                src[c] = nil
            }
        }
        if frames == 0 { return }
        let t = time.pointee
        if t.mFlags.contains(.sampleTimeValid) {
            if nextSampleTime >= 0 {
                let d = t.mSampleTime - nextSampleTime
                if d >= 1 { gaps.wrappingAdd(Int(d), ordering: .relaxed) }
            }
            nextSampleTime = t.mSampleTime + Double(frames)
        }
        var done = 0
        while done < frames {
            let n = min(frames - done, scratchFrames)
            for f in 0..<n {
                let at = done + f
                let row = scratch + f * k
                for c in 0..<k {
                    if let s = src[c] { row[c] = s[at * slotStride[c] + slotOffset[c]] } else { row[c] = 0 }
                }
            }
            let w = ring.write(scratch, frames: n)
            if w < n { dropped.wrappingAdd(n - w, ordering: .relaxed) }
            done += n
        }
        captured.wrappingAdd(frames, ordering: .releasing)
    }
}

func micPermission() -> String {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: return "allowed"
    case .denied: return "denied"
    case .restricted: return "restricted"
    case .notDetermined: return "not determined"
    @unknown default: return "unknown"
    }
}

let micHint = "The app that launched \(TOOL_NAME) (your terminal, tmux, or a parent .app such as TopicGet.app) needs "
    + "Microphone access: System Settings > Privacy & Security > Microphone."

/// TCC: denied/restricted input reads as silence, so refuse instead of recording zeros.
func requireMicrophone() {
    let state = micPermission()
    vlog("Microphone permission: \(state)")
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: return
    case .notDetermined:
        narrate("Asking macOS for Microphone access (the prompt names the app that launched \(TOOL_NAME))...")
        let sem = DispatchSemaphore(value: 0)
        let granted = Atomic<Bool>(false)
        AVCaptureDevice.requestAccess(for: .audio) { ok in granted.store(ok, ordering: .releasing); sem.signal() }
        var answered = false
        for _ in 0..<600 {   // 60 s, in slices, so Ctrl-C ends the wait through fail()'s cleanups
            if sem.wait(timeout: .now() + 0.1) == .success { answered = true; break }
            let sig = caughtSignal()
            if sig != 0 { fail(ExitCode.signal(sig), "interrupted while waiting for the Microphone permission answer") }
        }
        if !answered { fail(ExitCode.unconfirmed, "no answer to the Microphone permission prompt in 60 s. \(micHint)") }
        if !granted.load(ordering: .acquiring) { fail(ExitCode.failed, "Microphone access was not granted. \(micHint)") }
    default:
        fail(ExitCode.failed, "Microphone permission reads '\(state)', so input would be silence. \(micHint)")
    }
}

/// A capture of channels `chans` on `dev`, running until stop().
struct Capture {
    let dev: Dev
    let chans: [Int]
    let rate: Double
    let ring: Ring
    let io: CaptureIO
    let session: IOSession
}

func startCapture(_ dev: Dev, _ chans: [Int]) -> Capture {
    requireAlive(dev)
    if let why = unsupportedFormat(dev.id, .input) { fail(ExitCode.failed, "can't capture from '\(dev.name)': \(why)", device: dev) }
    let allSlots = channelSlots(dev.id, .input)
    let rate = caGet(dev.id, paddr(kAudioDevicePropertyNominalSampleRate), Float64(0)) ?? dev.rate
    guard rate > 0, allSlots.count >= dev.inputs else { fail(ExitCode.failed, "can't read the input layout of '\(dev.name)'", device: dev) }
    trapSignals()
    requireMicrophone()
    let ring = Ring(frames: Int(rate * 10), channels: chans.count)
    let io = CaptureIO(ring: ring, slots: chans.map { allSlots[$0 - 1] })
    let session = IOSession(dev.id)
    if let why = session.start(.input, { _, input, inputTime, out, _ in io.capture(input, inputTime, out) }) {
        fail(ExitCode.failed, "can't start '\(dev.name)': \(why)", device: dev)
    }
    vlog("capturing \(dev.name) channels \(chans) at \(rate) Hz; ring \(ring.capacity) frames")
    return Capture(dev: dev, chans: chans, rate: rate, ring: ring, io: io, session: session)
}

/// Peak, sum of squares and clipping per channel over a span.
struct Meter {
    var peak: [Float]
    var sumSq: [Double]
    var clipped: [Bool]
    var frames = 0
    var nonZero = false
    init(_ k: Int) { peak = Array(repeating: 0, count: k); sumSq = Array(repeating: 0, count: k); clipped = Array(repeating: false, count: k) }

    mutating func add(_ x: UnsafePointer<Float>, frames n: Int) {
        let k = peak.count
        for c in 0..<k {
            var p = peak[c]
            var s = 0.0
            for f in 0..<n {
                let v = abs(x[f * k + c])
                if v > p { p = v }
                s += Double(v) * Double(v)
            }
            if p > 0 { nonZero = true }
            if p >= 0.999 { clipped[c] = true }   // -0.01 dBFS
            peak[c] = p
            sumSq[c] += s
        }
        frames += n
    }
    func peakDB(_ c: Int) -> Double { dbfs(Double(peak[c])) }
    func rmsDB(_ c: Int) -> Double { frames == 0 ? -.infinity : dbfs((sumSq[c] / Double(frames)).squareRoot()) }
    var peakDB: Double { dbfs(Double(peak.max() ?? 0)) }
    var rmsDB: Double { frames == 0 ? -.infinity : dbfs((sumSq.reduce(0, +) / Double(frames * peak.count)).squareRoot()) }
    var anyClipped: Bool { clipped.contains(true) }
}

func silenceWarning(_ m: Meter, _ dev: Dev) {
    if m.frames > 0 && !m.nonZero {
        warn("warning: every sample from '\(dev.name)' was exactly 0 (digital silence). Normal for an idle loopback or digital "
             + "input; otherwise macOS may be withholding the microphone (permission: \(micPermission())). \(micHint)")
    }
}

func watchdog(_ cap: Capture, _ lastCalls: inout Int, _ lastProgress: inout Date) -> String? {
    let calls = cap.io.callbacks.load(ordering: .relaxed)
    if calls != lastCalls { lastCalls = calls; lastProgress = Date(); return nil }
    if Date().timeIntervalSince(lastProgress) > (calls == 0 ? 5 : 2) {
        let alive = (caGet(cap.dev.id, paddr(kAudioDevicePropertyDeviceIsAlive), UInt32(0)) ?? 0) != 0
        return calls == 0 ? "'\(cap.dev.name)' never started delivering audio"
            : "'\(cap.dev.name)' stopped delivering audio" + (alive ? "" : " (it is gone: unplugged?)")
    }
    return nil
}

func posixError(_ e: Int32) -> NSError {
    NSError(domain: NSPOSIXErrorDomain, code: Int(e), userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(e))])
}

/// Writes RIFF/WAVE (PCM 24-bit or IEEE float 32-bit, EXTENSIBLE above 2 channels) and rewrites
/// the header sizes as it goes, so the file is valid at every moment.
///
/// Resources start empty (fd -1, no buffer) and deinit releases only what is held. When an init
/// throws once every stored property is set, Swift runs deinit on the half-built object; freeing a
/// buffer the init had already freed was the 2026-11-01 SIGABRT. No path can do that now.
final class WavWriter {
    let path: String
    let channels: Int
    let rate: Int
    let float: Bool
    let bytesPerSample: Int
    private var fd: Int32 = -1                          // -1 = not open (never opened, or closed)
    private var bytes: UnsafeMutablePointer<UInt8>? = nil
    private var cap = 0
    private(set) var frames = 0
    private var dataBytes = 0
    private let header: [UInt8]
    private let dataSizeAt: Int
    private let factAt: Int?

    init(path: String, channels: Int, rate: Int, float: Bool, overwrite: Bool) throws {
        self.path = path
        self.channels = channels
        self.rate = rate
        self.float = float
        bytesPerSample = float ? 4 : 3
        let built = WavWriter.header(channels: channels, rate: rate, bytesPerSample: bytesPerSample, float: float)
        header = built.0
        dataSizeAt = built.1
        factAt = built.2
        let f = open(path, O_WRONLY | O_CREAT | (overwrite ? O_TRUNC : O_EXCL), 0o644)
        if f < 0 { throw posixError(errno) }
        fd = f
        guard writeAll(header, count: header.count) else {
            let e = errno
            Darwin.close(fd)
            fd = -1
            unlink(path)
            throw posixError(e)
        }
    }

    deinit {
        if fd >= 0 { Darwin.close(fd) }
        bytes?.deallocate()
    }

    /// The header bytes, and where the data size and fact frame count go.
    private static func header(channels: Int, rate: Int, bytesPerSample: Int, float: Bool) -> ([UInt8], Int, Int?) {
        var h: [UInt8] = []
        func u16(_ v: Int) { h += [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF)] }
        func u32(_ v: Int) { h += [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)] }
        func tag(_ s: String) { h += Array(s.utf8) }
        let bits = bytesPerSample * 8
        let block = channels * bytesPerSample
        tag("RIFF"); u32(0); tag("WAVE")
        tag("fmt ")
        if channels > 2 {
            u32(40); u16(0xFFFE); u16(channels); u32(rate); u32(rate * block); u16(block); u16(bits)
            u16(22); u16(bits); u32(0)                                  // valid bits; no speaker positions
            u32(float ? 3 : 1); u16(0); u16(0x10); h += [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71]
        } else if float {
            u32(18); u16(3); u16(channels); u32(rate); u32(rate * block); u16(block); u16(bits); u16(0)
        } else {
            u32(16); u16(1); u16(channels); u32(rate); u32(rate * block); u16(block); u16(bits)
        }
        var fact: Int? = nil
        if float { tag("fact"); u32(4); fact = h.count; u32(0) }
        tag("data")
        let dataSize = h.count
        u32(0)
        return (h, dataSize, fact)
    }

    private func writeAll(_ p: UnsafeRawPointer, count: Int) -> Bool {
        var off = 0
        while off < count {
            let n = Darwin.write(fd, p + off, count - off)
            if n < 0 { if errno == EINTR { continue }; return false }
            off += n
        }
        return true
    }

    func append(_ x: UnsafePointer<Float>, frames n: Int) -> Bool {
        guard fd >= 0 else { return false }
        let count = n * channels
        let need = count * bytesPerSample
        if bytes == nil || need > cap {
            bytes?.deallocate()
            cap = max(need, 4096 * channels * bytesPerSample)
            bytes = .allocate(capacity: cap)
        }
        guard let b = bytes else { return false }
        if float {
            memcpy(b, x, need)   // little-endian host
        } else {
            for i in 0..<count {
                let v = Int32(max(-8_388_608, min(8_388_607, (Double(x[i]) * 8_388_608).rounded())))
                b[3 * i] = UInt8(truncatingIfNeeded: v)
                b[3 * i + 1] = UInt8(truncatingIfNeeded: v >> 8)
                b[3 * i + 2] = UInt8(truncatingIfNeeded: v >> 16)
            }
        }
        guard writeAll(b, count: need) else { return false }
        frames += n
        dataBytes += need
        return true
    }

    private func put32(_ v: Int, at off: Int) -> Bool {
        var le = UInt32(truncatingIfNeeded: v).littleEndian
        return pwrite(fd, &le, 4, off_t(off)) == 4
    }

    /// Header sizes for what has been written so far.
    func patch() -> Bool {
        guard fd >= 0 else { return false }
        let padByte = dataBytes % 2
        var ok = put32(header.count - 8 + dataBytes + padByte, at: 4) && put32(dataBytes, at: dataSizeAt)
        if let f = factAt { ok = ok && put32(frames, at: f) }
        return ok
    }

    /// Pad, final sizes, close. Safe to call twice (the second call reports false).
    func close() -> Bool {
        guard fd >= 0 else { return false }
        var ok = true
        if dataBytes % 2 == 1 { var z: UInt8 = 0; ok = writeAll(&z, count: 1) }
        ok = patch() && ok
        ok = Darwin.close(fd) == 0 && ok
        fd = -1
        return ok
    }

    static func bytesPerSecond(_ channels: Int, _ rate: Int, float: Bool) -> Int { channels * rate * (float ? 4 : 3) }
}

/// Device-rate interleaved frames -> another rate, interleaved, through AVAudioConverter.
final class Resampler {
    let conv: AVAudioConverter
    let k: Int
    let inBuf: AVAudioPCMBuffer
    let outBuf: AVAudioPCMBuffer
    let interleaved: UnsafeMutablePointer<Float>
    var haveInput = false

    /// The only failure exit comes before anything deinit frees is acquired (Swift skips deinit for
    /// a half-built instance); keep every fallible step above the allocation.
    init?(from inRate: Double, to outRate: Double, channels k: Int, chunk: Int) {
        self.k = k
        let outCap = Int((Double(chunk) * outRate / inRate).rounded(.up)) + 1024
        guard let fi = floatFormat(inRate, k), let fo = floatFormat(outRate, k), let c = AVAudioConverter(from: fi, to: fo),
              let ib = AVAudioPCMBuffer(pcmFormat: fi, frameCapacity: AVAudioFrameCount(chunk)),
              let ob = AVAudioPCMBuffer(pcmFormat: fo, frameCapacity: AVAudioFrameCount(outCap)) else { return nil }
        c.sampleRateConverterQuality = AVAudioQuality.high.rawValue
        conv = c
        inBuf = ib
        outBuf = ob
        interleaved = .allocate(capacity: outCap * k)
    }
    deinit { interleaved.deallocate() }

    func process(_ x: UnsafePointer<Float>, frames n: Int, _ emit: (UnsafePointer<Float>, Int) -> Void) {
        let ch = inBuf.floatChannelData!
        for f in 0..<n { for c in 0..<k { ch[c][f] = x[f * k + c] } }
        inBuf.frameLength = AVAudioFrameCount(n)
        haveInput = true
        drain(end: false, emit)
    }

    func finish(_ emit: (UnsafePointer<Float>, Int) -> Void) { drain(end: true, emit) }

    private func drain(end: Bool, _ emit: (UnsafePointer<Float>, Int) -> Void) {
        while true {
            outBuf.frameLength = 0
            var err: NSError?
            let status = conv.convert(to: outBuf, error: &err) { [unowned self] _, st in
                if self.haveInput { self.haveInput = false; st.pointee = .haveData; return self.inBuf }
                st.pointee = end ? .endOfStream : .noDataNow
                return nil
            }
            let n = Int(outBuf.frameLength)
            if n > 0 {
                let ch = outBuf.floatChannelData!
                for f in 0..<n { for c in 0..<k { interleaved[f * k + c] = ch[c][f] } }
                emit(interleaved, n)
            }
            if status != .haveData { return }
        }
    }
}

func defaultRecordPath() -> String {
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyyMMdd-HHmmss"
    let base = FileManager.default.currentDirectoryPath + "/record-" + f.string(from: Date())
    var p = base + ".wav"
    var n = 2
    while FileManager.default.fileExists(atPath: p) { p = base + "-\(n).wav"; n += 1 }
    return p
}

func cmdRecord(_ a: Args) {
    if let extra = a.pos.first { usageError("unexpected argument '\(extra)' (the device goes in --from)") }
    guard let from = a.values["from"] else { usageError("record needs --from <target>") }
    let fixed = positive(a, "seconds")
    let untilSilence = positive(a, "until-silence", max: 3600)
    if fixed != nil && untilSilence != nil { usageError("use --seconds or --until-silence, not both") }
    if fixed == nil && untilSilence == nil { usageError("record needs --seconds <s> or --until-silence <s>") }
    if untilSilence == nil && a.values["max"] != nil { usageError("--max only applies to --until-silence") }
    if untilSilence == nil && a.values["threshold"] != nil { usageError("--threshold only applies to --until-silence") }
    let maxSeconds = positive(a, "max") ?? 60
    let threshold = number(a, "threshold", -120...0, unit: "dBFS", suffixes: ["dbfs", "db"]) ?? -45
    let outRateArg = number(a, "rate", 8000...384_000, unit: "Hz", suffixes: ["hz"])
    if let r = outRateArg, r != r.rounded() { usageError("--rate takes a whole number of Hz, not '\(a.values["rate"]!)'") }
    let float = a.flags.contains("float")
    let path = a.values["out"].map(absolutePath) ?? defaultRecordPath()
    if !path.lowercased().hasSuffix(".wav") { usageError("--out must name a .wav file, not '\(a.values["out"]!)'") }
    let t0 = Date()
    let snap = takeSnapshot()
    let dev = resolve(from, snap, need: .input)
    if dev.inputs == 0 { fail(ExitCode.refused, "'\(dev.name)' has no inputs", device: dev) }
    let chans = a.values["channels"].map { parseChannels($0, count: dev.inputs, dev, .input) } ?? [1]
    let k = chans.count
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: path, isDirectory: &isDir) {
        if isDir.boolValue { fail(ExitCode.failed, "\(path) is a directory") }
        if !a.flags.contains("force") { fail(ExitCode.refused, "refusing to overwrite \(path); pass --force to replace it") }
    }
    let devRate = caGet(dev.id, paddr(kAudioDevicePropertyNominalSampleRate), Float64(0)) ?? dev.rate
    let outRate = outRateArg ?? devRate
    let span = fixed ?? maxSeconds
    if Double(WavWriter.bytesPerSecond(k, Int(outRate), float: float)) * span > 4_294_000_000 {
        usageError("\(secs(span)) s of \(k) channel(s) at \(Int(outRate)) Hz would pass the 4 GB WAV limit")
    }
    let resampler: Resampler?
    if outRate != devRate {
        guard let r = Resampler(from: devRate, to: outRate, channels: k, chunk: 4096) else {
            fail(ExitCode.failed, "can't resample \(kHz(devRate)) to \(kHz(outRate))")
        }
        resampler = r
    } else {
        resampler = nil
    }
    trapSignals()   // from here on, Ctrl-C ends through the result or fail()'s cleanups
    // A new file is created before the capture starts, so a bad --out never opens the device, and
    // fail() removes it if the capture can't start. An existing file under --force is truncated only
    // once the capture runs, so a recording that never starts can't cost the old one.
    let replacing = a.flags.contains("force") && FileManager.default.fileExists(atPath: path)
    if replacing && access(path, W_OK) != 0 {
        fail(ExitCode.failed, "can't create \(path): \(String(cString: strerror(errno)))", device: dev)
    }
    func makeWriter() -> WavWriter {
        do { return try WavWriter(path: path, channels: k, rate: Int(outRate), float: float, overwrite: replacing) } catch {
            fail(ExitCode.failed, "can't create \(path): \(error.localizedDescription)", device: dev)
        }
    }
    var early: WavWriter? = nil
    if !replacing {
        let w = makeWriter()
        failCleanups.append { _ = w.close(); unlink(path) }
        early = w
    }
    narrate("Recording \(dev.name) channel" + (k == 1 ? " " : "s ") + chans.map(String.init).joined(separator: ",") + " -> \(path) "
            + (fixed.map { "for \(secs($0)) s" } ?? "until \(secs(untilSilence!)) s below \(JSON.number(threshold)) dBFS (max \(secs(maxSeconds)) s)")
            + "; Ctrl-C stops and keeps the file")
    let cap = startCapture(dev, chans)
    failCleanups.removeAll()   // capturing: from now on the file is kept and the result names it
    let writer: WavWriter
    if let w = early {
        writer = w
    } else {
        failCleanups.append { cap.session.stop() }
        writer = makeWriter()   // frames wait in the 10 s ring meanwhile
        failCleanups.removeAll()
    }
    let chunk = 4096
    let buf = UnsafeMutablePointer<Float>.allocate(capacity: chunk * k)
    defer { buf.deallocate() }
    var meter = Meter(k)
    var taken = 0
    var writeFailed = false
    var stopped: String? = nil
    var failure: String? = nil
    let target = fixed.map { Int(($0 * devRate).rounded()) }
    let maxFrames = Int((maxSeconds * devRate).rounded())
    let block = max(1, Int(devRate * 0.05))
    var blockAt = 0
    var blockSq = Array(repeating: 0.0, count: k)
    var heard = false
    var quietFrames = 0
    let quietNeeded = untilSilence.map { Int(($0 * devRate).rounded()) } ?? 0
    let thresholdSq = pow(10, threshold / 10)   // compare mean square against (10^(dB/20))^2
    var lastPatch = 0
    var lastCalls = 0
    var lastProgress = Date()

    func write(_ p: UnsafePointer<Float>, _ n: Int) {
        if writeFailed || n == 0 { return }
        if let r = resampler { r.process(p, frames: n) { q, m in if !writer.append(q, frames: m) { writeFailed = true } } }
        else if !writer.append(p, frames: n) { writeFailed = true }
    }

    /// Take n frames from buf; returns why to stop, if it's time.
    func consume(_ n: Int) -> String? {
        var take = n
        var why: String? = nil
        if let target, taken + take >= target { take = target - taken; why = "seconds" }
        if untilSilence != nil {
            let limit = min(take, maxFrames - taken)
            var stopAt: Int? = nil
            for f in 0..<limit {
                for c in 0..<k { let v = Double(buf[f * k + c]); blockSq[c] += v * v }
                blockAt += 1
                if blockAt == block {
                    let loud = blockSq.contains { $0 / Double(block) >= thresholdSq }
                    if loud { heard = true; quietFrames = 0 } else if heard { quietFrames += block }
                    blockAt = 0
                    for c in 0..<k { blockSq[c] = 0 }
                    if heard && quietFrames >= quietNeeded { stopAt = f + 1; break }
                }
            }
            if let s = stopAt { take = s; why = "silence" }
            else { take = limit; if taken + take >= maxFrames { why = "max" } }
        }
        meter.add(buf, frames: take)
        write(buf, take)
        taken += take
        return why
    }

    while stopped == nil {
        if caughtSignal() != 0 { stopped = "signal"; break }
        let n = cap.ring.read(into: buf, frames: chunk)
        if n == 0 {
            if let why = watchdog(cap, &lastCalls, &lastProgress) { failure = why; stopped = "error"; break }
            usleep(10_000)
            continue
        }
        lastProgress = Date()
        stopped = consume(n)
        if writeFailed { failure = "writing \(path) failed: \(String(cString: strerror(errno)))"; stopped = "error"; break }
        if taken - lastPatch >= Int(devRate / 2) { _ = writer.patch(); lastPatch = taken }
    }
    cap.session.stop()
    if stopped == "signal" {
        // Keep what was already captured before the stop.
        while true {
            let n = cap.ring.read(into: buf, frames: chunk)
            if n == 0 { break }
            if consume(n) != nil { break }
        }
    }
    resampler?.finish { q, m in if !writer.append(q, frames: m) { writeFailed = true } }
    let closed = writer.close()
    if writeFailed || !closed { failure = failure ?? "writing \(path) failed: \(String(cString: strerror(errno)))" }
    let dropped = cap.io.dropped.load(ordering: .relaxed) + cap.io.gaps.load(ordering: .relaxed)
    vlog("\(cap.io.callbacks.load(ordering: .relaxed)) IO cycles, \(cap.io.captured.load(ordering: .relaxed)) frames captured, "
         + "\(taken) kept, \(cap.io.dropped.load(ordering: .relaxed)) dropped, \(cap.io.gaps.load(ordering: .relaxed)) in device gaps")
    silenceWarning(meter, dev)
    if dropped > 0 { warn("warning: \(dropped) frames were lost while recording; the file has gaps") }
    if untilSilence != nil && stopped == "max" && !heard {
        failure = "timed out: nothing above \(JSON.number(threshold)) dBFS in \(secs(maxSeconds)) s (the file is kept)"
    }
    let seconds = Double(writer.frames) / outRate
    let ok = failure == nil
    if jsonMode {
        var o: [(String, JSON)] = [("success", .bool(ok))]
        if let failure { o.append(("error", .str(failure))) }
        o += [("file", .str(path)), ("from", dev.stub), ("channels", .arr(chans.map { .int($0) })), ("rate", .num(outRate)),
              ("seconds", secondsJSON(seconds)), ("peak_dbfs", dbJSON(meter.peakDB)), ("rms_dbfs", dbJSON(meter.rmsDB)),
              ("clipped", .bool(meter.anyClipped)), ("format", .str(float ? "f32" : "s24")), ("stopped", .str(stopped ?? "error")),
              ("dropped_frames", .int(dropped)), ("ms", .int(ms(since: t0)))]
        emit(.obj(o))
    } else {
        if let failure { warn("error: \(failure)") }
        say("Recorded \(String(format: "%.2f", seconds)) s to \(path) (\(k) ch, \(kHz(outRate)), \(float ? "32-bit float" : "24-bit")): "
            + "peak \(dbText(meter.peakDB)) dBFS, RMS \(dbText(meter.rmsDB)) dBFS" + (meter.anyClipped ? ", " + paint("clipped", "31") : "")
            + (stopped == "signal" ? " (stopped by signal)" : stopped == "silence" ? " (stopped on silence)" : stopped == "max" ? " (hit --max)" : ""))
    }
    if let failure {
        exit(failure.hasPrefix("timed out") ? ExitCode.unconfirmed : ExitCode.failed)
    }
    if stopped == "signal" { exit(ExitCode.signal(caughtSignal())) }
}

// MARK: - level

func levelChannels(_ m: Meter, _ chans: [Int]) -> JSON {
    .arr(chans.enumerated().map { i, ch in
        .obj([("channel", .int(ch)), ("peak_dbfs", dbJSON(m.peakDB(i))), ("rms_dbfs", dbJSON(m.rmsDB(i))), ("clipped", .bool(m.clipped[i]))])
    })
}

let clockStamp: DateFormatter = {
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "HH:mm:ss.SSS"
    return f
}()

func cmdLevel(_ a: Args) {
    if let extra = a.pos.first { usageError("unexpected argument '\(extra)' (the device goes in --from)") }
    guard let from = a.values["from"] else { usageError("level needs --from <target>") }
    let watchMode = a.flags.contains("watch")
    let span = positive(a, "seconds", max: watchMode ? 86400 : 3600)
    let snap = takeSnapshot()
    let dev = resolve(from, snap, need: .input)
    if dev.inputs == 0 { fail(ExitCode.refused, "'\(dev.name)' has no inputs", device: dev) }
    let chans = a.values["channels"].map { parseChannels($0, count: dev.inputs, dev, .input) } ?? Array(1...dev.inputs)
    let k = chans.count
    if watchMode {
        narrate("Levels of \(dev.name) every 100 ms, peak/RMS dBFS per channel; Ctrl-C to stop")
    } else {
        narrate("Measuring \(dev.name) for \(secs(span ?? 2)) s...")
    }
    let cap = startCapture(dev, chans)
    let chunk = 4096
    let buf = UnsafeMutablePointer<Float>.allocate(capacity: chunk * k)
    defer { buf.deallocate() }
    let window = watchMode ? max(1, Int(cap.rate * 0.1)) : Int.max
    let total = span.map { Int(($0 * cap.rate).rounded()) } ?? (watchMode ? Int.max : Int((2 * cap.rate).rounded()))
    var meter = Meter(k)       // whole span
    var win = Meter(k)         // current 100 ms window (--watch)
    var taken = 0
    var stopped = "seconds"
    var lastCalls = 0
    var lastProgress = Date()
    func line(_ m: Meter) {
        if jsonMode {
            emit(.obj([("at", .str(rfc3339.string(from: Date()))), ("channels", levelChannels(m, chans))]))
        } else {
            say(clockStamp.string(from: Date()) + "  " + chans.enumerated().map { i, ch in
                "\(ch): " + pad(dbText(m.peakDB(i)), 6, right: true) + " /" + pad(dbText(m.rmsDB(i)), 6, right: true)
                    + (m.clipped[i] ? " " + paint("CLIP", "31") : "")
            }.joined(separator: "   "))
        }
    }
    outer: while taken < total {
        if caughtSignal() != 0 { stopped = "signal"; break }
        let n = cap.ring.read(into: buf, frames: min(chunk, total - taken))
        if n == 0 {
            if let why = watchdog(cap, &lastCalls, &lastProgress) { cap.session.stop(); fail(ExitCode.failed, why, device: dev) }
            usleep(10_000)
            continue
        }
        lastProgress = Date()
        var at = 0
        while at < n {
            let step = watchMode ? min(n - at, window - win.frames) : n - at
            win.add(buf + at * k, frames: step)
            meter.add(buf + at * k, frames: step)
            at += step
            if watchMode && win.frames >= window { line(win); win = Meter(k) }
        }
        taken += n
        if caughtSignal() != 0 { stopped = "signal"; break outer }
    }
    cap.session.stop()
    silenceWarning(meter, dev)
    if watchMode {
        if jsonMode == false && stopped == "signal" { warn("level: stopped") }
        exit(ExitCode.ok)
    }
    let seconds = Double(meter.frames) / cap.rate
    if jsonMode {
        emit(.obj([("from", dev.stub), ("rate", .num(cap.rate)), ("seconds", secondsJSON(seconds)),
                   ("channels", levelChannels(meter, chans)), ("stopped", .str(stopped))]))
    } else {
        say("\(dev.name), \(String(format: "%.2f", seconds)) s at \(kHz(cap.rate)) (dBFS):")
        // A name that is just the channel number (built-in devices) says nothing: show "-".
        func name(_ ch: Int) -> String {
            let n = dev.inputNames.count >= ch ? dev.inputNames[ch - 1] : ""
            return n.isEmpty || n == String(ch) ? "-" : n
        }
        let nameW = max(4, chans.map { name($0).count }.max() ?? 4)
        say("  " + pad("CH", 3, right: true) + "  " + pad("NAME", nameW) + "  " + pad("PEAK", 6, right: true) + "  "
            + pad("RMS", 6, right: true) + "  CLIPPED")
        for (i, ch) in chans.enumerated() {
            say("  " + pad(String(ch), 3, right: true) + "  " + pad(name(ch), nameW) + "  "
                + pad(dbText(meter.peakDB(i)), 6, right: true) + "  " + pad(dbText(meter.rmsDB(i)), 6, right: true) + "  "
                + (meter.clipped[i] ? paint("yes", "31") : "-"))
        }
    }
    if stopped == "signal" { exit(ExitCode.signal(caughtSignal())) }
}

// MARK: - watch

let rfc3339: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return f
}()

let localStamp: DateFormatter = {
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyy-MM-dd HH:mm:ss"
    return f
}()

func deviceKey(_ d: Dev) -> String { d.uid.isEmpty ? "id:\(d.id)" : d.uid }

/// HAL listeners on the device list and the three defaults, plus a 1 s poll as a backstop.
/// A line prints only when what reads back differs from the last line's state.
final class Watcher {
    let queue = DispatchQueue(label: "audio-tool.watch")
    var last: [String: Dev] = [:]
    var lastDefault: [Role: String] = [:]
    var timer: DispatchSourceTimer? = nil

    init(_ snap: Snapshot) {
        for d in snap.devices { last[deviceKey(d)] = d }
        for r in Role.allCases { lastDefault[r] = defaultOf(r, snap).map(deviceKey) ?? "" }
    }

    func start() {
        for sel in [kAudioHardwarePropertyDevices] + Role.allCases.map(\.selector) {
            var a = paddr(sel)
            let st = AudioObjectAddPropertyListenerBlock(systemObject, &a, queue) { [weak self] _, _ in self?.rescan() }
            vlog("listener '\(statusText(OSStatus(bitPattern: sel)).split(separator: "'").first ?? "")' -> \(statusText(st))")
        }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 1, repeating: 1)
        t.setEventHandler { [weak self] in self?.rescan() }
        t.resume()
        timer = t
    }

    func rescan() {
        guard let ids = caIDs(systemObject, paddr(kAudioHardwarePropertyDevices)) else { return }
        let d = readDefaults()
        let devs = ids.map { readDev($0, d) }.sorted { (fold($0.name), $0.uid) < (fold($1.name), $1.uid) }
        var now: [String: Dev] = [:]
        for x in devs { now[deviceKey(x)] = x }
        for (key, old) in last.sorted(by: { $0.key < $1.key }) where now[key] == nil { report("removed", nil, old) }
        for x in devs where last[deviceKey(x)] == nil { report("added", nil, x) }
        last = now
        for r in Role.allCases {
            let dev = devs.first { $0.id == r.id(d) }
            let key = dev.map(deviceKey) ?? ""
            if key != lastDefault[r] { lastDefault[r] = key; report("default", r, dev) }
        }
    }

    func report(_ event: String, _ role: Role?, _ d: Dev?) {
        let at = Date()
        if jsonMode {
            var o: [(String, JSON)] = [("event", .str(event))]
            if let role { o.append(("role", .str(role.rawValue))) }
            o += [("uid", d.map { .str($0.uid) } ?? .null), ("name", d.map { .str($0.name) } ?? .null),
                  ("at", .str(rfc3339.string(from: at)))]
            emit(.obj(o))
        } else {
            let word = paint(pad(event, 8), event == "removed" ? "33" : "32")
            let what = d.map { "\($0.name)  (\($0.uid))" } ?? "none"
            say("\(localStamp.string(from: at))  \(word)  " + (role.map { "\($0.rawValue) -> " } ?? "") + what)
        }
    }
}

func cmdWatch(_ a: Args) {
    if let extra = a.pos.first { usageError("unexpected argument '\(extra)'") }
    let snap = takeSnapshot()
    let w = Watcher(snap)
    keepAlive.append(w)
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        signal(sig, SIG_IGN)
        let s = DispatchSource.makeSignalSource(signal: sig, queue: .main)
        s.setEventHandler {
            if !jsonMode { warn("watch: stopped") }
            exit(ExitCode.ok)
        }
        s.resume()
        keepAlive.append(s)
    }
    narrate("Watching \(snap.devices.count) devices and the default input/output/system; Ctrl-C to stop")
    w.start()
    dispatchMain()
}

// MARK: - Alias

let reservedAliases: Set<String> = ["help", "list", "ls", "set", "rm", "remove"]

func cmdAlias(_ a: Args) {
    let sub = (a.pos.first ?? "list").lowercased()
    let rest = Array(a.pos.dropFirst())
    switch sub {
    case "list", "ls":
        helpTopic = "alias list"
        if let extra = rest.first { usageError("unexpected argument '\(extra)'") }
        aliasList()
    case "set":
        helpTopic = "alias set"
        guard rest.count == 2 else {
            usageError(rest.count < 2 ? "alias set needs <alias> <target>" : "unexpected argument '\(rest[2])'")
        }
        aliasSet(rest[0], rest[1])
    case "rm", "remove":
        helpTopic = "alias rm"
        guard rest.count == 1 else {
            usageError(rest.isEmpty ? "alias rm needs <alias>" : "unexpected argument '\(rest[1])'")
        }
        aliasRemove(rest[0])
    default:
        helpTopic = "alias"
        usageError("unknown alias action '\(sub)'; use list, set or rm")
    }
}

func strictAliases() -> [String: String] {
    switch readAliasFile() {
    case .ok(let a): return a
    case .broken(let why): fail(ExitCode.failed, "\(why); fix or remove it before changing aliases")
    }
}

func aliasList() {
    let all = aliases()
    let devs = all.isEmpty ? [] : takeSnapshot().devices
    let rows = all.sorted { $0.key < $1.key }.map { (alias: $0.key, target: $0.value, dev: lookupStored($0.value, devs)) }
    if jsonMode {
        emit(.obj([("aliases", .arr(rows.map {
            JSON.obj([("alias", .str($0.alias)), ("target", .str($0.target)), ("device", $0.dev?.json ?? .null)])
        }))]))
        return
    }
    if rows.isEmpty { say("No aliases. Add one: \(TOOL_NAME) alias set <alias> <target>"); return }
    let w = max(5, rows.map { $0.alias.count }.max() ?? 5)
    let tw = max(6, rows.map { $0.target.count }.max() ?? 6)
    say(pad("ALIAS", w) + "  " + pad("TARGET", tw) + "  DEVICE")
    for r in rows { say(pad(r.alias, w) + "  " + pad(r.target, tw) + "  " + (r.dev.map { "\($0.index) \($0.name)" } ?? "(not present)")) }
}

func aliasSet(_ rawName: String, _ target: String) {
    let name = rawName.lowercased()
    guard name.range(of: "^[a-z0-9][a-z0-9._-]*$", options: .regularExpression) != nil else {
        usageError("alias '\(rawName)' must be letters, digits, '.', '_' or '-' (starting with a letter or digit)")
    }
    if name.allSatisfy(\.isNumber) { usageError("alias '\(rawName)' is all digits; that would shadow an index") }
    if reservedAliases.contains(name) { usageError("'\(name)' is reserved") }
    var all = strictAliases()
    let snap = takeSnapshot()
    if let d = snap.devices.first(where: { $0.uid.caseInsensitiveCompare(name) == .orderedSame }) {
        usageError("alias '\(rawName)' is the uid of \(d.name); it would shadow it")
    }
    var dev = resolve(target, snap)
    let was = all[name]
    func report(_ changed: Bool, _ now: String) {
        dev.alias = name
        if jsonMode {
            emit(.obj([("success", .bool(true)), ("changed", .bool(changed)), ("alias", .str(name)),
                       ("was", was.map { .str($0) } ?? .null), ("now", .str(now)), ("device", dev.json)]))
        } else {
            say(changed ? "Alias '\(name)' -> \(dev.label)" : "Alias '\(name)' already -> \(dev.label); nothing written")
        }
    }
    if let was, was == dev.uid { report(false, was); return }
    all[name] = dev.uid
    do { try writeAliases(all) } catch { fail(ExitCode.failed, "can't write \(aliasPath): \(error.localizedDescription)", device: dev) }
    guard case .ok(let back) = readAliasFile(), let now = back[name], now == dev.uid else {
        fail(ExitCode.unconfirmed, "unconfirmed: wrote alias '\(name)' but \(aliasPath) doesn't read it back", device: dev)
    }
    report(true, now)
}

func aliasRemove(_ rawName: String) {
    let name = rawName.lowercased()
    var all = strictAliases()
    func report(_ changed: Bool, _ was: String?) {
        if jsonMode {
            emit(.obj([("success", .bool(true)), ("changed", .bool(changed)), ("alias", .str(name)),
                       ("was", was.map { .str($0) } ?? .null), ("now", .null)]))
        } else {
            say(changed ? "Removed alias '\(name)' (was \(was ?? ""))" : "No alias '\(name)'; nothing removed")
        }
    }
    guard let was = all[name] else { report(false, nil); return }
    all[name] = nil
    do { try writeAliases(all) } catch { fail(ExitCode.failed, "can't write \(aliasPath): \(error.localizedDescription)") }
    guard case .ok(let back) = readAliasFile(), back[name] == nil else {
        fail(ExitCode.unconfirmed, "unconfirmed: removed alias '\(name)' but \(aliasPath) still has it")
    }
    report(true, was)
}

// MARK: - Help

let targetHelp = """
Targets:
  4                          1-based index from 'list' (shifts when devices come and go)
  scarlett                   alias (alias set scarlett 4)
  BuiltInSpeakerDevice       device uid, exact: the stable key ('list' shows it)
  "speakers"                 name substring, case-insensitive; exact name wins. Several matches
                             narrow to devices with the inputs/outputs the command needs; still
                             several = exit 2 listing them. No match = exit 1 listing the devices.
"""

let stubHelp = """
Device stub: {"uid":"BuiltInSpeakerDevice","name":"MacBook Pro Speakers"}
"""

let commonOpts = """
  -v, --verbose              CoreAudio calls, status codes, formats and timings on stderr
  -h, --help                 Show this help
"""

let channelHelp = "1 | 1,2 | 3-4 | L | R | LR (L=1, R=2)"

func controlHelp(_ verb: String) -> String {
    let isVol = verb == "volume"
    let what = isVol ? """
        No value: print the device's output volume (0-100). With a value: set it and read it
        back; already so, nothing is sent (changed:false). Uses the control macOS's volume
        slider uses (virtual main volume), else the main or per-channel volume. A read-back
        within 1 of the value asked counts (hardware steps); 'now' is what reads back.
        """ : """
        No value: print the device's output mute (on|off). With on, off or toggle: set it and
        read it back; already so, nothing is sent (changed:false). Mute is the main element,
        else every channel's mute (muted = all of them read muted).
        """
    return """
    Usage:
      \(TOOL_NAME) \(verb) <target> [\(isVol ? "0-100" : "on|off|toggle")] [--input] [-t <seconds>] [-j] [-v]

    \(what)
    --input works on the input side instead. A device without the control (e.g. a USB
    interface with only a hardware knob) is exit 3 with the reason, never a fake success.

    Options:
      --input                    The input \(verb) instead of the output
      -t, --timeout <seconds>    Read-back limit (default 5); no read-back: exit 4
      -j, --json                 {"device":Stub,"scope":"output","\(isVol ? "volume\":69" : "mute\":false")}
                                 set: {"success":true,"device":Stub,"scope":"output","changed":true,
                                       "was":\(isVol ? "69" : "false"),"now":\(isVol ? "40" : "true"),"ms":3}
    \(commonOpts)

    \(stubHelp)

    \(targetHelp)
    """
}

let helpTexts: [String: String] = [
    "": """
    \(TOOL_NAME) \(VERSION) - native macOS audio CLI (CoreAudio). Alias: audio

    Usage:
      \(TOOL_NAME) <command> [options]

    Commands:
      list                       Every audio device: uid, transport, channels, rate, defaults
      status                     Default input, output and system output, with volume and mute
      default <role> [target]    Show the default input|output|system, or set it and read it back
      volume <target> [0-100]    Print or set a device's volume (--input for the input side)
      mute <target> [state]      Print or set mute: on|off|toggle (--input for the input side)
      play <file> [--to target]  Play a sound file on one device, blocking until it's done
      record --from <target>     Record input channels to a WAV (-s seconds or --until-silence)
      level --from <target>      Peak and RMS dBFS per input channel (--watch: every 100 ms)
      watch                      One line per device added/removed and default change, until Ctrl-C
      alias <list|set|rm>        Name devices (~/.config/macos-gui/audio_aliases.json)
      help [command]             Help for a command (also: <command> -h)

    Options (every command):
      -j, --json                 JSON on stdout (watch, level --watch: JSON Lines)
      -v, --verbose              CoreAudio calls, status codes, formats and timings on stderr
      -h, --help                 Show help
      -V, --version              Show version

    \(targetHelp)

    Exit codes:
      0 ok   1 failed   2 usage (incl. ambiguous target)   3 refused (no such control or
      channel; would overwrite)   4 timed out / not confirmed by read-back
      128+signal: play, record or level cut short (130 Ctrl-C, 143 SIGTERM); watch exits 0

    Examples:
      \(TOOL_NAME) list
      \(TOOL_NAME) list -j | jq -r '.devices[] | select(.outputs > 0) | .uid'
      \(TOOL_NAME) alias set scarlett "Scarlett 2i2"
      \(TOOL_NAME) play ding.wav --to scarlett --channels 1
      \(TOOL_NAME) record --from scarlett --channels 1 --until-silence 1.5 -o phrase.wav -j
      \(TOOL_NAME) level --from scarlett --watch
      \(TOOL_NAME) default output speakers
      \(TOOL_NAME) watch -j
    """,
    "list": """
    Usage:
      \(TOOL_NAME) list [-j] [-v]

    Every CoreAudio device, sorted by name then uid, with a 1-based index that is a valid
    target for every other command. The index shifts when devices come and go and the id
    (AudioObjectID) changes across replugs and reboots: script with the uid or an alias.
    Human output also lists channel names when they say more than the channel number.

    Options:
      -j, --json                 {"devices":[Device...]}
    \(commonOpts)

    Device JSON:
      {"index":4,"id":89,"uid":"AppleUSBAudioEngine:...","name":"Scarlett 2i2 4th Gen",
       "manufacturer":"Focusrite","transport":"usb","inputs":4,"outputs":2,
       "input_channels":["Input 1","Input 2","Loopback 1","Loopback 2"],
       "output_channels":["Output 1","Output 2"],"rate":48000,"default_input":false,
       "default_output":false,"default_system_output":false,"alive":true,"alias":"scarlett"}
      transport: usb builtin bluetooth continuity aggregate virtual hdmi displayport airplay
                 thunderbolt pci firewire avb unknown
      *_channels: kAudioObjectPropertyElementName per channel ("" if one is unnamed), [] if none
    """,
    "status": """
    Usage:
      \(TOOL_NAME) status [-j] [-v]

    The default input, output and system output (alerts) devices, each with its volume
    (0-100) and mute on that side, or none/null where the device has no such control.

    Options:
      -j, --json                 {"input":{"device":Device,"volume":50,"mute":false},
                                  "output":{...},"system":{...}}    (device null when none)
    \(commonOpts)
    """,
    "default": """
    Usage:
      \(TOOL_NAME) default <input|output|system> [target] [-t <seconds>] [-j] [-v]

    No target: print that default device. With one: make it the default and read it back.
    Already the default: nothing is sent (changed:false). system is the device for alerts
    and sound effects. A device without that side's channels, or that says it can't be the
    default, is exit 3.

    Options:
      -t, --timeout <seconds>    Read-back limit (default 5); no read-back: exit 4
      -j, --json                 {"role":"output","device":Device}
                                 set: {"success":true,"role":"output","changed":true,
                                       "was":Stub,"now":Stub,"ms":4}
    \(commonOpts)

    \(stubHelp)

    \(targetHelp)
    """,
    "volume": controlHelp("volume"),
    "mute": controlHelp("mute"),
    "play": """
    Usage:
      \(TOOL_NAME) play <file> [--to <target>] [-c <channels>] [--gain <dB>] [-j] [-v]

    Play wav, aiff, caf, m4a, mp3 (anything Core Audio reads) on one device and block until
    the sound has left it. The file is resampled to the device's rate (AVAudioConverter) and
    played through a raw IOProc on that device, mixed with other apps; the defaults, the
    device's rate and its volume are never touched. The device's inputs stay closed.

    Options:
      --to <target>              Output device (default: the default output)
      -c, --channels <list>      Output channels: \(channelHelp)
                                 (default 1,2 for mono/stereo files, else 1..N). A mono file
                                 goes to every listed channel; one listed channel gets a mix of
                                 the file's channels; otherwise file channel i -> listed channel i
      --gain <dB>                -120..24 (default 0); samples clip at full scale
      -j, --json                 {"success":true,"played":"/abs/ding.wav","to":Stub,"channels":[1],
                                  "seconds":0.5,"ms":561,"stopped":"end"}
    \(commonOpts)

    seconds is the audio played. Ctrl-C/SIGTERM fade out over 10 ms, stop, print the result
    with "stopped":"signal" and exit 130/143.

    \(stubHelp)

    \(targetHelp)
    """,
    "record": """
    Usage:
      \(TOOL_NAME) record --from <target> (-s <seconds> | --until-silence <seconds> [--max <seconds>])
                         [-c <channels>] [--threshold <dBFS>] [--rate <Hz>] [--float]
                         [-o <file.wav>] [--force] [-j] [-v]

    Record input channels of one device to a WAV: one channel = mono, N channels = N in
    the order listed. 24-bit PCM at the device's rate unless --float or --rate. The header
    is rewritten every 0.5 s, so even a killed process leaves a valid file. Ctrl-C/SIGTERM
    stop and keep what was recorded ("stopped":"signal", exit 130/143).

    Options:
      --from <target>            Input device (required)
      -c, --channels <list>      Input channels: \(channelHelp) (default 1)
      -s, --seconds <s>          Record exactly this long
      --until-silence <s>        Stop after this long below --threshold, once something louder
                                 was heard (50 ms blocks; silent = every channel's RMS below it)
      --max <s>                  Cap for --until-silence (default 60). Nothing heard: exit 4
      --threshold <dBFS>         Silence level for --until-silence, -120..0 (default -45)
      --rate <Hz>                Resample to this rate, 8000..384000 (AVAudioConverter)
      --float                    32-bit float instead of 24-bit PCM
      -o, --out <file.wav>       Output path (default ./record-YYYYMMDD-HHMMSS.wav)
      --force                    Overwrite --out if it exists (otherwise exit 3), once recording starts
      -j, --json                 {"success":true,"file":"/abs/rec.wav","from":Stub,"channels":[1],
                                  "rate":48000,"seconds":2,"peak_dbfs":-12.31,"rms_dbfs":-30.05,
                                  "clipped":false,"format":"s24","stopped":"seconds",
                                  "dropped_frames":0,"ms":2093}
    \(commonOpts)

    stopped: seconds | silence | max | signal | error. peak/rms are over the whole file and
    every channel (null = -inf, digital silence). clipped = a sample at -0.01 dBFS or above.
    dropped_frames > 0 means the file has gaps. A failure after the file exists prints the
    same object with "success":false and "error", and keeps the file.

    \(stubHelp)

    \(targetHelp)
    """,
    "level": """
    Usage:
      \(TOOL_NAME) level --from <target> [-c <channels>] [-s <seconds>] [--watch] [-j] [-v]

    Peak and RMS (dBFS) per input channel over a span (default 2 s). Nothing is written.
    Silence is -inf (null in JSON); clipped = a sample at -0.01 dBFS or above.

    Options:
      --from <target>            Input device (required)
      -c, --channels <list>      Input channels: \(channelHelp) (default: all)
      -s, --seconds <s>          Span (default 2); with --watch, stop after this long
      --watch                    One line per 100 ms until Ctrl-C (exit 0)
      -j, --json                 {"from":Stub,"rate":48000,"seconds":2,"channels":[{"channel":1,
                                  "peak_dbfs":-20.5,"rms_dbfs":-41.02,"clipped":false}],"stopped":"seconds"}
                                 --watch: JSON Lines {"at":"<RFC3339 UTC>","channels":[...]}
    \(commonOpts)

    \(stubHelp)

    \(targetHelp)
    """,
    "watch": """
    Usage:
      \(TOOL_NAME) watch [-j] [-v]

    One line per device added or removed and per change of the default input, output or
    system output, until Ctrl-C (exit 0). Driven by HAL property listeners with a 1 s poll
    as a backstop; a line prints only when what reads back changed.

    Options:
      -j, --json                 JSON Lines: {"event":"added","uid":"...","name":"...","at":"<RFC3339 UTC>"}
                                 default changes: {"event":"default","role":"output","uid":...,
                                 "name":...,"at":...} (uid/name null when there is none)
    \(commonOpts)
    """,
    "alias": """
    Usage:
      \(TOOL_NAME) alias [list]
      \(TOOL_NAME) alias set <alias> <target>
      \(TOOL_NAME) alias rm <alias>

    Aliases point at a device UID in ~/.config/macos-gui/audio_aliases.json
    ({"scarlett": "AppleUSBAudioEngine:..."}), so they survive renames, replugs and index
    shifts. Names: letters, digits, '.', '_', '-'; case-insensitive; not all digits.

    Commands:
      list                       Aliases and the device each resolves to (default)
      set <alias> <target>       Point alias at target's uid, then read it back
      rm <alias>                 Remove alias (also: remove); absent: changed:false, exit 0

    Options:
      -j, --json                 see '\(TOOL_NAME) alias <list|set|rm> --help'
    \(commonOpts)
    """,
    "alias list": """
    Usage:
      \(TOOL_NAME) alias [list] [-j] [-v]

    Every alias, its stored target and the device it resolves to (null when not present).

    Options:
      -j, --json                 {"aliases":[{"alias":"scarlett","target":"<uid>","device":Device|null}]}
    \(commonOpts)
    """,
    "alias set": """
    Usage:
      \(TOOL_NAME) alias set <alias> <target> [-j] [-v]

    Resolve <target> now and store its uid under <alias>, then read the file back.
    Already pointing there: nothing written (changed:false).

    Options:
      -j, --json                 {"success":true,"changed":true,"alias":"scarlett","was":null,
                                  "now":"<uid>","device":Device}
    \(commonOpts)

    \(targetHelp)
    """,
    "alias rm": """
    Usage:
      \(TOOL_NAME) alias rm <alias> [-j] [-v]

    Remove an alias and read the file back. Absent: nothing written (changed:false, exit 0).
    'remove' is accepted for 'rm'.

    Options:
      -j, --json                 {"success":true,"changed":true,"alias":"scarlett","was":"<uid>","now":null}
    \(commonOpts)
    """,
    "help": """
    Usage:
      \(TOOL_NAME) help [command [subcommand]]

    Show help for the tool, a command, or an alias subcommand (help alias set).
    Same as '\(TOOL_NAME) <command> --help'.
    """,
]

func showHelp(_ topic: String) -> Never {
    guard let text = helpTexts[topic] else { usageError("no help for '\(topic)'") }
    print(text.trimmingCharacters(in: .newlines))
    exit(ExitCode.ok)
}

// MARK: - Arguments

struct Opt { let long: String; let short: Character?; let takesValue: Bool }
struct Args {
    var cmd: String?
    var pos: [String] = []
    var flags = Set<String>()
    var values: [String: String] = [:]
}

let globalOpts = [Opt(long: "json", short: "j", takesValue: false), Opt(long: "verbose", short: "v", takesValue: false),
                  Opt(long: "help", short: "h", takesValue: false), Opt(long: "version", short: "V", takesValue: false)]
let timeoutOpt = Opt(long: "timeout", short: "t", takesValue: true)
let inputOpt = Opt(long: "input", short: nil, takesValue: false)
let channelsOpt = Opt(long: "channels", short: "c", takesValue: true)
let secondsOpt = Opt(long: "seconds", short: "s", takesValue: true)
let fromOpt = Opt(long: "from", short: nil, takesValue: true)
let commandOpts: [String: [Opt]] = [
    "list": [], "status": [], "watch": [], "alias": [], "help": [], "version": [],
    "default": [timeoutOpt],
    "volume": [inputOpt, timeoutOpt],
    "mute": [inputOpt, timeoutOpt],
    "play": [Opt(long: "to", short: nil, takesValue: true), channelsOpt, Opt(long: "gain", short: nil, takesValue: true)],
    "record": [fromOpt, channelsOpt, secondsOpt, Opt(long: "until-silence", short: nil, takesValue: true),
               Opt(long: "max", short: nil, takesValue: true), Opt(long: "threshold", short: nil, takesValue: true),
               Opt(long: "rate", short: nil, takesValue: true), Opt(long: "float", short: nil, takesValue: false),
               Opt(long: "out", short: "o", takesValue: true), Opt(long: "force", short: nil, takesValue: false)],
    "level": [fromOpt, channelsOpt, secondsOpt, Opt(long: "watch", short: nil, takesValue: false)],
]
let commandSynonyms = ["aliases": "alias"]

func parseArgs(_ argv: [String]) -> Args {
    // -j/-v first, wherever they are, so errors raised while parsing are already formatted right.
    for t in argv {
        if t == "--" { break }
        if t == "--json" { jsonMode = true }
        if t == "--verbose" { verboseMode = true }
        if t.count > 1, t.hasPrefix("-"), !t.hasPrefix("--"), t.dropFirst().allSatisfy({ "jvhV".contains($0) }) {
            if t.contains("j") { jsonMode = true }
            if t.contains("v") { verboseMode = true }
        }
    }
    var a = Args()
    var i = 0
    var optionsDone = false
    func specs() -> [Opt] { globalOpts + (a.cmd.flatMap { commandOpts[$0] } ?? []) }
    func forCmd() -> String { a.cmd.map { " for '\($0)'" } ?? "" }
    while i < argv.count {
        let t = argv[i]
        i += 1
        if optionsDone || t == "-" || !t.hasPrefix("-") {
            if a.cmd == nil {
                let c = commandSynonyms[t.lowercased()] ?? t.lowercased()
                guard commandOpts[c] != nil else { usageError("unknown command '\(t)'") }
                a.cmd = c
                helpTopic = c
            } else {
                a.pos.append(t)
            }
            continue
        }
        if t == "--" { optionsDone = true; continue }
        if t.hasPrefix("--") {
            var name = String(t.dropFirst(2))
            var value: String? = nil
            if let eq = name.firstIndex(of: "=") {
                value = String(name[name.index(after: eq)...])
                name = String(name[..<eq])
            }
            guard let o = specs().first(where: { $0.long == name }) else { usageError("unknown option '--\(name)'\(forCmd())") }
            if o.takesValue {
                if value == nil {
                    guard i < argv.count else { usageError("--\(name) needs a value") }
                    value = argv[i]
                    i += 1
                }
                a.values[o.long] = value
            } else {
                if value != nil { usageError("--\(name) takes no value") }
                a.flags.insert(o.long)
            }
            continue
        }
        var chars = Array(t.dropFirst())
        while !chars.isEmpty {
            let c = chars.removeFirst()
            guard let o = specs().first(where: { $0.short == c }) else { usageError("unknown option '-\(c)'\(forCmd())") }
            if o.takesValue {
                let value: String
                if !chars.isEmpty {
                    value = String(chars)
                    chars = []
                } else {
                    guard i < argv.count else { usageError("-\(c) needs a value") }
                    value = argv[i]
                    i += 1
                }
                a.values[o.long] = value
            } else {
                a.flags.insert(o.long)
            }
        }
    }
    if a.flags.contains("json") { jsonMode = true }
    if a.flags.contains("verbose") { verboseMode = true }
    return a
}

// MARK: - Main

func main() {
    setvbuf(stdout, nil, _IOLBF, 0)   // line-buffered even when piped: watch/level --watch stream live
    let argv = Array(CommandLine.arguments.dropFirst())
    if argv.isEmpty { showHelp("") }
    let a = parseArgs(argv)

    if a.flags.contains("version") || a.cmd == "version" {
        if jsonMode { emit(.obj([("name", .str(TOOL_NAME)), ("version", .str(VERSION))])) } else { say("\(TOOL_NAME) \(VERSION)") }
        exit(ExitCode.ok)
    }
    guard let cmd = a.cmd else {
        if a.flags.contains("help") { showHelp("") }
        usageError("missing command")
    }
    let subs: Set<String> = ["list", "ls", "set", "rm", "remove"]
    func aliasTopic(_ s: String?) -> String {
        guard let s = s?.lowercased(), subs.contains(s) else { return "alias" }
        return "alias " + (s == "remove" ? "rm" : s == "ls" ? "list" : s)
    }
    if cmd == "help" {
        guard let first = a.pos.first?.lowercased() else { showHelp("") }
        let c = commandSynonyms[first] ?? first
        showHelp(c == "alias" ? aliasTopic(a.pos.dropFirst().first) : c)
    }
    let p0 = a.pos.first?.lowercased()
    if cmd == "alias" {
        if a.flags.contains("help") || p0 == "help" {
            showHelp(aliasTopic(p0 == "help" ? a.pos.dropFirst().first : p0))
        }
        if a.pos.count > 1, a.pos[1].lowercased() == "help" { showHelp(aliasTopic(p0)) }
    } else if a.flags.contains("help") || p0 == "help" {
        showHelp(cmd)
    }

    switch cmd {
    case "list": cmdList(a)
    case "status": cmdStatus(a)
    case "default": cmdDefault(a)
    case "volume": cmdVolume(a)
    case "mute": cmdMute(a)
    case "play": cmdPlay(a)
    case "record": cmdRecord(a)
    case "level": cmdLevel(a)
    case "watch": cmdWatch(a)
    case "alias": cmdAlias(a)
    default: usageError("unknown command '\(cmd)'")
    }
    exit(ExitCode.ok)
}

main()
