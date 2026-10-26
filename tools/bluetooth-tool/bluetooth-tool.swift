// bluetooth-tool - native macOS Bluetooth CLI built on IOBluetooth.
//
// Commands: list, status, power, connect, disconnect, toggle, wait, watch, logs, alias.
// -j/--json output is a contract other programs parse (see README.md). Every verb that
// changes something is a read-back transaction: read the state, act only if it differs,
// read it back, and report what was READ, never what was requested.

import Foundation
import IOBluetooth
import IOKit
import CoreBluetooth

let VERSION = "0.1.0"
let TOOL_NAME = "bluetooth-tool"

enum ExitCode {
    static let ok: Int32 = 0
    static let failed: Int32 = 1
    static let usage: Int32 = 2
    static let refused: Int32 = 3       // a safety guard said no
    static let unconfirmed: Int32 = 4   // timed out, or the state never read back
}

// MARK: - Private IOBluetooth power API (the same entry points blueutil uses)

@_silgen_name("IOBluetoothPreferenceGetControllerPowerState")
func IOBluetoothPreferenceGetControllerPowerState() -> Int32

@_silgen_name("IOBluetoothPreferenceSetControllerPowerState")
func IOBluetoothPreferenceSetControllerPowerState(_ state: Int32)

func readPower() -> String { IOBluetoothPreferenceGetControllerPowerState() == 0 ? "off" : "on" }

func hostControllerPower() -> String {
    guard let hc = IOBluetoothHostController.default() else { return "no controller" }
    switch hc.powerState.rawValue {
    case 1: return "on"
    case 0: return "off"
    default: return "uninitialized"
    }
}

// MARK: - Output

var jsonMode = false
var verboseMode = false
var helpTopic = ""   // command being parsed, for "Run '... --help'" hints
let env = ProcessInfo.processInfo.environment
let colorOK = isatty(STDOUT_FILENO) == 1 && env["NO_COLOR"] == nil && env["TERM"] != "dumb"

/// Ordered JSON value, so output keys appear in contract order.
enum JSON {
    case str(String), int(Int), bool(Bool), null, arr([JSON]), obj([(String, JSON)])

    var text: String {
        switch self {
        case .str(let s): return JSON.quote(s)
        case .int(let i): return String(i)
        case .bool(let b): return b ? "true" : "false"
        case .null: return "null"
        case .arr(let a): return "[" + a.map(\.text).joined(separator: ",") + "]"
        case .obj(let o): return "{" + o.map { JSON.quote($0.0) + ":" + $0.1.text }.joined(separator: ",") + "}"
        }
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

func fail(_ code: Int32, _ msg: String, device: Dev? = nil) -> Never {
    if jsonMode {
        var o: [(String, JSON)] = [("success", .bool(false)), ("error", .str(msg))]
        if let d = device { o.append(("device", d.json)) }
        emit(.obj(o))
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

// MARK: - Devices

struct Dev {
    var index = 0
    let name: String
    let address: String   // lowercase, colon-separated, zero-padded
    let cod: UInt32       // class of device; 0 for most BLE devices
    let paired: Bool
    var connected: Bool
    var alias: String?
    let raw: IOBluetoothDevice

    var kind: String { kindOf(self) }
    var label: String { "\(name) (\(address))" }
    var stateWord: String { connected ? "connected" : "disconnected" }
    var json: JSON {
        .obj([("index", .int(index)), ("name", .str(name)), ("address", .str(address)),
              ("kind", .str(kind)), ("paired", .bool(paired)), ("connected", .bool(connected)),
              ("alias", alias.map { .str($0) } ?? .null)])
    }
}

/// Accepts aa:bb:cc:dd:ee:ff, aa-bb-cc-dd-ee-ff or aabbccddeeff (any case); returns aa:bb:cc:dd:ee:ff.
func normalizeAddress(_ s: String) -> String? {
    let t = s.trimmingCharacters(in: .whitespaces).lowercased()
    let groups: [String]
    if t.contains(":") || t.contains("-") {
        groups = t.split(separator: t.contains(":") ? ":" : "-", omittingEmptySubsequences: false).map(String.init)
    } else if t.count == 12 {
        let c = Array(t)
        groups = stride(from: 0, to: 12, by: 2).map { String(c[$0..<$0 + 2]) }
    } else {
        return nil
    }
    guard groups.count == 6,
          groups.allSatisfy({ (1...2).contains($0.count) && $0.allSatisfy { $0.isASCII && $0.isHexDigit } })
    else { return nil }
    return groups.map { $0.count == 1 ? "0" + $0 : $0 }.joined(separator: ":")
}

func addressOf(_ d: IOBluetoothDevice) -> String {
    if let p = d.getAddress() {
        let b = p.pointee.data
        return [b.0, b.1, b.2, b.3, b.4, b.5].map { $0 < 16 ? "0" + String($0, radix: 16) : String($0, radix: 16) }
            .joined(separator: ":")
    }
    let s = d.addressString ?? ""
    return normalizeAddress(s) ?? s.lowercased()
}

/// Human kind from the Bluetooth class of device (major bits 8-12, minor bits 2-7).
func kindFromClass(_ cod: UInt32) -> String {
    let major = (cod >> 8) & 0x1F
    let minor = (cod >> 2) & 0x3F
    switch major {
    case 1: return "Computer"
    case 2: return "Phone"
    case 3: return "Network"
    case 4:
        switch minor {
        case 1, 2: return "Headset"
        case 4: return "Microphone"
        case 5, 7, 8, 10: return "Speaker"
        case 6: return "Headphones"
        case 12, 13: return "Camera"
        case 14, 15, 16: return "Video Display"
        case 18: return "Gamepad"
        default: return "Unknown"
        }
    case 5:
        switch minor >> 4 {
        case 1, 3: return "Keyboard"   // 3 = combo keyboard/pointing
        case 2: return (minor & 0xF) == 5 ? "Trackpad" : "Mouse"
        default:
            switch minor & 0xF {
            case 1, 2: return "Gamepad"
            case 3: return "Remote"
            default: return "Unknown"
            }
        }
    case 6:
        if minor & 0x20 != 0 { return "Printer" }
        if minor & 0x08 != 0 { return "Camera" }
        if minor & 0x04 != 0 { return "Video Display" }
        return "Unknown"
    case 7: return "Wearable"
    default: return "Unknown"
    }
}

/// system_profiler's device_minorType mapped onto the same vocabulary.
func kindFromProfiler(_ s: String) -> String {
    switch s.lowercased() {
    case "keyboard": return "Keyboard"
    case "mouse": return "Mouse"
    case "trackpad": return "Trackpad"
    case "gamepad", "game controller", "joystick": return "Gamepad"
    case "headphones": return "Headphones"
    case "headset": return "Headset"
    case "speaker", "loudspeaker", "portable audio", "hifi audio", "car audio": return "Speaker"
    case "microphone": return "Microphone"
    case "video display", "display", "tv": return "Video Display"
    case "phone", "smartphone", "cellular": return "Phone"
    case "computer", "desktop", "laptop", "server": return "Computer"
    case "remote", "remote control": return "Remote"
    case "camera": return "Camera"
    case "printer": return "Printer"
    case "watch", "wearable": return "Wearable"
    default: return "Unknown"
    }
}

/// BLE devices carry no class of device, so their kind comes from macOS's own record
/// (bluetoothd, via system_profiler). Loaded once per process, only when needed.
var profilerKindCache: [String: String]? = nil
func profilerKinds() -> [String: String] {
    if let c = profilerKindCache { return c }
    let t0 = Date()
    var map: [String: String] = [:]
    if let data = runCapture("/usr/sbin/system_profiler", ["SPBluetoothDataType", "-json"], timeout: 10),
       let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
       let items = root["SPBluetoothDataType"] as? [[String: Any]] {
        for item in items {
            for (key, value) in item where key.hasPrefix("device_") {
                guard let list = value as? [[String: Any]] else { continue }
                for entry in list {
                    for (_, props) in entry {
                        guard let p = props as? [String: Any],
                              let a = (p["device_address"] as? String).flatMap(normalizeAddress),
                              let minor = p["device_minorType"] as? String else { continue }
                        map[a] = kindFromProfiler(minor)
                    }
                }
            }
        }
    }
    vlog("system_profiler: \(map.count) device kinds in \(ms(since: t0)) ms")
    profilerKindCache = map
    return map
}

func kindOf(_ d: Dev) -> String {
    let k = kindFromClass(d.cod)
    return k != "Unknown" ? k : (profilerKinds()[d.address] ?? "Unknown")
}

func runCapture(_ path: String, _ args: [String], timeout: TimeInterval) -> Data? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
    let out = Pipe()
    p.standardOutput = out
    p.standardError = FileHandle.nullDevice
    p.standardInput = FileHandle.nullDevice
    do { try p.run() } catch { vlog("can't run \(path): \(error.localizedDescription)"); return nil }
    DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { if p.isRunning { p.terminate() } }
    let data = out.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    return p.terminationStatus == 0 ? data : nil
}

// MARK: - Permission (TCC)

func tccState() -> String {
    switch CBManager.authorization {
    case .allowedAlways: return "allowed"
    case .denied: return "denied"
    case .restricted: return "restricted"
    case .notDetermined: return "not determined"
    @unknown default: return "unknown"
    }
}

func tccBlocked() -> Bool { CBManager.authorization == .denied || CBManager.authorization == .restricted }

func tccHint() -> String {
    "Bluetooth permission reads '\(tccState())'. The app that launched \(TOOL_NAME) (your terminal, or a parent "
        + ".app such as TopicGet.app) needs Bluetooth access: System Settings > Privacy & Security > Bluetooth."
}

// MARK: - Aliases (~/.config/macos-gui/bluetooth_aliases.json: {"alias": "aa:bb:cc:dd:ee:ff"})

let home = env["HOME"] ?? NSHomeDirectory()
let aliasPath = home + "/.config/macos-gui/bluetooth_aliases.json"

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

struct Snapshot { let power: String; let devices: [Dev] }

/// Address match, else exact name, else unique name substring (case-insensitive).
func lookupStored(_ value: String, _ devs: [Dev]) -> Dev? {
    if let a = normalizeAddress(value) { return devs.first { $0.address == a } }
    let l = value.lowercased()
    let exact = devs.filter { $0.name.lowercased() == l }
    if exact.count == 1 { return exact[0] }
    let hits = devs.filter { $0.name.lowercased().contains(l) }
    return hits.count == 1 ? hits[0] : nil
}

func buildDevices(_ raw: [IOBluetoothDevice]) -> [Dev] {
    var seen = Set<String>()
    var devs: [Dev] = []
    for d in raw {
        let addr = addressOf(d)
        guard seen.insert(addr).inserted else { continue }
        let n = (d.name ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        devs.append(Dev(name: n.isEmpty ? addr : n, address: addr, cod: d.classOfDevice,
                        paired: d.isPaired(), connected: d.isConnected(), alias: nil, raw: d))
    }
    devs.sort { ($0.name.lowercased(), $0.address) < ($1.name.lowercased(), $1.address) }
    for i in devs.indices { devs[i].index = i + 1 }
    for (name, value) in aliases().sorted(by: { $0.key < $1.key }) {
        if let d = lookupStored(value, devs), let i = devs.firstIndex(where: { $0.address == d.address }),
           devs[i].alias == nil {
            devs[i].alias = name
        }
    }
    return devs
}

func pairedRaw() -> [IOBluetoothDevice]? {
    guard let list = IOBluetoothDevice.pairedDevices() else { return nil }
    return list.compactMap { $0 as? IOBluetoothDevice }
}

func takeSnapshot() -> Snapshot {
    let t0 = Date()
    let power = readPower()
    guard let raw = pairedRaw() else { fail(ExitCode.failed, "IOBluetooth returned no paired-device list. \(tccHint())") }
    let devs = buildDevices(raw)
    vlog("power \(power) (private getter; host controller: \(hostControllerPower())), \(devs.count) paired, "
         + "\(devs.filter(\.connected).count) connected, \(ms(since: t0)) ms; Bluetooth permission: \(tccState())")
    if devs.isEmpty && tccBlocked() { fail(ExitCode.failed, "IOBluetooth returned no paired devices. \(tccHint())") }
    return Snapshot(power: power, devices: devs)
}

/// Target = 1-based index | alias | address | name (exact, then unique substring).
func resolve(_ query: String, _ snap: Snapshot) -> Dev {
    let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
    if q.isEmpty { usageError("empty target") }
    let devs = snap.devices
    if q.allSatisfy({ $0.isASCII && $0.isNumber }) {
        guard let n = Int(q), n >= 1, n <= devs.count else {
            fail(ExitCode.failed, devs.isEmpty ? "no paired devices (target '\(q)')" : "no device at index \(q) (have 1-\(devs.count))")
        }
        return devs[n - 1]
    }
    if let v = aliases()[q.lowercased()] {
        if let d = lookupStored(v, devs) { return d }
        fail(ExitCode.failed, "alias '\(q.lowercased())' points to '\(v)', which matches no paired device")
    }
    if let a = normalizeAddress(q) {
        if let d = devs.first(where: { $0.address == a }) { return d }
        fail(ExitCode.failed, "no paired device with address \(a)")
    }
    let l = q.lowercased()
    let exact = devs.filter { $0.name.lowercased() == l }
    if exact.count == 1 { return exact[0] }
    let hits = exact.isEmpty ? devs.filter { $0.name.lowercased().contains(l) } : exact
    if hits.count == 1 { return hits[0] }
    if hits.isEmpty { fail(ExitCode.failed, "no paired device matches '\(q)'") }
    fail(ExitCode.failed, "ambiguous target '\(q)' matches \(hits.count) devices: "
         + hits.map { "\($0.index) \($0.label)" }.joined(separator: "; "))
}

// MARK: - Power-off guard: Bluetooth keyboards, mice and trackpads

struct HIDInput { let product: String; let kind: String; let transport: String; let address: String? }
struct Blocker { let label: String; let device: Dev? }
let inputKinds: Set<String> = ["Keyboard", "Mouse", "Trackpad"]

func hidInputKind(page: Int, usage: Int) -> String? {
    switch (page, usage) {
    case (0x01, 0x06), (0x01, 0x07): return "Keyboard"
    case (0x0D, 0x05): return "Trackpad"
    case (0x01, 0x01), (0x01, 0x02): return "Mouse"
    default: return nil
    }
}

func isBluetoothTransport(_ t: String) -> Bool { t.lowercased().contains("bluetooth") || t.uppercased().hasPrefix("BT") }

/// Live input devices in the IOKit HID registry whose transport is Bluetooth. Catches BLE
/// keyboards that IOBluetooth reports with class 0 or never marks connected.
func bluetoothHIDInputs() -> [HIDInput] {
    var iter: io_iterator_t = 0
    guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOHIDDevice"), &iter) == KERN_SUCCESS else {
        vlog("HID registry: query failed")
        return []
    }
    defer { IOObjectRelease(iter) }
    var found: [HIDInput] = []
    while true {
        let svc = IOIteratorNext(iter)
        if svc == 0 { break }
        defer { IOObjectRelease(svc) }
        func prop(_ key: String) -> Any? {
            IORegistryEntryCreateCFProperty(svc, key as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue()
        }
        guard let transport = prop("Transport") as? String, isBluetoothTransport(transport) else { continue }
        var usages: [(Int, Int)] = []
        if let p = prop("PrimaryUsagePage") as? Int, let u = prop("PrimaryUsage") as? Int { usages.append((p, u)) }
        if let pairs = prop("DeviceUsagePairs") as? [[String: Any]] {
            for pr in pairs {
                if let p = pr["DeviceUsagePage"] as? Int, let u = pr["DeviceUsage"] as? Int { usages.append((p, u)) }
            }
        }
        let kinds = Set(usages.compactMap { hidInputKind(page: $0.0, usage: $0.1) })
        guard let kind = ["Keyboard", "Trackpad", "Mouse"].first(where: kinds.contains) else { continue }
        var addr: String? = nil
        for key in ["DeviceAddress", "SerialNumber", "PhysicalDeviceUniqueID"] {
            if let v = prop(key) as? String, let a = normalizeAddress(v) { addr = a; break }
        }
        let product = ((prop("Product") as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        found.append(HIDInput(product: product.isEmpty ? "Bluetooth \(kind.lowercased())" : product,
                              kind: kind, transport: transport, address: addr))
    }
    vlog("HID registry: \(found.count) Bluetooth input device(s)")
    return found
}

func inputBlockers(_ snap: Snapshot) -> [Blocker] {
    var out: [Blocker] = []
    var seen = Set<String>()
    for d in snap.devices where d.connected && inputKinds.contains(d.kind) {
        seen.insert(d.address)
        out.append(Blocker(label: "\(d.kind.lowercased()) '\(d.name)' (\(d.address))", device: d))
    }
    for h in bluetoothHIDInputs() {
        let match = h.address.flatMap { a in snap.devices.first { $0.address == a } }
            ?? snap.devices.first { $0.name.caseInsensitiveCompare(h.product) == .orderedSame }
        if let d = match {
            if seen.insert(d.address).inserted {
                out.append(Blocker(label: "\(h.kind.lowercased()) '\(d.name)' (\(d.address))", device: d))
            }
        } else if seen.insert("hid:" + h.product).inserted {
            out.append(Blocker(label: "\(h.kind.lowercased()) '\(h.product)' (HID registry, \(h.transport))", device: nil))
        }
    }
    return out
}

// MARK: - Waiting

/// Runs the main run loop (so IOBluetooth can deliver state changes) for `seconds`.
func pump(_ seconds: TimeInterval) {
    let until = Date(timeIntervalSinceNow: seconds)
    while true {
        let left = until.timeIntervalSinceNow
        if left <= 0 { return }
        if CFRunLoopRunInMode(.defaultMode, left, true) == .finished {
            usleep(useconds_t(min(left, 0.05) * 1_000_000))
        }
    }
}

/// Re-reads `check` until it holds or `timeout` passes.
func poll(timeout: TimeInterval, every: TimeInterval = 0.1, _ check: () -> Bool) -> Bool {
    let deadline = Date(timeIntervalSinceNow: timeout)
    while true {
        if check() { return true }
        let left = deadline.timeIntervalSinceNow
        if left <= 0 { return false }
        pump(min(every, left))
    }
}

func ioReturnText(_ rc: IOReturn) -> String {
    String(format: "0x%08x", UInt32(bitPattern: rc)) + " (" + String(cString: mach_error_string(rc)) + ")"
}

// MARK: - Commands

func printTable(_ devs: [Dev]) {
    let idxW = String(devs.map(\.index).max() ?? 1).count
    let nameW = max(4, devs.map { $0.name.count }.max() ?? 4)
    let kindW = max(4, devs.map { $0.kind.count }.max() ?? 4)
    say("  " + pad("#", idxW, right: true) + "  " + pad("NAME", nameW) + "  " + pad("ADDRESS", 17) + "  "
        + pad("KIND", kindW) + "  " + pad("STATE", 9) + "  ALIAS")
    for d in devs {
        let state = d.connected ? paint(pad("connected", 9), "32") : pad("-", 9)
        say("  " + pad(String(d.index), idxW, right: true) + "  " + pad(d.name, nameW) + "  " + d.address + "  "
            + pad(d.kind, kindW) + "  " + state + "  " + (d.alias ?? "-"))
    }
}

func powerWord(_ p: String) -> String { p == "on" ? paint("on", "32") : paint("off", "33") }

func cmdList(_ a: Args) {
    if let extra = a.pos.first { usageError("unexpected argument '\(extra)'") }
    let snap = takeSnapshot()
    if jsonMode {
        emit(.obj([("power", .str(snap.power)), ("devices", .arr(snap.devices.map(\.json)))]))
        return
    }
    say("Bluetooth: \(powerWord(snap.power))  (\(snap.devices.count) paired, \(snap.devices.filter(\.connected).count) connected)")
    if snap.devices.isEmpty { say("No paired devices."); return }
    say("")
    printTable(snap.devices)
}

func cmdStatus(_ a: Args) {
    if let extra = a.pos.first { usageError("unexpected argument '\(extra)'") }
    let snap = takeSnapshot()
    let conn = snap.devices.filter(\.connected)
    if jsonMode {
        emit(.obj([("power", .str(snap.power)), ("connected", .arr(conn.map(\.json)))]))
        return
    }
    say("Bluetooth: \(powerWord(snap.power))")
    if conn.isEmpty {
        say("Connected: none")
    } else {
        say("Connected (\(conn.count)):")
        printTable(conn)
    }
    let blockers = inputBlockers(snap)
    say("Power-off guard: " + (blockers.isEmpty
        ? "clear (no Bluetooth keyboard, mouse or trackpad connected)"
        : "would refuse (" + blockers.map(\.label).joined(separator: ", ") + ")"))
}

func timeoutValue(_ a: Args, default d: Double = 10) -> Double {
    guard let s = a.values["timeout"] else { return d }
    guard let v = Double(s), v > 0, v <= 86400 else { usageError("--timeout takes seconds (0 < s <= 86400), not '\(s)'") }
    return v
}

func cmdPower(_ a: Args) {
    if a.pos.count > 1 { usageError("unexpected argument '\(a.pos[1])'") }
    let arg = a.pos.first?.lowercased()
    if let arg, !["on", "off", "toggle"].contains(arg) { usageError("power takes on, off or toggle, not '\(a.pos[0])'") }
    let timeout = timeoutValue(a)
    let t0 = Date()
    let was = readPower()
    vlog("power getter: \(was); host controller: \(hostControllerPower()); Bluetooth permission: \(tccState())")
    guard let arg else {
        if jsonMode { emit(.obj([("power", .str(was))])) } else { say(was) }
        return
    }
    func result(_ now: String, _ changed: Bool) -> JSON {
        .obj([("success", .bool(true)), ("was", .str(was)), ("now", .str(now)), ("changed", .bool(changed))])
    }
    let want = arg == "toggle" ? (was == "on" ? "off" : "on") : arg
    if want == was {
        if jsonMode { emit(result(was, false)) } else { say("Bluetooth already \(was); nothing sent") }
        return
    }
    if want == "off" {
        let blockers = inputBlockers(takeSnapshot())
        if !blockers.isEmpty {
            let what = blockers.map(\.label).joined(separator: ", ")
            if !a.flags.contains("force") {
                fail(ExitCode.refused, "refusing to turn Bluetooth off: it would cut off \(what); pass --force to do it anyway",
                     device: blockers.first?.device)
            }
            warn("warning: --force: cutting off \(what)")
        }
    }
    narrate("Turning Bluetooth \(want)...")
    IOBluetoothPreferenceSetControllerPowerState(want == "on" ? 1 : 0)
    let confirmed = poll(timeout: timeout) { readPower() == want }
    let now = readPower()
    vlog("read-back after \(ms(since: t0)) ms: \(now); host controller: \(hostControllerPower())")
    guard confirmed else {
        fail(ExitCode.unconfirmed, "unconfirmed: asked for power \(want) but it still reads \(now) after \(secs(timeout)) s. "
             + "macOS may not honor the private IOBluetoothPreferenceSetControllerPowerState for this process "
             + "(Bluetooth permission: \(tccState())); use Control Center or System Settings.")
    }
    let hc = hostControllerPower()
    if (hc == "on" || hc == "off") && hc != now {
        warn("warning: the private getter reads \(now) but IOBluetoothHostController still reads \(hc)")
    }
    if jsonMode { emit(result(now, true)) } else { say("Bluetooth \(was) -> \(now) (\(ms(since: t0)) ms)") }
}

/// connect | disconnect | toggle as one read-back transaction.
func cmdConnection(_ verb: String, _ a: Args) {
    guard let target = a.pos.first else { usageError("\(verb) needs a target (index, alias, name or address)") }
    if a.pos.count > 1 { usageError("unexpected argument '\(a.pos[1])'") }
    let timeout = timeoutValue(a)
    let t0 = Date()
    let snap = takeSnapshot()
    var dev = resolve(target, snap)
    let was = dev.raw.isConnected()
    let want = verb == "toggle" ? !was : verb == "connect"
    let action = want ? "connect" : "disconnect"
    func word(_ c: Bool) -> String { c ? "connected" : "disconnected" }
    func report(_ changed: Bool) {
        if jsonMode {
            emit(.obj([("success", .bool(true)), ("action", .str(action)), ("changed", .bool(changed)),
                       ("was", .str(word(was))), ("now", .str(dev.stateWord)), ("device", dev.json),
                       ("ms", .int(ms(since: t0)))]))
        } else if changed {
            say("\(action == "connect" ? "Connected" : "Disconnected") \(dev.label) in \(ms(since: t0)) ms")
        } else {
            say("\(dev.label) already \(dev.stateWord); nothing sent")
        }
    }
    dev.connected = was
    if was == want { report(false); return }
    if want && snap.power == "off" {
        fail(ExitCode.failed, "Bluetooth is off; turn it on first (\(TOOL_NAME) power on)", device: dev)
    }
    narrate("\(want ? "Connecting" : "Disconnecting") \(dev.label)...")
    let call = want ? "openConnection" : "closeConnection"
    let rc = want ? dev.raw.openConnection() : dev.raw.closeConnection()
    vlog("\(call) -> \(ioReturnText(rc)) after \(ms(since: t0)) ms")
    if rc != kIOReturnSuccess {
        dev.connected = dev.raw.isConnected()
        if dev.connected != want {
            fail(ExitCode.failed, "\(call) failed: \(ioReturnText(rc)); \(dev.name) reads \(dev.stateWord)", device: dev)
        }
        warn("warning: \(call) returned \(ioReturnText(rc)), but the device reads \(dev.stateWord)")
    }
    let confirmed = poll(timeout: max(0.1, timeout - Date().timeIntervalSince(t0))) { dev.raw.isConnected() == want }
    dev.connected = dev.raw.isConnected()
    guard confirmed else {
        fail(ExitCode.unconfirmed, "unconfirmed: asked to \(action) \(dev.name), but it still reads \(dev.stateWord) "
             + "after \(secs(timeout)) s", device: dev)
    }
    report(true)
}

func cmdWait(_ a: Args) {
    guard a.pos.count == 2 else {
        usageError(a.pos.count < 2 ? "wait needs <target> <connected|disconnected>" : "unexpected argument '\(a.pos[2])'")
    }
    let state = a.pos[1].lowercased()
    guard state == "connected" || state == "disconnected" else {
        usageError("state must be 'connected' or 'disconnected', not '\(a.pos[1])'")
    }
    let timeout = timeoutValue(a)
    let want = state == "connected"
    let snap = takeSnapshot()
    var dev = resolve(a.pos[0], snap)
    let t0 = Date()
    if dev.raw.isConnected() != want {
        if want && snap.power == "off" { warn("warning: Bluetooth is off") }
        narrate("Waiting up to \(secs(timeout)) s for \(dev.label) to be \(state)...")
    }
    let ok = poll(timeout: timeout, every: 0.2) { dev.raw.isConnected() == want }
    dev.connected = dev.raw.isConnected()
    let waited = ms(since: t0)
    guard ok else {
        fail(ExitCode.unconfirmed, "timed out after \(secs(timeout)) s: \(dev.name) is still \(dev.stateWord)", device: dev)
    }
    if jsonMode {
        emit(.obj([("success", .bool(true)), ("state", .str(state)), ("device", dev.json), ("waited_ms", .int(waited))]))
    } else {
        say("\(dev.label) is \(state) (waited \(waited) ms)")
    }
}

var keepAlive: [AnyObject] = []

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

/// Event-driven (IOBluetooth connect + per-device disconnect notifications) with a 1 s
/// poll as a backstop. A line prints only when the read state of a device changes.
final class Watcher: NSObject {
    let only: String?
    var last: [String: Dev] = [:]
    var disconnectNotes: [String: IOBluetoothUserNotification] = [:]
    var connectNote: IOBluetoothUserNotification?

    init(only: String?, initial: [Dev]) {
        self.only = only
        super.init()
        for d in initial { last[d.address] = d }
        trackDisconnects(initial)
    }

    func start() {
        connectNote = IOBluetoothDevice.register(forConnectNotifications: self, selector: #selector(changed(_:device:)))
        let t = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in self?.rescan() }
        RunLoop.main.add(t, forMode: .default)
    }

    @objc func changed(_ note: IOBluetoothUserNotification, device: IOBluetoothDevice) { rescan() }

    func rescan() {
        guard let raw = pairedRaw() else { return }
        let devs = buildDevices(raw)
        var current: [String: Dev] = [:]
        for d in devs { current[d.address] = d }
        for d in devs where only == nil || d.address == only {
            if d.connected != (last[d.address]?.connected ?? false) { report(d.connected ? "connected" : "disconnected", d) }
        }
        for (addr, old) in last where current[addr] == nil && old.connected && (only == nil || addr == only) {
            var gone = old
            gone.connected = false
            report("disconnected", gone)
        }
        last = current
        trackDisconnects(devs)
    }

    func trackDisconnects(_ devs: [Dev]) {
        for d in devs where d.connected && disconnectNotes[d.address] == nil {
            disconnectNotes[d.address] = d.raw.register(forDisconnectNotification: self, selector: #selector(changed(_:device:)))
        }
        for (addr, note) in disconnectNotes where !(devs.first(where: { $0.address == addr })?.connected ?? false) {
            note.unregister()
            disconnectNotes[addr] = nil
        }
    }

    func report(_ event: String, _ d: Dev) {
        let now = Date()
        if jsonMode {
            emit(.obj([("at", .str(rfc3339.string(from: now))), ("event", .str(event)), ("device", d.json)]))
        } else {
            let word = event == "connected" ? paint(pad(event, 12), "32") : paint(pad(event, 12), "33")
            say("\(localStamp.string(from: now))  \(word)  \(d.index)  \(d.label)  \(d.kind)")
        }
    }
}

func cmdWatch(_ a: Args) {
    if a.pos.count > 1 { usageError("unexpected argument '\(a.pos[1])'") }
    let snap = takeSnapshot()
    let only = a.pos.first.map { resolve($0, snap) }
    if snap.power == "off" { warn("warning: Bluetooth is off") }
    let w = Watcher(only: only?.address, initial: snap.devices)
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
    narrate(only.map { "Watching \($0.label); Ctrl-C to stop" }
            ?? "Watching \(snap.devices.count) paired devices; Ctrl-C to stop")
    w.start()
    while true { _ = RunLoop.main.run(mode: .default, before: .distantFuture) }
}

// MARK: - Logs

struct LogEntry {
    let time: String, local: String, process: String, message: String
    var json: JSON { .obj([("time", .str(time)), ("process", .str(process)), ("message", .str(message))]) }
    var human: String { "\(local)  \(process)  \(message.replacingOccurrences(of: "\n", with: "\n    "))" }
}

let logStampParser: DateFormatter = {
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSSSSSZ"
    return f
}()

func parseLogLine(_ line: Data) -> LogEntry? {
    guard let o = (try? JSONSerialization.jsonObject(with: Data(line))) as? [String: Any],
          let msg = o["eventMessage"] as? String else { return nil }
    if let type = o["eventType"] as? String, type != "logEvent" { return nil }
    let ts = o["timestamp"] as? String ?? ""
    let process = ((o["processImagePath"] as? String) ?? "").split(separator: "/").last.map(String.init) ?? "?"
    let time = logStampParser.date(from: ts).map { rfc3339.string(from: $0) } ?? ts
    return LogEntry(time: time, local: String(ts.prefix(23)), process: process, message: msg)
}

final class LockedData {
    private var d = Data()
    private let lock = NSLock()
    func append(_ x: Data) { lock.lock(); d.append(x); lock.unlock() }
    var text: String { lock.lock(); defer { lock.unlock() }; return String(decoding: d, as: UTF8.self) }
}

var interrupted = false

func cmdLogs(_ a: Args) {
    if let extra = a.pos.first { usageError("unexpected argument '\(extra)' (filter with --grep)") }
    let follow = a.flags.contains("follow")
    if follow && a.values["last"] != nil { usageError("--last and --follow don't combine; --follow streams from now") }
    if follow && a.values["limit"] != nil { usageError("--limit doesn't apply to --follow") }
    let last = a.values["last"] ?? "2h"
    guard last == "boot" || last.range(of: "^[0-9]+[mhd]?$", options: .regularExpression) != nil else {
        usageError("--last takes <number>[m|h|d] or 'boot' (e.g. 30m, 2h, 1d), not '\(last)'")
    }
    var limit = 200
    if let l = a.values["limit"] {
        guard let n = Int(l), n >= 0 else { usageError("--limit takes a whole number >= 0, not '\(l)'") }
        limit = n
    }
    var predicate = "(process == \"bluetoothd\" OR subsystem == \"com.apple.bluetooth\")"
        + " AND NOT (processImagePath ENDSWITH \"/\(TOOL_NAME)\")"
    if let g = a.values["grep"] {
        if g.isEmpty { usageError("--grep needs non-empty text") }
        let escaped = g.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        predicate += " AND eventMessage CONTAINS[c] \"\(escaped)\""
    }
    let args = follow
        ? ["stream", "--style", "ndjson", "--predicate", predicate]
        : ["show", "--style", "ndjson", "--last", last, "--predicate", predicate]
    vlog("/usr/bin/log " + args.map { $0.contains(" ") ? "'\($0)'" : $0 }.joined(separator: " "))

    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/log")   // not `log`: zsh has a builtin of that name
    p.arguments = args
    let outPipe = Pipe(), errPipe = Pipe()
    p.standardOutput = outPipe
    p.standardError = errPipe
    p.standardInput = FileHandle.nullDevice
    let errText = LockedData()
    errPipe.fileHandleForReading.readabilityHandler = { h in
        let d = h.availableData
        if d.isEmpty { h.readabilityHandler = nil } else { errText.append(d) }
    }
    do { try p.run() } catch { fail(ExitCode.failed, "can't run /usr/bin/log: \(error.localizedDescription)") }
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        signal(sig, SIG_IGN)
        let s = DispatchSource.makeSignalSource(signal: sig, queue: .global())
        s.setEventHandler { interrupted = true; if p.isRunning { p.terminate() } }
        s.resume()
        keepAlive.append(s)
    }

    let t0 = Date()
    let marker = Data("\"eventType\":\"logEvent\"".utf8)   // cheap pre-filter; activity events also carry eventMessage
    var ring: [Data] = []          // newest `limit` raw lines (limit > 0)
    var entries: [LogEntry] = []   // every entry (limit == 0)
    var seen = 0
    func handle(_ line: Data) {
        guard line.range(of: marker) != nil else { return }
        if follow {
            guard let e = parseLogLine(line) else { return }
            if jsonMode { emit(e.json) } else { say(e.human) }
        } else if limit == 0 {
            if let e = parseLogLine(line) { entries.append(e) }
        } else {
            seen += 1
            ring.append(Data(line))
            if ring.count > limit { ring.removeFirst(ring.count - limit) }
        }
    }
    let fh = outPipe.fileHandleForReading
    var buf = Data()
    while true {
        let chunk = fh.availableData
        if chunk.isEmpty { break }
        buf.append(chunk)
        var start = buf.startIndex
        while let nl = buf[start...].firstIndex(of: 0x0A) {
            handle(buf[start..<nl])
            start = buf.index(after: nl)
        }
        buf = Data(buf[start...])
    }
    if !buf.isEmpty { handle(buf) }
    p.waitUntilExit()
    errPipe.fileHandleForReading.readabilityHandler = nil
    if interrupted { exit(follow ? ExitCode.ok : 130) }
    if p.terminationStatus != 0 {
        let why = errText.text.trimmingCharacters(in: .whitespacesAndNewlines)
        fail(ExitCode.failed, "/usr/bin/log \(args[0]) exited \(p.terminationStatus)" + (why.isEmpty ? "" : ": \(why)"))
    }
    if follow { return }
    if limit > 0 { entries = ring.compactMap(parseLogLine) }
    vlog("log show: \(limit == 0 ? entries.count : seen) entries in \(ms(since: t0)) ms")
    if jsonMode {
        emit(.obj([("entries", .arr(entries.map(\.json)))]))
        return
    }
    if entries.isEmpty {
        narrate("No Bluetooth log entries in the last \(last)" + (a.values["grep"].map { " matching '\($0)'" } ?? "") + ".")
        return
    }
    for e in entries { say(e.human) }
    if limit > 0 && seen > limit { narrate("(newest \(limit) of \(seen); --limit 0 for all)") }
}

// MARK: - Alias

let reservedAliases: Set<String> = ["help", "list", "set", "rm", "remove"]

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
    let devs = all.isEmpty ? [] : (pairedRaw().map(buildDevices) ?? [])
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
    for r in rows { say(pad(r.alias, w) + "  " + pad(r.target, tw) + "  " + (r.dev.map { "\($0.index) \($0.name)" } ?? "(not paired)")) }
}

func aliasSet(_ rawName: String, _ target: String) {
    let name = rawName.lowercased()
    guard name.range(of: "^[a-z0-9][a-z0-9._-]*$", options: .regularExpression) != nil else {
        usageError("alias '\(rawName)' must be letters, digits, '.', '_' or '-' (starting with a letter or digit)")
    }
    if name.allSatisfy(\.isNumber) { usageError("alias '\(rawName)' is all digits; that would shadow an index") }
    if normalizeAddress(name) != nil { usageError("alias '\(rawName)' looks like an address") }
    if reservedAliases.contains(name) { usageError("'\(name)' is reserved") }
    var all = strictAliases()
    let snap = takeSnapshot()
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
    if let was, normalizeAddress(was) == dev.address { report(false, was); return }
    all[name] = dev.address
    do { try writeAliases(all) } catch { fail(ExitCode.failed, "can't write \(aliasPath): \(error.localizedDescription)", device: dev) }
    guard case .ok(let back) = readAliasFile(), let now = back[name], normalizeAddress(now) == dev.address else {
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
  3                          1-based index from 'list' (shifts when devices pair/unpair)
  receiver                   alias (alias set receiver 3)
  dh590                      name substring, case-insensitive (ambiguous = error)
  94:08:53:f1:06:55          address; also 94-08-53-f1-06-55 or 940853f10655
"""

let deviceHelp = """
Device JSON:
  {"index":1,"name":"STR-DH590","address":"94:08:53:f1:06:55","kind":"Speaker",
   "paired":true,"connected":false,"alias":"receiver"}      (alias is null when unset)
"""

let commonOpts = """
  -v, --verbose              API calls, return codes and timings on stderr
  -h, --help                 Show this help
"""

func connectionHelp(_ verb: String) -> String {
    let what: String
    switch verb {
    case "connect": what = "Connect a paired device (openConnection). Already connected: nothing is sent."
    case "disconnect": what = "Disconnect a device (closeConnection). Already disconnected: nothing is sent."
    default: what = "Connect the device if it reads disconnected, otherwise disconnect it."
    }
    return """
    Usage:
      \(TOOL_NAME) \(verb) <target> [-t <seconds>] [-j] [-v]

    \(what)
    The state is read back until it changes; if it doesn't within the timeout: exit 4.
    BLE devices reconnect on their own when woken; connect pages Classic devices and may fail for BLE.

    Options:
      -t, --timeout <seconds>    Read-back limit (default 10)
      -j, --json                 {"success":true,"action":"\(verb == "toggle" ? "disconnect" : verb)","changed":true,
                                  "was":"...","now":"...","device":Device,"ms":412}
    \(commonOpts)

    \(verb == "toggle" ? "'action' is what toggle did: connect or disconnect.\n\n" : "")\(targetHelp)
    """
}

let helpTexts: [String: String] = [
    "": """
    \(TOOL_NAME) \(VERSION) - native macOS Bluetooth CLI (IOBluetooth). Alias: bluetooth

    Usage:
      \(TOOL_NAME) <command> [options]

    Commands:
      list                       Paired devices with stable 1-based indices
      status                     Power and connected devices
      power [on|off|toggle]      Print power, or set it and read it back
      connect <target>           Connect a paired device
      disconnect <target>        Disconnect a device
      toggle <target>            Connect if disconnected, else disconnect
      wait <target> <state>      Block until connected|disconnected (-t seconds)
      watch [target]             One line per connect/disconnect until Ctrl-C
      logs                       Unified log: bluetoothd + com.apple.bluetooth
      alias <list|set|rm>        Name devices (~/.config/macos-gui/bluetooth_aliases.json)
      help [command]             Help for a command (also: <command> -h)

    Options (every command):
      -j, --json                 JSON on stdout (watch, logs -f: JSON Lines)
      -v, --verbose              API calls, return codes and timings on stderr
      -h, --help                 Show help
      -V, --version              Show version

    \(targetHelp)

    Exit codes:
      0 ok   1 failed   2 usage   3 refused by a safety guard   4 timed out / unconfirmed

    Examples:
      \(TOOL_NAME) list
      \(TOOL_NAME) list -j | jq '.devices[] | select(.connected)'
      \(TOOL_NAME) alias set receiver dh590
      \(TOOL_NAME) disconnect receiver
      \(TOOL_NAME) wait "MAGIC KEEB" connected -t 30
      \(TOOL_NAME) power off                # refuses while a BT keyboard/mouse/trackpad is connected
      \(TOOL_NAME) watch -j
      \(TOOL_NAME) logs --last 30m --grep disconnect
    """,
    "list": """
    Usage:
      \(TOOL_NAME) list [-j] [-v]

    Paired devices sorted by name, then address. The 1-based index is a valid target
    for every other command; it shifts when devices pair/unpair, so script with the
    address or an alias. kind comes from the class of device; BLE devices report none,
    so theirs comes from macOS's own record (system_profiler), else Unknown.

    Options:
      -j, --json                 {"power":"on","devices":[Device...]}
    \(commonOpts)

    \(deviceHelp)
    """,
    "status": """
    Usage:
      \(TOOL_NAME) status [-j] [-v]

    Controller power and the connected devices. Human output also says whether
    'power off' would be refused right now, and why.

    Options:
      -j, --json                 {"power":"on","connected":[Device...]}
    \(commonOpts)

    \(deviceHelp)
    """,
    "power": """
    Usage:
      \(TOOL_NAME) power [on|off|toggle] [--force] [-t <seconds>] [-j] [-v]

    No argument: print the controller power (on|off). With one: set it and read it back.
    Already in that state: nothing is sent (changed:false). 'off', and 'toggle' while on,
    refuses with exit 3 while a Bluetooth keyboard, mouse or trackpad is connected
    (IOBluetooth plus the IOKit HID registry), naming it.

    Options:
      --force                    Turn off even if that cuts off an input device
      -t, --timeout <seconds>    Read-back limit (default 10); no read-back: exit 4
      -j, --json                 {"success":true,"was":"on","now":"off","changed":true}
                                 no argument: {"power":"on"}
    \(commonOpts)

    Uses the private IOBluetoothPreferenceSetControllerPowerState (as blueutil does).
    If macOS ignores it, the read-back fails and this exits 4; it never reports success
    without reading the new state.
    """,
    "connect": connectionHelp("connect"),
    "disconnect": connectionHelp("disconnect"),
    "toggle": connectionHelp("toggle"),
    "wait": """
    Usage:
      \(TOOL_NAME) wait <target> <connected|disconnected> [-t <seconds>] [-j] [-v]

    Block until the device reads the given state; returns at once if it already does.

    Options:
      -t, --timeout <seconds>    Give up after this long (default 10): exit 4
      -j, --json                 {"success":true,"state":"connected","device":Device,"waited_ms":1200}
    \(commonOpts)

    \(targetHelp)
    """,
    "watch": """
    Usage:
      \(TOOL_NAME) watch [target] [-j] [-v]

    Print one line per connect/disconnect, for every paired device or just <target>,
    until Ctrl-C (exit 0). Driven by IOBluetooth connect/disconnect notifications with
    a 1 s poll as a backstop; a line prints only when a device's read state changes.

    Options:
      -j, --json                 JSON Lines: {"at":"<RFC3339 UTC>","event":"connected","device":Device}
    \(commonOpts)

    \(targetHelp)
    """,
    "logs": """
    Usage:
      \(TOOL_NAME) logs [--last <window>] [--grep <text>] [-n <count>] [-j] [-v]
      \(TOOL_NAME) logs -f [--grep <text>] [-j] [-v]

    Unified log for process bluetoothd and subsystem com.apple.bluetooth, oldest first.
    \(TOOL_NAME)'s own IOBluetooth chatter is left out.

    Options:
      --last <window>            <n>[m|h|d] or 'boot' (default 2h)
      -f, --follow               Stream new entries until Ctrl-C (not with --last/--limit)
      --grep <text>              Only messages containing text (case-insensitive)
      -n, --limit <count>        Newest N entries (default 200; 0 = all)
      -j, --json                 {"entries":[{"time":"<RFC3339 UTC>","process":"bluetoothd","message":"..."}]}
                                 --follow: one entry object per line
    \(commonOpts)
    """,
    "alias": """
    Usage:
      \(TOOL_NAME) alias [list]
      \(TOOL_NAME) alias set <alias> <target>
      \(TOOL_NAME) alias rm <alias>

    Aliases point at a device ADDRESS in ~/.config/macos-gui/bluetooth_aliases.json
    ({"receiver": "94:08:53:f1:06:55"}), so they survive renames and index shifts.
    Names: letters, digits, '.', '_', '-'; case-insensitive; not all digits, not an address.

    Commands:
      list                       Aliases and the device each resolves to (default)
      set <alias> <target>       Point alias at target's address, then read it back
      rm <alias>                 Remove alias (also: remove); absent: changed:false, exit 0

    Options:
      -j, --json                 see '\(TOOL_NAME) alias <list|set|rm> --help'
    \(commonOpts)
    """,
    "alias list": """
    Usage:
      \(TOOL_NAME) alias [list] [-j] [-v]

    Every alias, its stored target and the paired device it resolves to.

    Options:
      -j, --json                 {"aliases":[{"alias":"receiver","target":"94:08:53:f1:06:55","device":Device|null}]}
    \(commonOpts)
    """,
    "alias set": """
    Usage:
      \(TOOL_NAME) alias set <alias> <target> [-j] [-v]

    Resolve <target> now and store its address under <alias>, then read the file back.
    Already pointing there: nothing written (changed:false).

    Options:
      -j, --json                 {"success":true,"changed":true,"alias":"receiver","was":null,
                                  "now":"94:08:53:f1:06:55","device":Device}
    \(commonOpts)

    \(targetHelp)
    """,
    "alias rm": """
    Usage:
      \(TOOL_NAME) alias rm <alias> [-j] [-v]

    Remove an alias and read the file back. Absent: nothing written (changed:false, exit 0).
    'remove' is accepted for 'rm'.

    Options:
      -j, --json                 {"success":true,"changed":true,"alias":"receiver","was":"94:08:53:f1:06:55","now":null}
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
let commandOpts: [String: [Opt]] = [
    "list": [], "status": [], "watch": [], "alias": [], "help": [], "version": [],
    "power": [Opt(long: "force", short: nil, takesValue: false), timeoutOpt],
    "connect": [timeoutOpt], "disconnect": [timeoutOpt], "toggle": [timeoutOpt], "wait": [timeoutOpt],
    "logs": [Opt(long: "last", short: nil, takesValue: true), Opt(long: "follow", short: "f", takesValue: false),
             Opt(long: "grep", short: nil, takesValue: true), Opt(long: "limit", short: "n", takesValue: true)],
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
    setvbuf(stdout, nil, _IOLBF, 0)   // line-buffered even when piped: watch/logs -f stream live
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
    let subs: Set<String> = ["list", "set", "rm", "remove"]
    func aliasTopic(_ s: String?) -> String {
        guard let s = s?.lowercased(), subs.contains(s) else { return "alias" }
        return "alias " + (s == "remove" ? "rm" : s)
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
    case "power": cmdPower(a)
    case "connect", "disconnect", "toggle": cmdConnection(cmd, a)
    case "wait": cmdWait(a)
    case "watch": cmdWatch(a)
    case "logs": cmdLogs(a)
    case "alias": cmdAlias(a)
    default: usageError("unknown command '\(cmd)'")
    }
    exit(ExitCode.ok)
}

main()
