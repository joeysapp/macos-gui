import Foundation
import AppKit
import CoreGraphics
import ColorSync
import ImageIO
import UniformTypeIdentifiers

let VERSION = "0.3.0"
let TOOL_NAME = "background-tool"

// MARK: - Enums & Types

enum WallpaperMode: String, CaseIterable, Codable {
    case fill = "fill"
    case fit = "fit"
    case stretch = "stretch"
    case center = "center"

    var description: String {
        switch self {
        case .fill: return "Fill screen (crop/zoom to cover without distortion)"
        case .fit: return "Fit to screen (preserve aspect ratio with letterboxing/borders)"
        case .stretch: return "Stretch to screen (scale axes independently)"
        case .center: return "Center original size (1:1 pixels, bordered by background color)"
        }
    }

    var scaling: NSImageScaling {
        switch self {
        case .fill: return .scaleProportionallyUpOrDown
        case .fit: return .scaleProportionallyUpOrDown
        case .stretch: return .scaleAxesIndependently
        case .center: return .scaleNone
        }
    }

    var allowClipping: Bool {
        switch self {
        case .fill: return true
        case .fit: return false
        case .stretch: return true
        case .center: return false
        }
    }

    static func from(options: [NSWorkspace.DesktopImageOptionKey: Any]) -> WallpaperMode {
        let scalingRaw = (options[.imageScaling] as? NSNumber)?.uintValue ?? NSImageScaling.scaleProportionallyUpOrDown.rawValue
        let allowClip = (options[.allowClipping] as? NSNumber)?.boolValue ?? true

        if scalingRaw == NSImageScaling.scaleAxesIndependently.rawValue {
            return .stretch
        } else if scalingRaw == NSImageScaling.scaleNone.rawValue {
            return .center
        } else if !allowClip {
            return .fit
        } else {
            return .fill
        }
    }
}

enum DynamicScheduleMode: String, CaseIterable {
    case solar = "solar"
    case time = "time"
    case appearance = "appearance"

    static func parse(_ str: String) -> DynamicScheduleMode? {
        switch str.lowercased() {
        case "solar", "sun": return .solar
        case "time", "h24", "clock", "24h": return .time
        case "appearance", "apr", "light-dark", "lightdark", "dark-light": return .appearance
        default: return nil
        }
    }
}

// MARK: - Color Parsing

func parseColor(_ input: String) -> NSColor? {
    var str = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if str.hasPrefix("#") {
        str.removeFirst()
    }

    switch str {
    case "black": return .black
    case "white": return .white
    case "gray", "grey": return .gray
    case "darkgray", "darkgrey": return .darkGray
    case "lightgray", "lightgrey": return .lightGray
    case "red": return .red
    case "green": return .green
    case "blue": return .blue
    case "cyan": return .cyan
    case "magenta": return .magenta
    case "yellow": return .yellow
    case "orange": return .orange
    case "purple": return .purple
    case "brown": return .brown
    case "clear": return .clear
    default: break
    }

    if str.count == 3 {
        let r = str[str.startIndex]
        let g = str[str.index(str.startIndex, offsetBy: 1)]
        let b = str[str.index(str.startIndex, offsetBy: 2)]
        str = "\(r)\(r)\(g)\(g)\(b)\(b)"
    }

    guard let hexVal = UInt64(str, radix: 16) else { return nil }

    if str.count == 6 {
        let r = CGFloat((hexVal >> 16) & 0xFF) / 255.0
        let g = CGFloat((hexVal >> 8) & 0xFF) / 255.0
        let b = CGFloat(hexVal & 0xFF) / 255.0
        return NSColor(srgbRed: r, green: g, blue: b, alpha: 1.0)
    } else if str.count == 8 {
        let r = CGFloat((hexVal >> 24) & 0xFF) / 255.0
        let g = CGFloat((hexVal >> 16) & 0xFF) / 255.0
        let b = CGFloat((hexVal >> 8) & 0xFF) / 255.0
        let a = CGFloat(hexVal & 0xFF) / 255.0
        return NSColor(srgbRed: r, green: g, blue: b, alpha: a)
    }
    return nil
}

func colorToHex(_ color: NSColor) -> String {
    guard let rgb = color.usingColorSpace(.sRGB) else {
        return "custom"
    }
    let r = Int(round(rgb.redComponent * 255.0))
    let g = Int(round(rgb.greenComponent * 255.0))
    let b = Int(round(rgb.blueComponent * 255.0))
    return String(format: "#%02X%02X%02X", r, g, b)
}

func generateSolidColorImage(color: NSColor, hexName: String) -> URL? {
    let size = NSSize(width: 64, height: 64)
    let image = NSImage(size: size)
    image.lockFocus()
    color.drawSwatch(in: NSRect(origin: .zero, size: size))
    image.unlockFocus()

    guard let tiffData = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiffData),
          let pngData = bitmap.representation(using: .png, properties: [:]) else {
        return nil
    }

    let cacheDir = FileManager.default.temporaryDirectory.appendingPathComponent("macos-gui-wallpapers")
    try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    let cleanName = hexName.replacingOccurrences(of: "#", with: "")
    let fileURL = cacheDir.appendingPathComponent("solid_\(cleanName).png")
    do {
        try pngData.write(to: fileURL)
        return fileURL
    } catch {
        return nil
    }
}

// MARK: - Display & Screen Info

struct ScreenInfo {
    let index: Int
    let screen: NSScreen
    let id: UInt32
    let name: String
    let uuid: String
    let resolution: String

    init(index: Int, screen: NSScreen) {
        self.index = index
        self.screen = screen
        let desc = screen.deviceDescription
        let num = (desc[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value ?? UInt32(index + 1)
        self.id = num
        self.name = screen.localizedName
        self.resolution = "\(Int(screen.frame.width))x\(Int(screen.frame.height))"

        var u = "unknown"
        if let cfUuid = CGDisplayCreateUUIDFromDisplayID(num) {
            u = CFUUIDCreateString(nil, cfUuid.takeRetainedValue()) as String
        }
        self.uuid = u
    }
}

func getAllScreens() -> [ScreenInfo] {
    return NSScreen.screens.enumerated().map { ScreenInfo(index: $0.offset, screen: $0.element) }
}

func resolveTargetScreens(target: String?, screens: [ScreenInfo], asJson: Bool) -> [ScreenInfo] {
    guard let target = target?.trimmingCharacters(in: .whitespacesAndNewlines), !target.isEmpty else {
        return screens
    }

    if target.lowercased() == "all" {
        return screens
    }

    let lower = target.lowercased()

    // 1. Match by full or prefix UUID
    let uuidMatches = screens.filter { $0.uuid.lowercased().hasPrefix(lower) || $0.uuid.lowercased() == lower }
    if !uuidMatches.isEmpty {
        return uuidMatches
    }

    // 2. Match by 0-based index or display ID
    if let intVal = UInt32(target) {
        if let match = screens.first(where: { UInt32($0.index) == intVal || $0.id == intVal }) {
            return [match]
        }
    }

    // 3. Match by localized display name substring (e.g. "built-in", "retina", "lg")
    let nameMatches = screens.filter { $0.name.lowercased().contains(lower) }
    if !nameMatches.isEmpty {
        return nameMatches
    }

    if asJson {
        let errDict: [String: Any] = [
            "error": "No display found matching '\(target)'",
            "code": 1,
            "success": false,
            "available_displays": screens.map { [
                "index": $0.index,
                "id": $0.id,
                "name": $0.name,
                "uuid": $0.uuid,
                "resolution": $0.resolution
            ] }
        ]
        if let data = try? JSONSerialization.data(withJSONObject: errDict, options: [.prettyPrinted, .sortedKeys]),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
        exit(1)
    }

    fputs("error: No display found matching '\(target)'. Available displays:\n", stderr)
    for s in screens {
        fputs("  [\(s.index)] ID: \(s.id) | UUID: \(s.uuid) | \"\(s.name)\" (\(s.resolution))\n", stderr)
    }
    exit(1)
}

// MARK: - Interval Parsing

func parseInterval(_ str: String) -> TimeInterval? {
    let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !trimmed.isEmpty else { return nil }

    let scanner = Scanner(string: trimmed)
    guard let number = scanner.scanDouble() else { return nil }

    let unit = String(trimmed[scanner.currentIndex...]).trimmingCharacters(in: .whitespaces)
    switch unit {
    case "", "s", "sec", "seconds":
        return number
    case "m", "min", "minutes":
        return number * 60.0
    case "h", "hr", "hours":
        return number * 3600.0
    case "d", "day", "days":
        return number * 86400.0
    default:
        return nil
    }
}

// MARK: - Supported File Finding

let supportedExtensions: Set<String> = ["jpg", "jpeg", "png", "heic", "webp", "tiff", "tif", "bmp"]

func findImages(in directory: URL) -> [URL] {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]) else {
        return []
    }

    var results: [URL] = []
    for case let fileURL as URL in enumerator {
        let ext = fileURL.pathExtension.lowercased()
        if supportedExtensions.contains(ext) {
            results.append(fileURL)
        }
    }
    return results.sorted {
        $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending
    }
}

// MARK: - Output & JSON Structs

struct ScreenJSON: Codable {
    let index: Int
    let id: UInt32
    let name: String
    let uuid: String
    let resolution: String
    let wallpaper: String
    let mode: String
    let fillColor: String
}

func exitJSONError(message: String, code: Int32) -> Never {
    let dict: [String: Any] = [
        "error": message,
        "code": code,
        "success": false
    ]
    if let data = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]),
       let str = String(data: data, encoding: .utf8) {
        print(str)
    }
    exit(code)
}

func exitError(message: String, code: Int32, asJson: Bool) -> Never {
    if asJson {
        exitJSONError(message: message, code: code)
    } else {
        fputs("error: \(message)\n", stderr)
        exit(code)
    }
}

// MARK: - Read-Back Wallpaper Application

func applyWallpaper(imageURL: URL,
                    targets: [ScreenInfo],
                    mode: WallpaperMode?,
                    fillColor: NSColor?,
                    dryRun: Bool,
                    asJson: Bool,
                    quiet: Bool,
                    verbose: Bool) {
    let workspace = NSWorkspace.shared
    let targetPath = imageURL.standardizedFileURL.path

    var results: [[String: Any]] = []
    var allConfirmed = true

    for screenInfo in targets {
        var options: [NSWorkspace.DesktopImageOptionKey: Any] = workspace.desktopImageOptions(for: screenInfo.screen) ?? [:]

        if let mode = mode {
            options[.imageScaling] = mode.scaling.rawValue
            options[.allowClipping] = mode.allowClipping
        }
        if let fillColor = fillColor {
            options[.fillColor] = fillColor
        }

        let modeStr = (mode ?? WallpaperMode.from(options: options)).rawValue
        let colorStr = (fillColor != nil) ? colorToHex(fillColor!) : (options[.fillColor] as? NSColor).map(colorToHex) ?? "default"

        if verbose {
            fputs("[debug] Applying '\(targetPath)' to display [\(screenInfo.index)] UUID: \(screenInfo.uuid)\n", stderr)
        }

        if !asJson && !quiet {
            print("🖥  Display [\(screenInfo.index)] (ID: \(screenInfo.id) | UUID: \(screenInfo.uuid) | \"\(screenInfo.name)\")")
            print("   🖼  Image: \(targetPath)")
            print("   📐  Mode:  \(modeStr)")
            print("   🎨  Color: \(colorStr)")
        }

        if dryRun {
            if !asJson && !quiet {
                print("   [dry-run] skipped actual system call")
            }
            results.append([
                "display": screenInfo.index,
                "uuid": screenInfo.uuid,
                "applied": targetPath,
                "dry_run": true
            ])
            continue
        }

        do {
            try workspace.setDesktopImageURL(imageURL, for: screenInfo.screen, options: options)
        } catch {
            if asJson {
                exitJSONError(message: "Error setting wallpaper on display [\(screenInfo.index)]: \(error.localizedDescription)", code: 1)
            } else {
                fputs("   ❌ Error setting wallpaper on display [\(screenInfo.index)]: \(error.localizedDescription)\n", stderr)
                exit(1)
            }
        }

        // Read-back transaction verification loop (up to 2 seconds, checking every 100ms)
        var confirmed = false
        var currentReadBack = ""
        for _ in 0..<20 {
            Thread.sleep(forTimeInterval: 0.1)
            if let cur = workspace.desktopImageURL(for: screenInfo.screen)?.standardizedFileURL.path {
                currentReadBack = cur
                if cur == targetPath {
                    confirmed = true
                    break
                }
            }
        }

        if confirmed {
            if !asJson && !quiet {
                print("   ✅ Applied and verified successfully")
            }
            results.append([
                "display": screenInfo.index,
                "uuid": screenInfo.uuid,
                "wallpaper": targetPath,
                "confirmed": true
            ])
        } else {
            allConfirmed = false
            if !asJson && !quiet {
                fputs("   ⚠️ Read-back verification timed out (read '\(currentReadBack)')\n", stderr)
            }
            results.append([
                "display": screenInfo.index,
                "uuid": screenInfo.uuid,
                "wallpaper": currentReadBack,
                "confirmed": false
            ])
        }
    }

    if asJson {
        let output: [String: Any] = [
            "success": allConfirmed,
            "results": results
        ]
        if let data = try? JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys]),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }

    if !dryRun && !allConfirmed {
        exit(4) // Code 4: Unconfirmed / Timed Out
    }
}

// MARK: - Dynamic HEIC Builder

func buildDynamicHEIC(images: [URL],
                      scheduleMode: DynamicScheduleMode,
                      outputURL: URL,
                      lightOverride: String?,
                      darkOverride: String?,
                      verbose: Bool) throws {
    guard images.count >= 2 else {
        throw NSError(domain: TOOL_NAME, code: 2, userInfo: [NSLocalizedDescriptionKey: "Dynamic wallpaper requires at least 2 images"])
    }

    if verbose {
        fputs("[debug] Ingesting \(images.count) frames for schedule '\(scheduleMode.rawValue)'...\n", stderr)
    }

    // Load source images
    var cgImages: [CGImage] = []
    for imgURL in images {
        guard let isrc = CGImageSourceCreateWithURL(imgURL as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(isrc, 0, nil) else {
            throw NSError(domain: TOOL_NAME, code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to decode image at \(imgURL.path)"])
        }
        cgImages.append(cg)
    }

    // Resolve light and dark indices
    var lightIdx = 0
    var darkIdx = min(1, images.count - 1)

    if let lStr = lightOverride {
        if let intVal = Int(lStr), intVal >= 0 && intVal < images.count {
            lightIdx = intVal
        } else if let matchIdx = images.firstIndex(where: { $0.lastPathComponent.lowercased().contains(lStr.lowercased()) }) {
            lightIdx = matchIdx
        }
    } else {
        // Auto-detect light/day
        if let match = images.firstIndex(where: {
            let n = $0.lastPathComponent.lowercased()
            return n.contains("light") || n.contains("day") || n.contains("noon")
        }) {
            lightIdx = match
        } else if scheduleMode == .solar {
            // Noon frame is roughly in the middle
            lightIdx = images.count / 2
        }
    }

    if let dStr = darkOverride {
        if let intVal = Int(dStr), intVal >= 0 && intVal < images.count {
            darkIdx = intVal
        } else if let matchIdx = images.firstIndex(where: { $0.lastPathComponent.lowercased().contains(dStr.lowercased()) }) {
            darkIdx = matchIdx
        }
    } else {
        // Auto-detect dark/night
        if let match = images.firstIndex(where: {
            let n = $0.lastPathComponent.lowercased()
            return n.contains("dark") || n.contains("night") || n.contains("dusk")
        }) {
            darkIdx = match
        } else if scheduleMode == .solar {
            // Midnight is at the beginning or end
            darkIdx = 0
        }
    }

    let tagName: String
    let plistDict: [String: Any]

    switch scheduleMode {
    case .appearance:
        tagName = "apr"
        plistDict = [
            "l": lightIdx,
            "d": darkIdx
        ]

    case .solar:
        tagName = "solar"
        let count = images.count
        var solarEntries: [[String: Any]] = []
        for i in 0..<count {
            // Hour from 0.0 to 24.0
            let hour = (Double(i) / Double(count)) * 24.0
            // Solar altitude in degrees [-60, +60]
            let alt = 60.0 * sin((hour - 6.0) / 24.0 * 2.0 * .pi)
            // Azimuth in degrees [0, 360]
            let az = (hour / 24.0) * 360.0
            solarEntries.append([
                "a": alt,
                "z": az,
                "i": i
            ])
        }
        plistDict = [
            "ap": [
                "l": lightIdx,
                "d": darkIdx
            ],
            "si": solarEntries
        ]

    case .time:
        tagName = "h24"
        let count = images.count
        var timeEntries: [[String: Any]] = []

        // Check if filenames contain timestamps (e.g. 12_00 or 12.jpg)
        var parsedTimes: [Double] = []
        for (i, imgURL) in images.enumerated() {
            let name = imgURL.deletingPathExtension().lastPathComponent
            var fraction: Double? = nil
            let parts = name.components(separatedBy: CharacterSet(charactersIn: "_-.: "))
            if parts.count >= 2, let h = Double(parts[0]), let m = Double(parts[1]), h >= 0 && h < 24 && m >= 0 && m < 60 {
                fraction = (h * 60.0 + m) / 1440.0
            } else if let num = Double(parts[0]), num >= 0 && num <= 24 {
                fraction = num / 24.0
            }
            parsedTimes.append(fraction ?? (Double(i) / Double(count)))
        }

        // Ensure 0.0 is present
        if !parsedTimes.contains(0.0) {
            timeEntries.append(["t": 0.0, "i": 0])
        }
        for (i, t) in parsedTimes.enumerated() {
            timeEntries.append(["t": t, "i": i])
        }
        timeEntries.sort { ($0["t"] as? Double ?? 0) < ($1["t"] as? Double ?? 0) }

        plistDict = [
            "ap": [
                "l": lightIdx,
                "d": darkIdx
            ],
            "ti": timeEntries
        ]
    }

    let plistData = try PropertyListSerialization.data(fromPropertyList: plistDict, format: .binary, options: 0)
    let tagBase64 = plistData.base64EncodedString()

    try? FileManager.default.removeItem(at: outputURL)
    guard let dest = CGImageDestinationCreateWithURL(outputURL as CFURL, UTType.heic.identifier as CFString, cgImages.count, nil) else {
        throw NSError(domain: TOOL_NAME, code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create HEIC destination at \(outputURL.path)"])
    }

    let metadata = CGImageMetadataCreateMutable()
    CGImageMetadataRegisterNamespaceForPrefix(
        metadata,
        "http://ns.apple.com/namespace/1.0/" as CFString,
        "apple_desktop" as CFString,
        nil
    )
    guard let tag = CGImageMetadataTagCreate(
        "http://ns.apple.com/namespace/1.0/" as CFString,
        "apple_desktop" as CFString,
        tagName as CFString,
        .string,
        tagBase64 as CFString
    ) else {
        throw NSError(domain: TOOL_NAME, code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create apple_desktop:\(tagName) metadata tag"])
    }
    CGImageMetadataSetTagWithPath(metadata, nil, "apple_desktop:\(tagName)" as CFString, tag)

    CGImageDestinationAddImageAndMetadata(dest, cgImages[0], metadata, nil)
    for cg in cgImages.dropFirst() {
        CGImageDestinationAddImage(dest, cg, nil)
    }

    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: TOOL_NAME, code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to finalize dynamic HEIC file"])
    }
}

// MARK: - Subcommands

func cmdList(asJson: Bool) {
    let screens = getAllScreens()
    let workspace = NSWorkspace.shared

    if asJson {
        let items = screens.map { s -> ScreenJSON in
            let curURL = workspace.desktopImageURL(for: s.screen)?.path ?? "none"
            let opts = workspace.desktopImageOptions(for: s.screen) ?? [:]
            let mode = WallpaperMode.from(options: opts).rawValue
            let colorStr = (opts[.fillColor] as? NSColor).map(colorToHex) ?? "none"
            return ScreenJSON(
                index: s.index,
                id: s.id,
                name: s.name,
                uuid: s.uuid,
                resolution: s.resolution,
                wallpaper: curURL,
                mode: mode,
                fillColor: colorStr
            )
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(items), let str = String(data: data, encoding: .utf8) {
            print(str)
        }
        return
    }

    print("Found \(screens.count) connected display(s):\n")

    for s in screens {
        let curURL = workspace.desktopImageURL(for: s.screen)?.path ?? "none"
        let opts = workspace.desktopImageOptions(for: s.screen) ?? [:]
        let mode = WallpaperMode.from(options: opts).rawValue
        let colorStr = (opts[.fillColor] as? NSColor).map(colorToHex) ?? "none"

        print("  Display [\(s.index)]")
        print("    ID:          \(s.id)")
        print("    UUID:        \(s.uuid)")
        print("    Name:        \(s.name)")
        print("    Resolution:  \(s.resolution)")
        print("    Wallpaper:   \(curURL)")
        print("    Mode:        \(mode)")
        print("    Fill Color:  \(colorStr)")
        print("")
    }
}

func cmdGet(targets: [ScreenInfo], asJson: Bool) {
    let workspace = NSWorkspace.shared
    if asJson {
        var dict: [String: Any] = [:]
        for s in targets {
            let curURL = workspace.desktopImageURL(for: s.screen)?.path ?? "none"
            dict["display_\(s.index)"] = [
                "id": s.id,
                "uuid": s.uuid,
                "name": s.name,
                "wallpaper": curURL
            ]
        }
        if let data = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
        return
    }

    for s in targets {
        let curURL = workspace.desktopImageURL(for: s.screen)?.path ?? "none"
        if targets.count == 1 {
            print(curURL)
        } else {
            print("[\(s.index)] \(s.name) (\(s.uuid)): \(curURL)")
        }
    }
}

// MARK: - Help

func printUsage() {
    print("""
Usage:
  \(TOOL_NAME) <image> [options]
  \(TOOL_NAME) set <image> [options]
  \(TOOL_NAME) dynamic <directory|images...> [options]
  \(TOOL_NAME) color <hex|name> [options]
  \(TOOL_NAME) random <directory> [options]
  \(TOOL_NAME) rotate <directory> [interval] [options]
  \(TOOL_NAME) spaces <directory|images...> [options]
  \(TOOL_NAME) list [options]
  \(TOOL_NAME) get [options]

Commands:
  set      Set static or dynamic wallpaper on target displays
  dynamic  Ingest frame directory or files and generate dynamic HEIC wallpaper
  color    Set solid background color on target displays
  random   Pick and apply random wallpaper from directory
  rotate   Run background rotation loop changing wallpapers at interval
  spaces   Assign different sequential wallpapers across multiple displays
  list     List connected displays, IDs, UUIDs, resolutions, and current wallpapers
  get      Print current wallpaper path for target displays

Options:
  -d, --display <target>   Target display by index (0, 1), ID, name substring, or UUID
  -m, --mode <mode>        Scaling mode: fill, fit, stretch, center [default: fill]
  -c, --color <hex|name>   Fill / matte border color (e.g. '#1e1e1e', 'black')
  -n, --dry-run            Simulate operations without making system calls
  -j, --json               Output machine-readable JSON format
  -v, --verbose            Emit detailed debug logs and timings to stderr
  -q, --quiet              Suppress normal output
  -V, --version            Show version
  -h, --help               Show help

Dynamic Options:
  -s, --schedule <mode>    Schedule type: solar, time, appearance [default: auto]
  -o, --out <path>         Output path for generated .heic file
  --apply                  Apply generated dynamic wallpaper to target display(s)
  --light <index|name>     Override image index or filename for light appearance
  --dark <index|name>      Override image index or filename for dark appearance

Exit Codes:
  0  Success / verified
  1  Runtime failure
  2  Usage error
  4  Unconfirmed / read-back timed out

Examples:
  \(TOOL_NAME) list -j
  \(TOOL_NAME) set ~/Pictures/wallpaper.heic -d 37D8832A-2D66-02CA-B9F7-8F30A301B230
  \(TOOL_NAME) dynamic ~/Pictures/SolarFrames/ --schedule solar --apply -d 0
  \(TOOL_NAME) dynamic day.jpg night.jpg --schedule appearance --out ~/Pictures/day_night.heic
""")
}

// MARK: - Main Execution

func main() {
    let args = Array(CommandLine.arguments.dropFirst())

    if args.isEmpty {
        printUsage()
        exit(2)
    }

    let asJson = args.contains("-j") || args.contains("--json")
    let verbose = args.contains("-v") || args.contains("--verbose")
    let quiet = args.contains("-q") || args.contains("--quiet")
    let dryRun = args.contains("-n") || args.contains("--dry-run")

    var targetDisplay: String? = nil
    var mode: WallpaperMode? = nil
    var color: NSColor? = nil
    var rawColorString: String? = nil
    var dynamicSchedule: DynamicScheduleMode? = nil
    var dynamicOut: String? = nil
    var dynamicApply = false
    var lightOverride: String? = nil
    var darkOverride: String? = nil
    var positional: [String] = []

    var i = 0
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "-j", "--json", "-v", "--verbose", "-q", "--quiet", "-n", "--dry-run":
            break
        case "-V", "--version", "version":
            print("\(TOOL_NAME) \(VERSION)")
            exit(0)
        case "-h", "--help":
            printUsage()
            exit(0)
        case "-d", "--display":
            i += 1
            guard i < args.count else {
                exitError(message: "--display requires a target argument (index, ID, name, or UUID)", code: 2, asJson: asJson)
            }
            targetDisplay = args[i]
        case "-m", "--mode":
            i += 1
            guard i < args.count else {
                exitError(message: "--mode requires an argument (fill, fit, stretch, center)", code: 2, asJson: asJson)
            }
            guard let parsedMode = WallpaperMode(rawValue: args[i].lowercased()) else {
                exitError(message: "invalid mode '\(args[i])'. Valid modes: fill, fit, stretch, center", code: 2, asJson: asJson)
            }
            mode = parsedMode
        case "-c", "--color":
            i += 1
            guard i < args.count else {
                exitError(message: "--color requires a hex or color name", code: 2, asJson: asJson)
            }
            rawColorString = args[i]
            guard let parsedColor = parseColor(args[i]) else {
                exitError(message: "invalid color '\(args[i])'. Try hex (e.g. '#1e1e1e') or color name (e.g. 'black')", code: 2, asJson: asJson)
            }
            color = parsedColor
        case "-s", "--schedule":
            i += 1
            guard i < args.count else {
                exitError(message: "--schedule requires a mode (solar, time, appearance)", code: 2, asJson: asJson)
            }
            guard let parsedSchedule = DynamicScheduleMode.parse(args[i]) else {
                exitError(message: "unknown schedule mode '\(args[i])'. Use solar, time, or appearance", code: 2, asJson: asJson)
            }
            dynamicSchedule = parsedSchedule
        case "-o", "--out":
            i += 1
            guard i < args.count else {
                exitError(message: "--out requires a destination file path", code: 2, asJson: asJson)
            }
            dynamicOut = args[i]
        case "--apply":
            dynamicApply = true
        case "--light":
            i += 1
            guard i < args.count else {
                exitError(message: "--light requires an index or filename", code: 2, asJson: asJson)
            }
            lightOverride = args[i]
        case "--dark":
            i += 1
            guard i < args.count else {
                exitError(message: "--dark requires an index or filename", code: 2, asJson: asJson)
            }
            darkOverride = args[i]
        default:
            if arg.hasPrefix("-") {
                exitError(message: "unknown option '\(arg)'. See --help for available options.", code: 2, asJson: asJson)
            } else {
                positional.append(arg)
            }
        }
        i += 1
    }

    let allScreens = getAllScreens()
    let targets = resolveTargetScreens(target: targetDisplay, screens: allScreens, asJson: asJson)

    guard !positional.isEmpty else {
        if asJson {
            cmdList(asJson: true)
            return
        }
        if let col = color, let rawCol = rawColorString {
            guard let imgURL = generateSolidColorImage(color: col, hexName: rawCol) else {
                exitError(message: "failed to create solid color image", code: 1, asJson: asJson)
            }
            applyWallpaper(imageURL: imgURL, targets: targets, mode: mode ?? .fill, fillColor: col, dryRun: dryRun, asJson: asJson, quiet: quiet, verbose: verbose)
            return
        }
        printUsage()
        exit(2)
    }

    let command = positional[0]
    let rest = Array(positional.dropFirst())

    switch command.lowercased() {
    case "list", "status":
        cmdList(asJson: asJson)

    case "get":
        cmdGet(targets: targets, asJson: asJson)

    case "color":
        guard !rest.isEmpty || rawColorString != nil else {
            exitError(message: "'color' command requires a color hex or name", code: 2, asJson: asJson)
        }
        let colStr = rest.first ?? rawColorString!
        guard let col = parseColor(colStr) else {
            exitError(message: "invalid color '\(colStr)'", code: 2, asJson: asJson)
        }
        guard let imgURL = generateSolidColorImage(color: col, hexName: colStr) else {
            exitError(message: "failed to create solid color image", code: 1, asJson: asJson)
        }
        applyWallpaper(imageURL: imgURL, targets: targets, mode: mode ?? .fill, fillColor: col, dryRun: dryRun, asJson: asJson, quiet: quiet, verbose: verbose)

    case "set":
        guard let path = rest.first else {
            exitError(message: "'set' requires an image path", code: 2, asJson: asJson)
        }
        let imgURL = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: imgURL.path) else {
            exitError(message: "file not found: \(imgURL.path)", code: 1, asJson: asJson)
        }
        applyWallpaper(imageURL: imgURL, targets: targets, mode: mode, fillColor: color, dryRun: dryRun, asJson: asJson, quiet: quiet, verbose: verbose)

    case "dynamic":
        guard !rest.isEmpty else {
            exitError(message: "'dynamic' requires a directory or list of images. See --help.", code: 2, asJson: asJson)
        }

        var inputImages: [URL] = []
        let firstArgURL = URL(fileURLWithPath: (rest[0] as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false

        if FileManager.default.fileExists(atPath: firstArgURL.path, isDirectory: &isDir), isDir.boolValue {
            inputImages = findImages(in: firstArgURL)
            if inputImages.isEmpty {
                exitError(message: "no supported images found in directory: \(firstArgURL.path)", code: 1, asJson: asJson)
            }
        } else {
            inputImages = rest.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            for u in inputImages {
                guard FileManager.default.fileExists(atPath: u.path) else {
                    exitError(message: "image file not found: \(u.path)", code: 1, asJson: asJson)
                }
            }
        }

        guard inputImages.count >= 2 else {
            exitError(message: "dynamic wallpaper requires at least 2 images (found \(inputImages.count))", code: 2, asJson: asJson)
        }

        // Determine schedule mode if not explicitly specified
        let effectiveSchedule: DynamicScheduleMode
        if let s = dynamicSchedule {
            effectiveSchedule = s
        } else if inputImages.count == 2 {
            effectiveSchedule = .appearance
        } else if inputImages.count >= 16 {
            effectiveSchedule = .solar
        } else {
            effectiveSchedule = .time
        }

        // Determine output destination
        let finalOutURL: URL
        if let outPath = dynamicOut {
            finalOutURL = URL(fileURLWithPath: (outPath as NSString).expandingTildeInPath)
        } else if isDir.boolValue {
            let dirName = firstArgURL.lastPathComponent
            finalOutURL = firstArgURL.appendingPathComponent("\(dirName)-dynamic.heic")
        } else {
            let firstBase = firstArgURL.deletingPathExtension().lastPathComponent
            finalOutURL = firstArgURL.deletingLastPathComponent().appendingPathComponent("\(firstBase)-dynamic.heic")
        }

        if !asJson && !quiet {
            print("🌅 Building dynamic HEIC wallpaper:")
            print("   📁 Input:    \(inputImages.count) images from \(isDir.boolValue ? firstArgURL.path : "file list")")
            print("   ⚙️  Schedule: \(effectiveSchedule.rawValue)")
            print("   📦 Output:   \(finalOutURL.path)")
        }

        do {
            try buildDynamicHEIC(
                images: inputImages,
                scheduleMode: effectiveSchedule,
                outputURL: finalOutURL,
                lightOverride: lightOverride,
                darkOverride: darkOverride,
                verbose: verbose
            )
        } catch {
            exitError(message: "Failed to generate dynamic HEIC: \(error.localizedDescription)", code: 1, asJson: asJson)
        }

        let outSize = (try? FileManager.default.attributesOfItem(atPath: finalOutURL.path)[.size] as? Int) ?? 0

        if !asJson && !quiet {
            let mb = Double(outSize) / 1_048_576.0
            print(String(format: "   ✅ Successfully generated %.2f MB dynamic wallpaper", mb))
        }

        // If --apply or if no output was explicitly specified, apply it
        let shouldApply = dynamicApply || (dynamicOut == nil)
        if shouldApply {
            applyWallpaper(
                imageURL: finalOutURL,
                targets: targets,
                mode: mode,
                fillColor: color,
                dryRun: dryRun,
                asJson: asJson,
                quiet: quiet,
                verbose: verbose
            )
        } else if asJson {
            let dict: [String: Any] = [
                "success": true,
                "output": finalOutURL.path,
                "size": outSize,
                "schedule": effectiveSchedule.rawValue,
                "frames": inputImages.count,
                "applied": false
            ]
            if let data = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]),
               let str = String(data: data, encoding: .utf8) {
                print(str)
            }
        }

    case "random":
        guard let dirPath = rest.first else {
            exitError(message: "'random' requires a directory path", code: 2, asJson: asJson)
        }
        let dirURL = URL(fileURLWithPath: (dirPath as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue else {
            exitError(message: "not a directory: \(dirURL.path)", code: 1, asJson: asJson)
        }
        let images = findImages(in: dirURL)
        guard !images.isEmpty else {
            exitError(message: "no supported images found in \(dirURL.path)", code: 1, asJson: asJson)
        }
        let selected = images.randomElement()!
        applyWallpaper(imageURL: selected, targets: targets, mode: mode, fillColor: color, dryRun: dryRun, asJson: asJson, quiet: quiet, verbose: verbose)

    case "rotate":
        guard let dirPath = rest.first else {
            exitError(message: "'rotate' requires a directory path", code: 2, asJson: asJson)
        }
        let intervalStr = rest.count > 1 ? rest[1] : "30m"
        guard let interval = parseInterval(intervalStr), interval > 0 else {
            exitError(message: "invalid interval '\(intervalStr)' (try 30s, 5m, 1h, 1d)", code: 2, asJson: asJson)
        }
        let dirURL = URL(fileURLWithPath: (dirPath as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue else {
            exitError(message: "not a directory: \(dirURL.path)", code: 1, asJson: asJson)
        }
        let images = findImages(in: dirURL)
        guard !images.isEmpty else {
            exitError(message: "no supported images found in \(dirURL.path)", code: 1, asJson: asJson)
        }

        print("🎲 Wallpaper rotation active")
        print("📁 Directory: \(dirURL.path) (\(images.count) images)")
        print("⏱  Interval:  \(intervalStr) (\(Int(interval))s)")
        print("🖥  Targets:   \(targets.map { "[\($0.index)] \($0.name)" }.joined(separator: ", "))")
        print("Press Ctrl-C to stop.\n")

        while true {
            let picked = images.randomElement()!
            applyWallpaper(imageURL: picked, targets: targets, mode: mode, fillColor: color, dryRun: dryRun, asJson: asJson, quiet: quiet, verbose: verbose)
            Thread.sleep(forTimeInterval: interval)
        }

    case "spaces":
        guard let sourcePath = rest.first else {
            exitError(message: "'spaces' requires a directory of images or multiple images", code: 2, asJson: asJson)
        }
        let sourceURL = URL(fileURLWithPath: (sourcePath as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        var imagesToAssign: [URL] = []

        if FileManager.default.fileExists(atPath: sourceURL.path, isDirectory: &isDir), isDir.boolValue {
            imagesToAssign = findImages(in: sourceURL)
            guard !imagesToAssign.isEmpty else {
                exitError(message: "no supported images found in \(sourceURL.path)", code: 1, asJson: asJson)
            }
        } else {
            imagesToAssign = rest.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        }

        print("🖥  Configuring wallpapers across \(targets.count) target display(s):")
        for (idx, targetScreen) in targets.enumerated() {
            let img = imagesToAssign[idx % imagesToAssign.count]
            print("   Display [\(targetScreen.index)] (\(targetScreen.name)) ← \(img.lastPathComponent)")
            applyWallpaper(imageURL: img, targets: [targetScreen], mode: mode, fillColor: color, dryRun: dryRun, asJson: asJson, quiet: quiet, verbose: verbose)
        }

    default:
        // Convenient shorthand:
        // background-tool path/to/image.png
        // background-tool path/to/folder
        let pathURL = URL(fileURLWithPath: (command as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: pathURL.path, isDirectory: &isDir) {
            if isDir.boolValue {
                let images = findImages(in: pathURL)
                guard !images.isEmpty else {
                    exitError(message: "no supported images in \(pathURL.path)", code: 1, asJson: asJson)
                }
                let picked = images.randomElement()!
                applyWallpaper(imageURL: picked, targets: targets, mode: mode, fillColor: color, dryRun: dryRun, asJson: asJson, quiet: quiet, verbose: verbose)
            } else {
                applyWallpaper(imageURL: pathURL, targets: targets, mode: mode, fillColor: color, dryRun: dryRun, asJson: asJson, quiet: quiet, verbose: verbose)
            }
        } else {
            exitError(message: "unknown command or file path not found: '\(command)'. See --help.", code: 2, asJson: asJson)
        }
    }
}

main()
