import Foundation
import AppKit

let VERSION = "0.2.0"
let TOOL_NAME = "background_set"

// MARK: - Enums & Types

enum WallpaperMode: String, CaseIterable {
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

// MARK: - Display Helper

struct ScreenInfo {
    let index: Int
    let screen: NSScreen
    let id: Int
    let name: String
    let resolution: String

    init(index: Int, screen: NSScreen) {
        self.index = index
        self.screen = screen
        let desc = screen.deviceDescription
        let num = (desc[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.intValue ?? (index + 1)
        self.id = num
        self.name = screen.localizedName
        self.resolution = "\(Int(screen.frame.width))x\(Int(screen.frame.height))"
    }
}

func getAllScreens() -> [ScreenInfo] {
    return NSScreen.screens.enumerated().map { ScreenInfo(index: $0.offset, screen: $0.element) }
}

func resolveTargetScreens(target: String?, screens: [ScreenInfo]) -> [ScreenInfo] {
    guard let target = target?.trimmingCharacters(in: .whitespacesAndNewlines), !target.isEmpty else {
        return screens
    }

    if target.lowercased() == "all" {
        return screens
    }

    // Match by 0-based index or 1-based display number
    if let intVal = Int(target) {
        if let match = screens.first(where: { $0.index == intVal || $0.id == intVal }) {
            return [match]
        }
    }

    // Match by localized display name substring (e.g. "built-in", "retina", "lg")
    let lower = target.lowercased()
    let matches = screens.filter { $0.name.lowercased().contains(lower) }
    if !matches.isEmpty {
        return matches
    }

    fputs("error: No display found matching '\(target)'. Available displays:\n", stderr)
    for s in screens {
        fputs("  [\(s.index)] ID: \(s.id) - \"\(s.name)\" (\(s.resolution))\n", stderr)
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

// MARK: - Image Directory Finding

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
    return results.sorted { $0.lastPathComponent < $1.lastPathComponent }
}

// MARK: - Core Wallpaper Setting

func applyWallpaper(imageURL: URL,
                    targets: [ScreenInfo],
                    mode: WallpaperMode?,
                    fillColor: NSColor?,
                    dryRun: Bool) {
    let workspace = NSWorkspace.shared

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

        print("🖥  Display [\(screenInfo.index)] (ID: \(screenInfo.id) - \"\(screenInfo.name)\")")
        print("   🖼  Image: \(imageURL.path)")
        print("   📐  Mode:  \(modeStr)")
        print("   🎨  Color: \(colorStr)")

        if dryRun {
            print("   [dry-run] skipped actual system call")
            continue
        }

        do {
            try workspace.setDesktopImageURL(imageURL, for: screenInfo.screen, options: options)
            print("   ✅ Applied successfully")
        } catch {
            fputs("   ❌ Error setting wallpaper on display [\(screenInfo.index)]: \(error.localizedDescription)\n", stderr)
        }
    }
}

// MARK: - CLI Commands

struct ScreenJSON: Codable {
    let index: Int
    let id: Int
    let name: String
    let resolution: String
    let wallpaper: String
    let mode: String
    let fillColor: String
}

func printUsage() {
    print("""
Usage:
  \(TOOL_NAME) <image> [options]
  \(TOOL_NAME) set <image> [options]
  \(TOOL_NAME) color <hex|name> [options]
  \(TOOL_NAME) random <directory> [options]
  \(TOOL_NAME) rotate <directory> [interval] [options]
  \(TOOL_NAME) spaces <directory|images...> [options]
  \(TOOL_NAME) list [options]
  \(TOOL_NAME) get [options]

Modes:
  fill     Fill screen (crop/zoom to cover without distorting) [default]
  fit      Fit to screen (preserve aspect ratio with matte borders)
  stretch  Stretch to fill screen (scale axes independently)
  center   Center at original pixel size without scaling

Options:
  -j, --json                     Output machine-readable JSON format
  -d, --display <id|index|name>  Target specific display (0, 1, display ID, or name). Default: all displays
  -m, --mode <mode>              Wallpaper scaling mode: fill, fit, stretch, center
  -c, --color <hex|name>         Matte background / fill color (e.g. '#1e1e1e', 'black', 'white')
  -n, --dry-run                  Print what would happen without changing anything
  -h, --help                     Show this help message
  -v, --version                  Show tool version

Examples:
  \(TOOL_NAME) ~/Pictures/wallpaper.png
  \(TOOL_NAME) ~/Pictures/art.png --mode fit --color "#141414"
  \(TOOL_NAME) color "#0d1117" --display 0
  \(TOOL_NAME) random ~/Pictures/Wallpapers
  \(TOOL_NAME) rotate ~/Pictures/Wallpapers 15m --mode fill
  \(TOOL_NAME) spaces ~/Pictures/Monitors/
  \(TOOL_NAME) list
  \(TOOL_NAME) list -j
""")
}

func cmdList(asJson: Bool = false) {
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
                resolution: s.resolution,
                wallpaper: curURL,
                mode: mode,
                fillColor: colorStr
            )
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
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
        print("    Name:        \(s.name)")
        print("    Resolution:  \(s.resolution)")
        print("    Wallpaper:   \(curURL)")
        print("    Mode:        \(mode)")
        print("    Fill Color:  \(colorStr)")
        print("")
    }
}

func cmdGet(targets: [ScreenInfo], asJson: Bool = false) {
    let workspace = NSWorkspace.shared
    if asJson {
        var dict: [String: String] = [:]
        for s in targets {
            let curURL = workspace.desktopImageURL(for: s.screen)?.path ?? "none"
            dict["display_\(s.index)"] = curURL
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        if let data = try? encoder.encode(dict), let str = String(data: data, encoding: .utf8) {
            print(str)
        }
        return
    }

    for s in targets {
        let curURL = workspace.desktopImageURL(for: s.screen)?.path ?? "none"
        if targets.count == 1 {
            print(curURL)
        } else {
            print("[\(s.index)] \(s.name): \(curURL)")
        }
    }
}

// MARK: - Main CLI Parsing

func main() {
    let args = CommandLine.arguments

    if args.count < 2 {
        printUsage()
        exit(1)
    }

    var targetDisplay: String? = nil
    var mode: WallpaperMode? = nil
    var color: NSColor? = nil
    var rawColorString: String? = nil
    var dryRun = false
    var asJson = false
    var positional: [String] = []

    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "-j", "--json", "json":
            asJson = true
        case "-h", "--help":
            printUsage()
            exit(0)
        case "-v", "--version":
            print("\(TOOL_NAME) \(VERSION)")
            exit(0)
        case "-n", "--dry-run":
            dryRun = true
        case "-d", "--display":
            i += 1
            guard i < args.count else {
                fputs("error: --display requires an argument\n", stderr)
                exit(1)
            }
            targetDisplay = args[i]
        case "-m", "--mode":
            i += 1
            guard i < args.count else {
                fputs("error: --mode requires an argument (fill, fit, stretch, center)\n", stderr)
                exit(1)
            }
            guard let parsedMode = WallpaperMode(rawValue: args[i].lowercased()) else {
                fputs("error: invalid mode '\(args[i])'. Valid modes: fill, fit, stretch, center\n", stderr)
                exit(1)
            }
            mode = parsedMode
        case "-c", "--color":
            i += 1
            guard i < args.count else {
                fputs("error: --color requires a hex or color name argument\n", stderr)
                exit(1)
            }
            rawColorString = args[i]
            guard let parsedColor = parseColor(args[i]) else {
                fputs("error: invalid color '\(args[i])'. Try hex (e.g. '#1e1e1e') or color name (e.g. 'black')\n", stderr)
                exit(1)
            }
            color = parsedColor
        default:
            if arg.hasPrefix("-") {
                fputs("error: unknown option '\(arg)'\n", stderr)
                exit(1)
            } else {
                positional.append(arg)
            }
        }
        i += 1
    }

    let allScreens = getAllScreens()
    let targets = resolveTargetScreens(target: targetDisplay, screens: allScreens)

    guard !positional.isEmpty else {
        if asJson {
            cmdList(asJson: true)
            return
        }
        // If color was specified without positional, treat as solid color command
        if let col = color, let rawCol = rawColorString {
            guard let imgURL = generateSolidColorImage(color: col, hexName: rawCol) else {
                fputs("error: failed to create solid color image\n", stderr)
                exit(1)
            }
            applyWallpaper(imageURL: imgURL, targets: targets, mode: mode ?? .fill, fillColor: col, dryRun: dryRun)
            return
        }
        printUsage()
        exit(1)
    }

    let command = positional[0]
    let rest = Array(positional.dropFirst())

    switch command {
    case "list", "status":
        cmdList(asJson: asJson)

    case "get":
        cmdGet(targets: targets, asJson: asJson)

    case "color":
        guard !rest.isEmpty || rawColorString != nil else {
            fputs("error: 'color' command requires a color hex or name\n", stderr)
            exit(1)
        }
        let colStr = rest.first ?? rawColorString!
        guard let col = parseColor(colStr) else {
            fputs("error: invalid color '\(colStr)'\n", stderr)
            exit(1)
        }
        guard let imgURL = generateSolidColorImage(color: col, hexName: colStr) else {
            fputs("error: failed to create solid color image\n", stderr)
            exit(1)
        }
        applyWallpaper(imageURL: imgURL, targets: targets, mode: mode ?? .fill, fillColor: col, dryRun: dryRun)

    case "set":
        guard let path = rest.first else {
            fputs("error: 'set' requires an image path\n", stderr)
            exit(1)
        }
        let imgURL = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: imgURL.path) else {
            fputs("error: file not found: \(imgURL.path)\n", stderr)
            exit(1)
        }
        applyWallpaper(imageURL: imgURL, targets: targets, mode: mode, fillColor: color, dryRun: dryRun)

    case "random":
        guard let dirPath = rest.first else {
            fputs("error: 'random' requires a directory path\n", stderr)
            exit(1)
        }
        let dirURL = URL(fileURLWithPath: (dirPath as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue else {
            fputs("error: not a directory: \(dirURL.path)\n", stderr)
            exit(1)
        }
        let images = findImages(in: dirURL)
        guard !images.isEmpty else {
            fputs("error: no supported images found in \(dirURL.path)\n", stderr)
            exit(1)
        }
        let selected = images.randomElement()!
        applyWallpaper(imageURL: selected, targets: targets, mode: mode, fillColor: color, dryRun: dryRun)

    case "rotate":
        guard let dirPath = rest.first else {
            fputs("error: 'rotate' requires a directory path\n", stderr)
            exit(1)
        }
        let intervalStr = rest.count > 1 ? rest[1] : "30m"
        guard let interval = parseInterval(intervalStr), interval > 0 else {
            fputs("error: invalid interval '\(intervalStr)' (try 30s, 5m, 1h, 1d)\n", stderr)
            exit(1)
        }
        let dirURL = URL(fileURLWithPath: (dirPath as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue else {
            fputs("error: not a directory: \(dirURL.path)\n", stderr)
            exit(1)
        }
        let images = findImages(in: dirURL)
        guard !images.isEmpty else {
            fputs("error: no supported images found in \(dirURL.path)\n", stderr)
            exit(1)
        }

        print("🎲 Wallpaper rotation active")
        print("📁 Directory: \(dirURL.path) (\(images.count) images)")
        print("⏱  Interval:  \(intervalStr) (\(Int(interval))s)")
        print("🖥  Targets:   \(targets.map { "[\($0.index)] \($0.name)" }.joined(separator: ", "))")
        print("Press Ctrl-C to stop.\n")

        // Run foreground rotation loop
        while true {
            let picked = images.randomElement()!
            applyWallpaper(imageURL: picked, targets: targets, mode: mode, fillColor: color, dryRun: dryRun)
            Thread.sleep(forTimeInterval: interval)
        }

    case "spaces":
        // Distribute different wallpapers across connected displays
        guard let sourcePath = rest.first else {
            fputs("error: 'spaces' requires a directory of images or multiple images\n", stderr)
            exit(1)
        }
        let sourceURL = URL(fileURLWithPath: (sourcePath as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        var imagesToAssign: [URL] = []

        if FileManager.default.fileExists(atPath: sourceURL.path, isDirectory: &isDir), isDir.boolValue {
            imagesToAssign = findImages(in: sourceURL)
            guard !imagesToAssign.isEmpty else {
                fputs("error: no supported images found in \(sourceURL.path)\n", stderr)
                exit(1)
            }
        } else {
            // Positional image list
            imagesToAssign = rest.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        }

        print("🖥  Configuring wallpapers across \(targets.count) target display(s):")
        for (idx, targetScreen) in targets.enumerated() {
            let img = imagesToAssign[idx % imagesToAssign.count]
            print("   Display [\(targetScreen.index)] (\(targetScreen.name)) ← \(img.lastPathComponent)")
            applyWallpaper(imageURL: img, targets: [targetScreen], mode: mode, fillColor: color, dryRun: dryRun)
        }

    default:
        // Convenient shorthand:
        // background_set path/to/image.png
        // background_set path/to/folder
        let pathURL = URL(fileURLWithPath: (command as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: pathURL.path, isDirectory: &isDir) {
            if isDir.boolValue {
                let images = findImages(in: pathURL)
                guard !images.isEmpty else {
                    fputs("error: no supported images in \(pathURL.path)\n", stderr)
                    exit(1)
                }
                let picked = images.randomElement()!
                applyWallpaper(imageURL: picked, targets: targets, mode: mode, fillColor: color, dryRun: dryRun)
            } else {
                applyWallpaper(imageURL: pathURL, targets: targets, mode: mode, fillColor: color, dryRun: dryRun)
            }
        } else {
            fputs("error: unknown command or file path not found: '\(command)'\n", stderr)
            exit(1)
        }
    }
}

main()
