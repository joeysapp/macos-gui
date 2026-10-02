import Foundation
import CoreGraphics
import AppKit

let VERSION = "0.1.0"
let TOOL_NAME = "display_ctl"

// MARK: - Display Info Struct

struct DisplayInfo {
    let index: Int
    let id: CGDirectDisplayID
    let name: String
    let isMain: Bool
    let isBuiltin: Bool
    let logicalWidth: Int
    let logicalHeight: Int
    let pixelWidth: Int
    let pixelHeight: Int
    let refreshRate: Double
    let mirrorMaster: CGDirectDisplayID // 0 if not mirroring

    var isMirrored: Bool {
        return mirrorMaster != 0
    }
}

// MARK: - Discovery & Resolution

func getOnlineDisplays() -> [DisplayInfo] {
    var count: UInt32 = 0
    CGGetOnlineDisplayList(0, nil, &count)
    guard count > 0 else { return [] }

    var displayIDs = [CGDirectDisplayID](repeating: 0, count: Int(count))
    CGGetOnlineDisplayList(count, &displayIDs, &count)

    var screensByNumber: [CGDirectDisplayID: NSScreen] = [:]
    for s in NSScreen.screens {
        let num = (s.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value ?? 0
        if num != 0 {
            screensByNumber[num] = s
        }
    }

    var result: [DisplayInfo] = []
    for (idx, id) in displayIDs.enumerated() {
        let isMain = CGDisplayIsMain(id) != 0
        let isBuiltin = CGDisplayIsBuiltin(id) != 0
        let mirrorMaster = CGDisplayMirrorsDisplay(id)

        let screen = screensByNumber[id]
        let name = screen?.localizedName ?? (isBuiltin ? "Built-in Display" : "Display \(id)")

        let logicalW = Int(screen?.frame.width ?? CGFloat(CGDisplayPixelsWide(id)))
        let logicalH = Int(screen?.frame.height ?? CGFloat(CGDisplayPixelsHigh(id)))

        var pixW = CGDisplayPixelsWide(id)
        var pixH = CGDisplayPixelsHigh(id)
        var hz: Double = 60.0

        if let mode = CGDisplayCopyDisplayMode(id) {
            pixW = mode.pixelWidth
            pixH = mode.pixelHeight
            hz = mode.refreshRate
        }

        result.append(DisplayInfo(
            index: idx,
            id: id,
            name: name,
            isMain: isMain,
            isBuiltin: isBuiltin,
            logicalWidth: logicalW,
            logicalHeight: logicalH,
            pixelWidth: pixW,
            pixelHeight: pixH,
            refreshRate: hz,
            mirrorMaster: mirrorMaster
        ))
    }
    return result
}

func resolveTargetDisplay(query: String?, displays: [DisplayInfo], allowDefaultSecondary: Bool = false) -> DisplayInfo? {
    if let q = query?.trimmingCharacters(in: .whitespacesAndNewlines), !q.isEmpty {
        // Match by index (0, 1) or ID (1, 29, etc.)
        if let num = UInt32(q) {
            if let match = displays.first(where: { $0.id == num || UInt32($0.index) == num }) {
                return match
            }
        }

        // Match by name substring
        let lower = q.lowercased()
        if let match = displays.first(where: { $0.name.lowercased().contains(lower) }) {
            return match
        }

        fputs("error: Display matching '\(q)' not found.\n", stderr)
        return nil
    }

    if allowDefaultSecondary {
        // Default to first non-main (secondary/external/airplay) display
        if let secondary = displays.first(where: { !$0.isMain }) {
            return secondary
        }
    }

    return displays.first(where: { $0.isMain }) ?? displays.first
}

// MARK: - CoreGraphics Display Operations

func setMirror(target: DisplayInfo, master: DisplayInfo) -> Bool {
    var config: CGDisplayConfigRef?
    var err = CGBeginDisplayConfiguration(&config)
    guard err == .success, let cfg = config else {
        fputs("error: Failed to begin display configuration (\(err.rawValue))\n", stderr)
        return false
    }

    err = CGConfigureDisplayMirrorOfDisplay(cfg, target.id, master.id)
    if err != .success {
        fputs("error: Failed to configure mirroring: \(err.rawValue)\n", stderr)
        CGCancelDisplayConfiguration(cfg)
        return false
    }

    err = CGCompleteDisplayConfiguration(cfg, .permanently)
    if err != .success {
        fputs("error: Failed to apply display configuration: \(err.rawValue)\n", stderr)
        return false
    }

    print("✅ Display [\(target.index)] \"\(target.name)\" is now mirroring [\(master.index)] \"\(master.name)\" (ID: \(master.id))")
    return true
}

func setExtend(target: DisplayInfo) -> Bool {
    var config: CGDisplayConfigRef?
    var err = CGBeginDisplayConfiguration(&config)
    guard err == .success, let cfg = config else {
        fputs("error: Failed to begin display configuration (\(err.rawValue))\n", stderr)
        return false
    }

    err = CGConfigureDisplayMirrorOfDisplay(cfg, target.id, kCGNullDirectDisplay)
    if err != .success {
        fputs("error: Failed to configure extended mode: \(err.rawValue)\n", stderr)
        CGCancelDisplayConfiguration(cfg)
        return false
    }

    err = CGCompleteDisplayConfiguration(cfg, .permanently)
    if err != .success {
        fputs("error: Failed to apply display configuration: \(err.rawValue)\n", stderr)
        return false
    }

    print("✅ Display [\(target.index)] \"\(target.name)\" is now set to Extended Desktop (independent)")
    return true
}

func listResolutions(target: DisplayInfo) {
    guard let modes = CGDisplayCopyAllDisplayModes(target.id, nil) as? [CGDisplayMode] else {
        fputs("error: Could not query display modes for [\(target.index)] \"\(target.name)\"\n", stderr)
        return
    }

    print("Available resolutions for [\(target.index)] \"\(target.name)\" (ID: \(target.id)):")

    var seen: Set<String> = []
    for mode in modes {
        let hz = Int(round(mode.refreshRate))
        let key = "\(mode.width)x\(mode.height)@\(hz)Hz (pixel: \(mode.pixelWidth)x\(mode.pixelHeight))"
        if !seen.contains(key) {
            seen.insert(key)
            print("  \(key)")
        }
    }
}

func setResolution(target: DisplayInfo, width: Int, height: Int, refreshRate: Double? = nil) -> Bool {
    guard let modes = CGDisplayCopyAllDisplayModes(target.id, nil) as? [CGDisplayMode] else {
        fputs("error: Could not query display modes for display \(target.id)\n", stderr)
        return false
    }

    var bestMode: CGDisplayMode? = nil
    for mode in modes {
        if mode.width == width && mode.height == height {
            if let targetHz = refreshRate {
                if abs(mode.refreshRate - targetHz) < 1.0 {
                    bestMode = mode
                    break
                }
            } else if bestMode == nil || mode.refreshRate > bestMode!.refreshRate {
                bestMode = mode
            }
        }
    }

    guard let selected = bestMode else {
        fputs("error: Mode \(width)x\(height) not found for [\(target.index)] \"\(target.name)\". Run '\(TOOL_NAME) resolutions \(target.index)' to see available modes.\n", stderr)
        return false
    }

    var config: CGDisplayConfigRef?
    var err = CGBeginDisplayConfiguration(&config)
    guard err == .success, let cfg = config else {
        fputs("error: Failed to begin display configuration\n", stderr)
        return false
    }

    err = CGConfigureDisplayWithDisplayMode(cfg, target.id, selected, nil)
    guard err == .success else {
        fputs("error: Failed to configure display mode\n", stderr)
        CGCancelDisplayConfiguration(cfg)
        return false
    }

    err = CGCompleteDisplayConfiguration(cfg, .permanently)
    guard err == .success else {
        fputs("error: Failed to apply display mode\n", stderr)
        return false
    }

    let hzStr = selected.refreshRate > 0 ? " @ \(Int(round(selected.refreshRate)))Hz" : ""
    print("✅ Changed [\(target.index)] \"\(target.name)\" resolution to \(width)x\(height)\(hzStr)")
    return true
}

// MARK: - CLI Commands

func printUsage() {
    print("""
Usage:
  \(TOOL_NAME) list
  \(TOOL_NAME) mirror [target-display] [master-display]
  \(TOOL_NAME) extend [target-display]
  \(TOOL_NAME) toggle [target-display]
  \(TOOL_NAME) resolutions [target-display]
  \(TOOL_NAME) resolution [target-display] <width>x<height> [refreshRate]
  \(TOOL_NAME) status

Options:
  -h, --help       Show this help message
  -v, --version    Show tool version

Examples:
  \(TOOL_NAME) list
  \(TOOL_NAME) mirror                    # Mirror secondary (AirPlay/external) to main display
  \(TOOL_NAME) mirror 1 0                # Mirror display 1 to display 0
  \(TOOL_NAME) extend                    # Extend secondary display (unmirror)
  \(TOOL_NAME) toggle                    # Toggle mirroring on secondary display
  \(TOOL_NAME) resolutions "LG"          # List supported resolutions for LG TV
  \(TOOL_NAME) resolution "LG" 1920x1080 # Set resolution
""")
}

func cmdList() {
    let displays = getOnlineDisplays()
    print("Found \(displays.count) online display(s):\n")

    for d in displays {
        let typeStr = d.isBuiltin ? "Built-in" : "External / AirPlay"
        let roleStr = d.isMain ? "Main Display" : "Secondary Display"
        let mirrorStr = d.isMirrored ? "Mirroring Display ID \(d.mirrorMaster)" : "Extended Desktop (Independent)"
        let hzStr = d.refreshRate > 0 ? " @ \(Int(round(d.refreshRate)))Hz" : ""

        print("  Display [\(d.index)]")
        print("    ID:          \(d.id)")
        print("    Name:        \(d.name)")
        print("    Type:        \(typeStr) (\(roleStr))")
        print("    Resolution:  \(d.logicalWidth)x\(d.logicalHeight) (Physical: \(d.pixelWidth)x\(d.pixelHeight))\(hzStr)")
        print("    Mode:        \(mirrorStr)")
        print("")
    }
}

// MARK: - Main CLI Entry Point

func main() {
    let args = CommandLine.arguments

    if args.count < 2 {
        cmdList()
        exit(0)
    }

    let command = args[1]

    switch command {
    case "-h", "--help", "help":
        printUsage()
        exit(0)

    case "-v", "--version", "version":
        print("\(TOOL_NAME) \(VERSION)")
        exit(0)

    case "list", "status":
        cmdList()

    case "mirror":
        let displays = getOnlineDisplays()
        guard displays.count > 1 else {
            fputs("error: Only 1 display connected. Mirroring requires at least 2 displays.\n", stderr)
            exit(1)
        }

        let targetArg = args.count > 2 ? args[2] : nil
        let masterArg = args.count > 3 ? args[3] : nil

        guard let target = resolveTargetDisplay(query: targetArg, displays: displays, allowDefaultSecondary: true) else {
            exit(1)
        }

        let master: DisplayInfo
        if let mArg = masterArg {
            guard let resolvedMaster = resolveTargetDisplay(query: mArg, displays: displays) else {
                exit(1)
            }
            master = resolvedMaster
        } else {
            // Default master is the main display
            master = displays.first(where: { $0.isMain }) ?? displays[0]
        }

        if target.id == master.id {
            fputs("error: Target display and master display cannot be the same display.\n", stderr)
            exit(1)
        }

        let ok = setMirror(target: target, master: master)
        exit(ok ? 0 : 1)

    case "extend", "unmirror":
        let displays = getOnlineDisplays()
        guard displays.count > 1 else {
            print("Only 1 display connected (already independent).")
            exit(0)
        }

        let targetArg = args.count > 2 ? args[2] : nil
        guard let target = resolveTargetDisplay(query: targetArg, displays: displays, allowDefaultSecondary: true) else {
            exit(1)
        }

        let ok = setExtend(target: target)
        exit(ok ? 0 : 1)

    case "toggle", "toggle-mirror":
        let displays = getOnlineDisplays()
        guard displays.count > 1 else {
            fputs("error: Only 1 display connected.\n", stderr)
            exit(1)
        }

        let targetArg = args.count > 2 ? args[2] : nil
        guard let target = resolveTargetDisplay(query: targetArg, displays: displays, allowDefaultSecondary: true) else {
            exit(1)
        }

        if target.isMirrored {
            let ok = setExtend(target: target)
            exit(ok ? 0 : 1)
        } else {
            let master = displays.first(where: { $0.isMain && $0.id != target.id }) ?? displays[0]
            let ok = setMirror(target: target, master: master)
            exit(ok ? 0 : 1)
        }

    case "resolutions", "modes":
        let displays = getOnlineDisplays()
        let targetArg = args.count > 2 ? args[2] : nil
        guard let target = resolveTargetDisplay(query: targetArg, displays: displays, allowDefaultSecondary: true) else {
            exit(1)
        }
        listResolutions(target: target)

    case "resolution", "mode", "res":
        let displays = getOnlineDisplays()
        guard args.count >= 3 else {
            fputs("error: Usage: \(TOOL_NAME) resolution [target-display] <width>x<height> [refreshRate]\n", stderr)
            exit(1)
        }

        var targetArg: String? = nil
        var resArg: String = ""
        var hzArg: Double? = nil

        if args[2].contains("x") {
            // No target display passed, target default
            resArg = args[2]
            if args.count > 3 {
                hzArg = Double(args[3].replacingOccurrences(of: "hz", with: "", options: .caseInsensitive))
            }
        } else {
            targetArg = args[2]
            guard args.count >= 4 else {
                fputs("error: Missing <width>x<height> argument\n", stderr)
                exit(1)
            }
            resArg = args[3]
            if args.count > 4 {
                hzArg = Double(args[4].replacingOccurrences(of: "hz", with: "", options: .caseInsensitive))
            }
        }

        let parts = resArg.lowercased().split(separator: "x")
        guard parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]) else {
            fputs("error: Invalid resolution '\(resArg)'. Expected format: 1920x1080\n", stderr)
            exit(1)
        }

        guard let target = resolveTargetDisplay(query: targetArg, displays: displays, allowDefaultSecondary: true) else {
            exit(1)
        }

        let ok = setResolution(target: target, width: w, height: h, refreshRate: hzArg)
        exit(ok ? 0 : 1)

    default:
        fputs("error: Unknown command '\(command)'\n", stderr)
        printUsage()
        exit(1)
    }
}

main()
