import Foundation
import AppKit

let VERSION = "0.1.0"
let TOOL_NAME = "hot-corner-tool"

// MARK: - CoreDock Binding

typealias CoreDockSendNotificationFunc = @convention(c) (CFString, UnsafeMutableRawPointer?) -> Void

func sendDockNotification(_ name: String, verbose: Bool) -> Bool {
    if verbose {
        fputs("[debug] Loading HIServices framework...\n", stderr)
    }
    guard let handle = dlopen("/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/HIServices.framework/Versions/A/HIServices", RTLD_LAZY) else {
        if verbose {
            fputs("[debug] Failed to dlopen HIServices\n", stderr)
        }
        return false
    }
    defer { dlclose(handle) }

    guard let sym = dlsym(handle, "CoreDockSendNotification") else {
        if verbose {
            fputs("[debug] Failed to resolve CoreDockSendNotification symbol\n", stderr)
        }
        return false
    }
    let fn = unsafeBitCast(sym, to: CoreDockSendNotificationFunc.self)
    if verbose {
        fputs("[debug] Invoking CoreDockSendNotification with '\(name)'\n", stderr)
    }
    fn(name as CFString, nil)
    return true
}

func toggleNotificationCenter(verbose: Bool) -> (success: Bool, error: String?) {
    if verbose {
        fputs("[debug] Executing AppleScript targeting ControlCenter Notification Center...\n", stderr)
    }
    let script = """
    tell application "System Events"
        tell process "ControlCenter"
            click (menu bar item 1 of menu bar 1 whose description contains "Clock" or description contains "Notification Center")
        end tell
    end tell
    """
    var errDict: NSDictionary?
    if let appleScript = NSAppleScript(source: script) {
        appleScript.executeAndReturnError(&errDict)
        if let err = errDict {
            let msg = err[NSAppleScript.errorMessage] as? String ?? "\(err)"
            if verbose {
                fputs("[debug] AppleScript error: \(msg)\n", stderr)
            }
            return (false, msg)
        }
        return (true, nil)
    }
    return (false, "Failed to compile AppleScript")
}

// MARK: - Output & Exit Helpers

func emitJSON(dict: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]),
       let str = String(data: data, encoding: .utf8) {
        print(str)
    }
}

func exitJSONError(message: String, code: Int32) -> Never {
    let dict: [String: Any] = [
        "error": message,
        "code": code,
        "success": false
    ]
    emitJSON(dict: dict)
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

// MARK: - Help

func printUsage() {
    print("""
Usage:
  \(TOOL_NAME) <command> [options]

Commands:
  desktop              Toggle Show Desktop (hide/show application windows)
  mission-control      Toggle Mission Control (all windows and spaces overview)
  app-windows          Toggle Application Windows (Exposé for frontmost app)
  notification-center  Toggle Notification Center & widget sidebar

Aliases:
  desktop              show-desktop, desk
  mission-control      mc
  app-windows          windows, expose
  notification-center  notifications, nc

Options:
  -j, --json           Output result in JSON format
  -v, --verbose        Emit debug diagnostics to stderr
  -q, --quiet          Suppress normal stdout output
  -V, --version        Show version information
  -h, --help           Show this help message

Exit Codes:
  0  Success / action triggered
  1  Runtime failure (e.g. missing permission)
  2  Usage error (unknown command or option)

Examples:
  \(TOOL_NAME) desktop
  \(TOOL_NAME) mission-control -j
  \(TOOL_NAME) app-windows
  \(TOOL_NAME) notification-center -q
""")
}

func printCommandHelp(_ cmd: String) {
    switch cmd {
    case "desktop", "show-desktop", "desk":
        print("""
Usage: \(TOOL_NAME) desktop [options]

Toggles the macOS "Show Desktop" action, moving all application windows aside
to reveal the desktop and desktop icons. Run again to restore windows.

Options:
  -j, --json     Output result in JSON format
  -v, --verbose  Emit debug diagnostics to stderr
  -q, --quiet    Suppress normal output
  -h, --help     Show this help message
""")
    case "mission-control", "mc":
        print("""
Usage: \(TOOL_NAME) mission-control [options]

Toggles macOS Mission Control, displaying an overview of all open windows,
full-screen spaces, and desktop spaces. Run again to dismiss.

Options:
  -j, --json     Output result in JSON format
  -v, --verbose  Emit debug diagnostics to stderr
  -q, --quiet    Suppress normal output
  -h, --help     Show this help message
""")
    case "app-windows", "windows", "expose":
        print("""
Usage: \(TOOL_NAME) app-windows [options]

Toggles Application Windows (App Exposé), showing all open windows of the
currently active (frontmost) application. Run again to dismiss.

Options:
  -j, --json     Output result in JSON format
  -v, --verbose  Emit debug diagnostics to stderr
  -q, --quiet    Suppress normal output
  -h, --help     Show this help message
""")
    case "notification-center", "notifications", "nc":
        print("""
Usage: \(TOOL_NAME) notification-center [options]

Toggles the macOS Notification Center and widget sidebar. Run again to dismiss.

Options:
  -j, --json     Output result in JSON format
  -v, --verbose  Emit debug diagnostics to stderr
  -q, --quiet    Suppress normal output
  -h, --help     Show this help message
""")
    default:
        printUsage()
    }
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
    var positional: [String] = []

    var i = 0
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "-j", "--json", "-v", "--verbose", "-q", "--quiet":
            break // Handled upfront
        case "-V", "--version", "version":
            print("\(TOOL_NAME) \(VERSION)")
            exit(0)
        case "-h", "--help":
            if !positional.isEmpty {
                printCommandHelp(positional[0])
            } else {
                printUsage()
            }
            exit(0)
        case "help":
            if i + 1 < args.count {
                printCommandHelp(args[i + 1])
            } else {
                printUsage()
            }
            exit(0)
        default:
            if arg.hasPrefix("-") {
                exitError(message: "unknown option '\(arg)'. See --help for available options.", code: 2, asJson: asJson)
            } else {
                positional.append(arg)
            }
        }
        i += 1
    }

    guard let command = positional.first else {
        printUsage()
        exit(2)
    }

    let startTime = Date().timeIntervalSince1970

    switch command.lowercased() {
    case "desktop", "show-desktop", "desk":
        let ok = sendDockNotification("com.apple.showdesktop.awake", verbose: verbose)
        if !ok {
            exitError(message: "Failed to invoke Show Desktop via Dock service", code: 1, asJson: asJson)
        }
        if asJson {
            emitJSON(dict: [
                "action": "desktop",
                "success": true,
                "timestamp": startTime
            ])
        } else if !quiet {
            print("🖥  Desktop toggled")
        }
        exit(0)

    case "mission-control", "mc":
        let ok = sendDockNotification("com.apple.expose.awake", verbose: verbose)
        if !ok {
            exitError(message: "Failed to invoke Mission Control via Dock service", code: 1, asJson: asJson)
        }
        if asJson {
            emitJSON(dict: [
                "action": "mission-control",
                "success": true,
                "timestamp": startTime
            ])
        } else if !quiet {
            print("🪟 Mission Control toggled")
        }
        exit(0)

    case "app-windows", "windows", "expose":
        let ok = sendDockNotification("com.apple.expose.front.awake", verbose: verbose)
        if !ok {
            exitError(message: "Failed to invoke Application Windows via Dock service", code: 1, asJson: asJson)
        }
        if asJson {
            emitJSON(dict: [
                "action": "app-windows",
                "success": true,
                "timestamp": startTime
            ])
        } else if !quiet {
            print("📑 Application Windows toggled")
        }
        exit(0)

    case "notification-center", "notifications", "nc":
        let result = toggleNotificationCenter(verbose: verbose)
        if !result.success {
            exitError(message: result.error ?? "Failed to toggle Notification Center", code: 1, asJson: asJson)
        }
        if asJson {
            emitJSON(dict: [
                "action": "notification-center",
                "success": true,
                "timestamp": startTime
            ])
        } else if !quiet {
            print("🔔 Notification Center toggled")
        }
        exit(0)

    default:
        exitError(message: "unknown command '\(command)'. Available commands: desktop, mission-control, app-windows, notification-center", code: 2, asJson: asJson)
    }
}

main()
