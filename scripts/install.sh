#!/bin/zsh

set -e

# Default installation paths
DEFAULT_BIN_DIR="$HOME/.local/bin"
DEFAULT_COMPLETION_DIR="$HOME/.zsh/completions"

REPO_DIR="${0:A:h:h}"
BIN_DIR="${INSTALL_BIN_DIR:-$DEFAULT_BIN_DIR}"
COMP_DIR="${INSTALL_COMP_DIR:-$DEFAULT_COMPLETION_DIR}"

DRY_RUN=0
UNINSTALL=0
FORCE=0

usage() {
    cat <<EOF
Usage:
  ./scripts/install.sh [options]

Options:
  --bin-dir <dir>        Directory to install executables (default: $DEFAULT_BIN_DIR)
  --comp-dir <dir>       Directory to install completions (default: $DEFAULT_COMPLETION_DIR)
  --uninstall            Remove symlinked tools and completions
  -n, --dry-run          Show what would be installed/removed without doing it
  --force                Install even while one of the tools is running
  -h, --help             Show this help message

Description:
  Installs individual macOS GUI tools and their zsh autocompletions.
  Each tool remains an independent utility (no monolithic wrapper).

  It refuses (exit 3) while any of the tools is running (a watch, a wallpaper
  rotation, a call from topic-get) and lists them: a Swift rebuild rewrites the
  binary in place, which can kill the process running it. --force skips that.
EOF
}

while (( $# )); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        --uninstall)
            UNINSTALL=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --bin-dir)
            shift
            BIN_DIR="$1"
            shift
            ;;
        --comp-dir)
            shift
            COMP_DIR="$1"
            shift
            ;;
        *)
            print -u2 "error: unknown argument '$1'"
            usage
            exit 1
            ;;
    esac
done

# Compile Swift tools if needed
compile_tools() {
    local bg_src="$REPO_DIR/tools/background-tool/background-tool.swift"
    local bg_bin="$REPO_DIR/tools/background-tool/background-tool"
    local hc_src="$REPO_DIR/tools/hot-corner-tool/hot-corner-tool.swift"
    local hc_bin="$REPO_DIR/tools/hot-corner-tool/hot-corner-tool"
    local disp_src="$REPO_DIR/tools/display_ctl/display_ctl.swift"
    local disp_bin="$REPO_DIR/tools/display_ctl/display_ctl"
    local bt_src="$REPO_DIR/tools/bluetooth-tool/bluetooth-tool.swift"
    local bt_bin="$REPO_DIR/tools/bluetooth-tool/bluetooth-tool"
    local bt_plist="$REPO_DIR/tools/bluetooth-tool/Info.plist"
    local au_src="$REPO_DIR/tools/audio-tool/audio-tool.swift"
    local au_bin="$REPO_DIR/tools/audio-tool/audio-tool"
    local au_plist="$REPO_DIR/tools/audio-tool/Info.plist"

    # `if` blocks, not `(( ! DRY_RUN )) && ...`: under `set -e` a false `&&` as a function's
    # last status aborts the script in --dry-run.
    if command -v swiftc >/dev/null 2>&1; then
        if [[ -f "$bg_src" ]] && [[ ! -x "$bg_bin" || "$bg_src" -nt "$bg_bin" ]]; then
            print "🔨 Compiling native background-tool..."
            if (( ! DRY_RUN )); then swiftc -module-cache-path /tmp/mcache -O "$bg_src" -o "$bg_bin"; fi
        fi
        if [[ -f "$hc_src" ]] && [[ ! -x "$hc_bin" || "$hc_src" -nt "$hc_bin" ]]; then
            print "🔨 Compiling native hot-corner-tool..."
            if (( ! DRY_RUN )); then swiftc -module-cache-path /tmp/mcache -O "$hc_src" -o "$hc_bin"; fi
        fi
        if [[ -f "$disp_src" ]] && [[ ! -x "$disp_bin" || "$disp_src" -nt "$disp_bin" ]]; then
            print "🔨 Compiling native display_ctl..."
            if (( ! DRY_RUN )); then swiftc -module-cache-path /tmp/mcache -O "$disp_src" -o "$disp_bin"; fi
        fi
        # Info.plist is embedded (NSBluetoothAlwaysUsageDescription) so the binary can get its own TCC prompt.
        if [[ -f "$bt_src" ]] && [[ ! -x "$bt_bin" || "$bt_src" -nt "$bt_bin" || "$bt_plist" -nt "$bt_bin" ]]; then
            print "🔨 Compiling native bluetooth-tool (embedding Info.plist)..."
            if (( ! DRY_RUN )); then
                swiftc -module-cache-path /tmp/mcache -O "$bt_src" -o "$bt_bin" \
                    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "$bt_plist"
            fi
        fi
        # Info.plist is embedded (NSMicrophoneUsageDescription) so the binary can get its own TCC prompt.
        if [[ -f "$au_src" ]] && [[ ! -x "$au_bin" || "$au_src" -nt "$au_bin" || "$au_plist" -nt "$au_bin" ]]; then
            print "🔨 Compiling native audio-tool (embedding Info.plist)..."
            if (( ! DRY_RUN )); then
                swiftc -module-cache-path /tmp/mcache -O "$au_src" -o "$au_bin" \
                    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "$au_plist"
            fi
        fi
    else
        print "⚠️  swiftc not found, skipping Swift compilation."
    fi
}

install_symlink() {
    local src="$1"
    local dest="$2"

    if (( UNINSTALL )); then
        if [[ -L "$dest" || -f "$dest" ]]; then
            if (( DRY_RUN )); then
                print "  🗑  [dry-run] Remove $dest"
            else
                print "  🗑  Removing $dest"
                rm -f "$dest"
            fi
        fi
        return 0
    fi

    if (( DRY_RUN )); then
        print "  🔗 [dry-run] Link $src -> $dest"
    else
        mkdir -p "${dest:h}"
        ln -sf "$src" "$dest"
        print "  🔗 Linked $dest -> $src"
    fi
}

# Tools to link: <source_path>:<dest_name>
typeset -a tools
tools=(
    "$REPO_DIR/tools/airplay-tool/airplay-tool:airplay-tool"
    "$REPO_DIR/tools/audio-tool/audio-tool:audio-tool"
    "$REPO_DIR/tools/background-tool/background-tool:background-tool"
    "$REPO_DIR/tools/bluetooth-tool/bluetooth-tool:bluetooth-tool"
    "$REPO_DIR/tools/display_ctl/display_ctl:display-tool"
    "$REPO_DIR/tools/hot-corner-tool/hot-corner-tool:hot-corner-tool"
    "$REPO_DIR/clipboard-network:clipboard-tool"
)

# Completions to link: <source_path>:<dest_name>
typeset -a completions
completions=(
    "$REPO_DIR/tools/airplay-tool/_airplay-tool:_airplay-tool"
    "$REPO_DIR/tools/audio-tool/_audio-tool:_audio-tool"
    "$REPO_DIR/tools/background-tool/_background-tool:_background-tool"
    "$REPO_DIR/tools/bluetooth-tool/_bluetooth-tool:_bluetooth-tool"
    "$REPO_DIR/tools/display_ctl/_display_ctl:_display-tool"
    "$REPO_DIR/tools/hot-corner-tool/_hot-corner-tool:_hot-corner-tool"
    "$REPO_DIR/_clipboard-network:_clipboard-tool"
)

# Obsolete legacy definitions to remove if present
typeset -a legacy_tools
legacy_tools=(
    "$BIN_DIR/background-set"
    "$BIN_DIR/background_set"
    "$BIN_DIR/display_ctl"
    "$BIN_DIR/airplay"
    "$BIN_DIR/airplay_connect"
    "$BIN_DIR/bluetooth"
    "$BIN_DIR/audio"
    "$BIN_DIR/clipboard-network"
)

typeset -a legacy_completions
legacy_completions=(
    "$COMP_DIR/_background-set"
    "$COMP_DIR/_background_set"
    "$COMP_DIR/_display_ctl"
    "$COMP_DIR/_airplay"
    "$COMP_DIR/_airplay_connect"
    "$COMP_DIR/_bluetooth"
    "$COMP_DIR/_audio"
    "$COMP_DIR/_clipboard-network"
)

clean_legacy() {
    local target="$1"
    if [[ -L "$target" ]]; then
        local dest="$(readlink "$target" 2>/dev/null || true)"
        if [[ "$dest" == "$REPO_DIR"* ]]; then
            if (( DRY_RUN )); then
                print "  🧹 [dry-run] Remove legacy $target"
            else
                print "  🧹 Removed legacy $target"
                rm -f "$target"
            fi
        fi
    fi
}

# Processes running any of the tools right now, as "pid command", one per line.
# Matched on the tool's own name at the end of a path or command word, so
# bluetoothd and AirPlayXPCHelper aren't mistaken for bluetooth and airplay, and
# only as the command itself or the script a shell/Python/osascript runs: anywhere in
# the line, a short name like `audio` matched arguments (".../audio file").
tools_in_use() {
    local -a names
    local entry
    for entry in "${tools[@]}"; do
        names+=("${entry##*:}" "${${entry%%:*}:t}")
    done
    local -a uniq=(${(u)names})
    local interp='([^ ]*/)?(sh|bash|zsh|dash|[Pp]ython[0-9.]*|osascript)( +-[^ ]+)* +'
    local pattern="^ *[0-9]+ (${interp})?([^ ]*/)?(${(j:|:)uniq})( |\$)"
    ps -axo pid=,command= | awk -v me=$$ '$1 != me' | grep -E "$pattern" | grep -v -E "scripts/install.sh" || true
}

if (( UNINSTALL )); then
    print "Uninstalling macOS GUI tools..."
else
    running="$(tools_in_use)"
    if [[ -n "$running" ]] && (( ! FORCE )); then
        if (( DRY_RUN )); then
            print "⚠️  [dry-run] would refuse: these are running one of the tools:"
            print -- "$running" | sed 's/^/    /'
        else
            print -u2 "Not installing while these are running one of the tools:"
            print -- "$running" | sed 's/^/    /' >&2
            print -u2 "Wait for them (or stop them), or rerun with --force."
            exit 3
        fi
    fi
    print "Installing macOS GUI tools..."
    compile_tools
fi

print "\nBinaries (to $BIN_DIR):"
for item in "${legacy_tools[@]}"; do
    clean_legacy "$item"
done
for entry in "${tools[@]}"; do
    src="${entry%%:*}"
    name="${entry##*:}"
    if [[ -f "$src" || -x "$src" ]]; then
        install_symlink "$src" "$BIN_DIR/$name"
    fi
done

print "\nCompletions (to $COMP_DIR):"
for item in "${legacy_completions[@]}"; do
    clean_legacy "$item"
done
for entry in "${completions[@]}"; do
    src="${entry%%:*}"
    name="${entry##*:}"
    if [[ -f "$src" ]]; then
        install_symlink "$src" "$COMP_DIR/$name"
    fi
done

if (( ! UNINSTALL )); then
    print "\n🎉 Installation complete!"
    print "\nEnsure the following is in your ~/.zshrc:"
    print "--------------------------------------------------------"
    print '  # macOS GUI Tools PATH & Completions'
    print '  export PATH="$HOME/.local/bin:$PATH"'
    print '  fpath=(~/.zsh/completions $fpath)'
    print '  autoload -Uz compinit && compinit'
    print "--------------------------------------------------------"
fi
