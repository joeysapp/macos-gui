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

usage() {
    cat <<EOF
Usage:
  ./scripts/install.sh [options]

Options:
  --bin-dir <dir>        Directory to install executables (default: $DEFAULT_BIN_DIR)
  --comp-dir <dir>       Directory to install completions (default: $DEFAULT_COMPLETION_DIR)
  --uninstall            Remove symlinked tools and completions
  -n, --dry-run          Show what would be installed/removed without doing it
  -h, --help             Show this help message

Description:
  Installs individual macOS GUI tools and their zsh autocompletions.
  Each tool remains an independent utility (no monolithic wrapper).
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
    local bg_src="$REPO_DIR/tools/background_set/background_set.swift"
    local bg_bin="$REPO_DIR/tools/background_set/background_set"
    local disp_src="$REPO_DIR/tools/display_ctl/display_ctl.swift"
    local disp_bin="$REPO_DIR/tools/display_ctl/display_ctl"

    if command -v swiftc >/dev/null 2>&1; then
        if [[ -f "$bg_src" ]] && [[ ! -x "$bg_bin" || "$bg_src" -nt "$bg_bin" ]]; then
            print "🔨 Compiling native background_set..."
            (( ! DRY_RUN )) && swiftc -O "$bg_src" -o "$bg_bin"
        fi
        if [[ -f "$disp_src" ]] && [[ ! -x "$disp_bin" || "$disp_src" -nt "$disp_bin" ]]; then
            print "🔨 Compiling native display_ctl..."
            (( ! DRY_RUN )) && swiftc -O "$disp_src" -o "$disp_bin"
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
            print "  🗑  Removing $dest"
            (( ! DRY_RUN )) && rm -f "$dest"
        fi
        return
    fi

    if (( DRY_RUN )); then
        print "  🔗 [dry-run] Link $src -> $dest"
    else
        mkdir -p "${dest:h}"
        ln -sf "$src" "$dest"
        print "  🔗 Linked $dest -> $src"
    fi
}

if (( UNINSTALL )); then
    print "Uninstalling macOS GUI tools..."
else
    print "Installing macOS GUI tools..."
    compile_tools
fi

# Tools to link: <source_path>:<dest_name>
typeset -a tools
tools=(
    "$REPO_DIR/tools/background_set/background_set:background_set"
    "$REPO_DIR/tools/display_ctl/display_ctl:display_ctl"
    "$REPO_DIR/tools/airplay/airplay_connect:airplay_connect"
    "$REPO_DIR/clipboard-network:clipboard-network"
)

# Completions to link: <source_path>:<dest_name>
typeset -a completions
completions=(
    "$REPO_DIR/tools/background_set/_background_set:_background_set"
    "$REPO_DIR/tools/display_ctl/_display_ctl:_display_ctl"
    "$REPO_DIR/tools/airplay/_airplay_connect:_airplay_connect"
    "$REPO_DIR/_clipboard-network:_clipboard-network"
)

print "\nBinaries (to $BIN_DIR):"
for entry in "${tools[@]}"; do
    src="${entry%%:*}"
    name="${entry##*:}"
    if [[ -f "$src" || -x "$src" ]]; then
        install_symlink "$src" "$BIN_DIR/$name"
    fi
done

print "\nCompletions (to $COMP_DIR):"
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
