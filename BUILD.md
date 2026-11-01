# RULES

Moving forward, all scripts must adhere to the strict, shared list of requirements. These are rigid requirements and will remain as such indefinitely - e.g. the repo will have one commit author 

- **ALL COMMANDS MUST ALWAYS SHIP WITH -h/--help/help | json/-j for output, v/verbose if the script warrants it (haven't seen yet for our single-purpose work here)**
- **ALL COMMANDS AND NESTED COMMANDS AS WELL AS ALL OF THEIR RESPECTIVE OPTIONS MUST BE PLACED IN HELP/DOCS WITH A ~TERSE EXPLANATION OF HOW TO USE.**
- **ALL COMMANDS MUST SHIP WITH ZSH AUTOCOMPLETE THAT IS PLACED IN ZSH AUTOCOMPLETIONS ON INSTALL.**
- **ALL COMMANDS MUST SHARE NAMING CONVENTIONS, DESIGN/USAGE PATTERNS (similar flows, menus, options, nice-to-have extras that ALL SCRIPTS ARE BEING REQUIRED TO DESIGN, E.G.: - IN-BUILT BASIC ROTATING WALLPAPERS - FULLY SHELL HARDENED FOR PIPING, SUBSHELLING, REDIRECTION, ETC.**
- **FOLLOW GIT RULES RELIGIOUSLY:**
 - **ALL COMMITS MUST BE SINGLE-LINE AND <120 CHARACTERS. COMMITS ARE AT-GLANCE HEADERS AND MORE IMPORTANTLY ARE NEVER READ AGAIN AFTER REVIEW FROM THE HUMAN PERSPECTIVE FOR A MYRIAD OF REASONS, MANY OF WHICH SHOULD BE APPARENT. INSTEAD, KEEP COMMITS SUCCINCT AND INSTEAD SPEND THE GAINED TIME RE-READING THE ~BASICALLY SAME~ CONTENT WE MAINTAIN AND ALL REFERENCE IN DOCS. MAKE A FEW SENTENCES OF UPDATES THERE IF THE WORK AND CHANGES JUSTIFY THE ADDITIONAL TIME.**
 - **ALL COMMITS MUST BE AUTHORED/SIGNED BY THE ACTING USER BY USING THE CONFIG FOUND AT $HOME/.gitconfig OR /Users/zooey/.gitconfig**.
 - **RULE WAS SET IN STONE AFTER MULTIPLE INDEPENDENT AGENT OR OTHERWISE AUTOMATED ACTOR INTRODUCED MULTIPLE PRODUCTION-FACING FIRES THAT ENTIRE TEAM HAD TO DROP TO MITIGATE.**
 - **BUILDS SHOULD TAKE PLACE IN AN INDEPENDENT WORKING TREE/BRANCH TO AVOID DISTRACTION/CONFLICTS.**
 - **MAINTAIN AND REFERENCE DOCUMENTATION. COMMIT MESSAGES 

Note: A helper alias for git commit is underwayif branch/worktree individual commits would be helpful during builds, something in similar shape to:
 `git -c "user.name=$GITUSER" -c "user.email=$GITEMAIL" commit ..`
Welcome requests/def files to placed/added to e.g. .zshrc/.zshpriv-local.

---

Read the following concerns in each area of the project with its background, state and goal/planned next state. Note that no naming/paradigm has been landed upon yet and is open for architectural conversation.

Spend time comprehensively refining the two existing airplay files into a complete and well-packaged tool. In repackaging/cleaning up and effort being done to unify and clean the repo, shorten the command to airplay-tool/util or something catching/memorable and clearly a macOS/linux thing to remember; airplay itself is good to me but may get confused with system tools/airplay itself. Personally against losing airplay though not something like airpal or something forgettable. The other in-progress tool being the background tool; I do have a note about aliases/naming in a moment. there is much to be done in cleaning up and refinement of the repo state and any future roadmap items.

- Adding a basic toggle for calling `airplay toggle` - defined was:
 - If currently casting over airplay, toggle stops casting.
 (Proposed to me, proposing to you here)
 - If not currently casting, toggle would look at the install's state/vals and use the last-airplay host to connect to once they appear. If they do not appear after Ns ~5-10? stderr perhaps? Note - this may be overstepping the package bounds e.g. remain dumb scripts, but we already do some rotation/random bg sets. This might be nice to have, but if too kitschy (I can see it) we can defer. 
  - Checking: The docs/help show shorthand/nicknames for controls. I could 100% see a. State/aliases/etc. filet the installed .local if we started to provide this (honestly, very useful and helpful daily) personal identification to legitimate static identifiers exactly like here. 
    I have an ending remark about naming/aliases for the repo but I could persuaded it's worth a single state/json file to have some niceties but do realize it's nothing more than a paper trial easily lost

- Feels noteworthy to mention: airplay status command is quite quick. Menu bar dropdown opens nearly as finishing pressing enter. The slowest hang is on list - not a surprise at first glance, but the hang is front-loaded - call list, nothing happens 1-2s, then then GUI dropdown opens. I realize this is probably a requirement of network disc etc., but honestly I would so prefer us to be using a swift/other more graceful control for Airplay here. I'm honestly reluctant to put it out there since it's such a crude automation. 
  The below first roadmap item states a fundamental desire goal to hook into macOS and I agree wholeheartedly it is a new core philosophy to examine and rule to enforce, that is, if there is a easily controlled and saved default, configuration, installation or otherwise, many of our scripts would be missing significant controls otherwise. 

**IMPORTANT**: Following the new rule to include any and all relevant/useful controls for airplay, I suggest perhaps a default/.. other script to disable/enable bluetooth, view logs (like airplay) would be a good step forward there. 

- 2026-10-26 (built, branch `bluetooth-tool`): `tools/bluetooth-tool` (Swift/IOBluetooth, also `bluetooth`) with list, status, power [on|off|toggle], connect/disconnect/toggle, wait, watch, logs, alias; `-j` on all, zsh completion, wired into install.sh. Changes are read-back transactions (exit 4 if the state never reads back), and `power off` refuses (exit 3) while a Bluetooth keyboard/mouse/trackpad is connected unless `--force`. Power set uses the private blueutil API and is not yet run on hardware; see tools/bluetooth-tool/README.md. Also fixed install.sh `--uninstall --dry-run`, which used to abort after the first entry.
- 2026-10-26: install.sh refuses (exit 3) while any of the tools is running, listing the processes, since a Swift rebuild rewrites the binary in place; `--force` skips it. Asked for so a reinstall is safe whenever nothing is using the tools.
- 2026-11-01 (built, branch `audio-tool`): `tools/audio-tool` (Swift/CoreAudio, also `audio`) with list, status, default input|output|system, volume, mute, play, record, level, watch, alias; `-j` on all, zsh completion, wired into install.sh. It exists because `say -a '?'` doesn't list the Scarlett: play/record/level run a raw IOProc on the named device (resampled with AVAudioConverter, defaults untouched), and record writes a WAV that stays valid if killed. Verified on hardware: a -30 dBFS tone played to Scarlett output 1 read back at -30.09 dBFS on loopback 3 only, and a tone to the (non-default) laptop speakers reached the MacBook mic 23 dB above the room. The set paths of default/volume/mute are not yet run on hardware (reads and refusals only); see tools/audio-tool/README.md. Also narrowed install.sh's in-use check to the command itself or an interpreter's script, since the short `audio` alias matched arguments in unrelated command lines.

# Roadmap

- (High value and likely everywhere) Frankly, nearly any user-driven action on macOS could be useful in a script/automation/tool given enough time. I am talking about both having a kind of a single or handful of semi-glossary acting, no-frills scripts that just surface all of the entirely user-facing/open game controls/tooling that are either walled or hidden behind osascript's rigidity/difficulties, 
 - Dismissing the top-right notification area for lingering alerts via a general macOS .. notifications/alerts CLI tool
 - I am looking into setting up Time Machine at a later date and e.g. will be building a script for that for audit/checking and iCloud/drive capacities, etc.
 - I will surely also purchase more Macs in the future - I have a dot file repo for defaults, build scripts etc. that is decent but has a single defaults "file" that is intended to be ran on install but A. Drifted a ton - hard to keep a private repo up to date etc.; but it also misses some frankly fundamental OS settings that I know are recordable/maintainable in some fashion - so doable in the agentic age.

- I'd like to look into network/BT usage - I recently found out all three of my old iPhones are still appearing in one (only one) of my MacBooks' finder windows under network, all mounted with eject buttons. I know two are a 5S and 6 I used for network cameras (OBS Camera app, let you control the iOS zoom/focus/exposure etc. through OBS and over websocket - was quite cool, tangent but where that thought is that is at)
 - I primarily would use my relatively-new old phone to do something with the camera. I recall being able to capture photos from a trusted plugged phone (maybe.) I know all primary phones have that though 24/7. Need to look into my other laptop being able to connect/control over BT..

---

Old/priors states and disc. from moving from old repo to new:

## Tools
Needs updating, the general jist at the start:

- background setting
 - targetable displays AND spaces by index/id would be great (perhaps it would set all spaces/all displays by default? targetting by ID is granular setting)
 - full options like stretch/zoom/modes available (with a hex # if they still allow you to place an image w/o zoom/stretch with a background color - doubtful though)
 - the scheduled/rotation stuff could honestly be either marked as "get another tool to do this" maybe?
- airplay/display mirroring 
- clipboard over network (needs windows/setup)
- application/window controls
 - raising application works OK, more controls like minimize/maximize/..* any and all controls we could receive would be great
 - (posit) windows management is solved by other tools AND macOS itself now (tmp/max_options.png) - could those be somehow roped into our controls? i would love to say e.g. move spotify to left 1/2 horizontal of screen.
- screenshot tool
 - similar display/space targeting like background would be nice, other screenshot builtin tooling
 - video capture
- 'automation' tool - general mouse movement/visual identification
 - would like to migrate this from a job-site application tool to more of a general utility
 - (perhaps for the future templates stuff - that could go in system config or something. low priority, but added to gitignore for now)
- ...more

## Installation of tools
Currently all that must be done is adding aliases and copying autocompletes. 

A future nice-to-have would be doing all of that in a single script - not sure how to do that though, like adding these scripts to path + zsh autocompletes. It might be best to be documented + .. maybe scripted as possible? (This is new territory to me - making a package of tools that are easy to use/install and apply to lots of things.)

Another note about installs - I don't want to have some kind of global `macos-util` tool that shows all tools available.. but again I have never bundled a set of scripts for use like this. Need best practices/documentation for me moving forward - help with my script structure on background would help me too - that's the most-used script by far at this point

## Scratch
Lots of notes laying around, applescript reference scripts, etc.; all of what's in scripts/ is for reference and I wouldn't want published with the package. Still learning what we can get out osascript/macOS. It'll be a while until I want to push up to github - gotta rename that repo there. Until then I'll just back up locally

## Planning for restructuring / lightweight script suite with easy install
Current structure is a mess and needs some thought (the entire repo used to be automation for applying to job sites)
```mockup-000/sketch
README
scripts
  install.sh
tools
  background-set
  airplay-tool
  .. rest
  ..
```
