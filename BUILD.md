# NEW Requirements
Moving forward, all scripts must adhere to the strict, shared list of requirements: 
- **ALL COMMANDS MUST ALWAYS SHIP WITH -h/--help/help | json/-j for output, v/verbose if the script warrants it (haven't seen yet for our single-purpose work here)**
- **ALL COMMANDS AND NESTED COMMANDS AS WELL AS ALL OF THEIR RESPECTIVE OPTIONS MUST BE PLACED IN HELP/DOCS WITH A ~TERSE EXPLANATION OF HOW TO USE.**
- **ALL COMMANDS MUST SHIP WITH ZSH AUTOCOMPLETE THAT IS PLACED IN ZSH AUTOCOMPLETIONS ON INSTALL.**
- **ALL COMMANDS MUST SHARE NAMING CONVENTIONS, DESIGN/USAGE PATTERNS (similar flows, menus, options, nice-to-have extras that ALL SCRIPTS ARE BEING REQUIRED TO DESIGN, E.G.: - IN-BUILT BASIC ROTATING WALLPAPERS - FULLY SHELL HARDENED FOR PIPING, SUBSHELLING, REDIRECTION, ETC.**

---

Read the following concerns in each area of the project with its background, state and goal/planned next state. Note that no naming/paradigm has been landed upon yet and is open for architectural conversation.

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
