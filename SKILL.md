---
name: visionmodelreplacer
description: Local, vision-model-free desktop UI perception and automation tool stack using screendump, locater, kbm, and visiond. Use whenever the user asks to inspect the desktop, parse UI layouts into text/ASCII/JSON, find pixel coordinates of UI elements (by text, color, or shape), or automate keyboard and mouse actions without using vision models.
---

# VisionModelReplacer Skill (v5 - Unified Autonomous UI Stack)

A high-performance, local, vision-model-free desktop UI automation stack located at `/home/nabeel/Documents/projects/visionmodelreplacer-v5`. Converts screen pixels into structured semantic text/JSON and provides coordinate-level actuation with visual verification.

---

## Core Tool Suite

| Tool | Primary Role | Key Capabilities |
|---|---|---|
| **`screendump`** | Screen Perception | Multi-window detection, 2D columns (0 bleed), lazy targeted box OCR, window scoping (`--window W1`), and pre-dump actuation. |
| **`locater`** | Target Pinpointer & Actuator | Pixel-exact coordinates by text, color, shape, window scoping, or spatial relation (`--right-of`, `--below`). Supports `--type-and-enter`, `--verify`, and `--dump`. |
| **`visiond`** | Background Daemon | Keeps OpenCV, OCR, and Wayland automation warm in RAM. Sub-100ms IPC via `/run/user/<uid>/visiond.sock`. |
| **`kbm`** | Wayland Input Driver | Thin wrapper over `ydotool` (mouse move/click) and `vtype` (uinput virtual keyboard) with `--verify` diff checks. |

---

## Quickstart & Session Initialization

### 1. Ensure Input Services Are Running
```bash
# Start ydotool system daemon if inactive (required by kbm)
sudo systemctl start ydotool

# Verify pointer tracking
kbm pos
```

### 2. Start the Acceleration Daemon (Recommended)
```bash
# In the v5 project directory:
cd /home/nabeel/Documents/projects/visionmodelreplacer-v5
uv run visiond start &

# Check daemon health
uv run visiond status
```

---

## Standard Workflows & Command Recipes

### A. Perceiving Desktop Layout (v5 Modes & Scoping)
```bash
# Smart Auto Mode (Default): multi-window boundary aware, zero bleed, high-signal widget OCR
uv run screendump -s -j

# Window-Scoped Dump (Fastest: ~3.3s on 1080p, 5x speedup):
uv run screendump -s --window W3 -j            # Dumps only browser window W3
uv run screendump -s --window W1               # Dumps only terminal window W1

# Fast layout sweep (2x downsampled OCR, 3x-5x faster):
uv run screendump -s --mode fast

# Targeted layout parse (only OCRs interactive widgets, 65% leaner payload):
uv run screendump -s --mode targeted

# Save ASCII wireframe to file:
uv run screendump -s --out /tmp/screen_layout.txt
```

### B. Locating UI Elements & Spatial Queries
```bash
# Window-scoped text search (2x-10x faster):
uv run locater -s --window W3 --text "Hackathon" -j

# Window-scoped spatial query (restricts candidates to target window, 1.8x faster):
uv run locater -s --window W3 --right-of "Email" --shape rect --type "me@example.com" --verify

# Spatial query: find button or dropdown below a section header:
uv run locater -s --below "Password" --type "pass123"

# Auto-scroll if element is off-screen (scrolls mouse wheel up to N times):
uv run locater -s --text "Save Changes" --scroll-until 4 --scroll-dy -5 --click

# Search by button shape and brand color:
uv run locater -s --window W3 --shape rect --color "#0a66c2" --tol 40 -j
```

### C. Zero-Turn Interaction & Single-Turn Act + Perceive
Instead of locating, clicking, typing, and requesting a fresh screen dump in separate round-trips, chain them in one command:
```bash
# Type into search box, press enter, and get updated layout immediately (1-turn interaction):
uv run locater -s --window W3 --text "Search" --type-and-enter "google search" --dump

# Find button, click it, settle 200ms, and return updated layout:
uv run locater -s --window W3 --text "Start a post" --click --dump

# Direct pre-action in screendump: click coordinates and dump updated screen:
uv run screendump -s --click 1150 277 --settle 200 -j
```

### D. Direct Keystrokes & Verified Typing
```bash
# Move cursor & click:
kbm move 1150 277
kbm click 1150 277

# Keyboard shortcuts:
kbm key ctrl+l       # focus browser address bar
kbm key ctrl+t       # new browser tab
kbm key enter        # submit
kbm key esc          # dismiss modal

# Type text with automatic visual diff verification (warns if no pixels changed):
kbm type --verify "https://github.com"
```

---

## Practical Performance Guidelines

1. **Always Scope by Window When Possible (`--window W<id>`)**:
   - Running full-screen 1080p OCR on the whole desktop takes ~16s under load.
   - Using `--window W3` drops latency to **3.3s** (**5.2× speedup**) and prevents false positive matches from adjacent apps.

2. **Heavy Background CPU Load (e.g. Scanners / OpenVAS)**:
   - When background processes consume significant CPU, pass `--window` or `--mode fast` / `--mode targeted` to minimize CPU cycles.

3. **Optimistic Pipelining**:
   - Use `--type-and-enter` for searches, address bars, and forms to eliminate keystroke delays.
   - Use `--click --dump` or `--type --verify --dump` to receive new state in the same conversational turn.
