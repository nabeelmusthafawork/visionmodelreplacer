---
name: visionmodelreplacer
description: Universal, vision-model-free global desktop UI perception and automation tool stack using screendump, locater, kbm, visiond, and flow. Use whenever the user asks to inspect the desktop, parse UI layouts into text/ASCII/JSON, find pixel coordinates of UI elements (by text, color, or shape), or automate keyboard and mouse actions across any native desktop application, terminal, IDE, file manager, or browser without using vision models.
---

# VisionModelReplacer Skill (v6 - Universal Global Desktop Controller)

A lightning-fast, local, vision-model-free desktop UI automation stack located at `/home/nabeel/Documents/projects/visionmodelreplacer`. Controls **any native Linux desktop software** (IDEs, terminals, file managers, system settings, dialogs, media players) as well as web browsers, completely bypassing the need for cloud vision models.

---

## Core Tool Suite

| Tool | Primary Role | Key Capabilities |
|---|---|---|
| **`flow`** | **Universal Action Pipeline** | **Single-turn multi-step macro runner**. Executes chained app launching, window focus, clicks, right-clicks, drag-and-drop, and typing in 1 request. |
| **`screendump`** | Screen Perception | Multi-window boundary detection, 2D columns (0 bleed), sub-region crop OCR (0.59s), and window scoping (`--window W1`). |
| **`locater`** | Target Pinpointer & Actuator | Pixel-exact coordinates by text, color, shape, window scoping, or spatial relation (`--right-of`, `--below`). Supports right/double click. |
| **`visiond`** | Background Daemon | Keeps OpenCV, OCR, and Wayland automation warm in RAM. Sub-100ms IPC via `/run/user/<uid>/visiond.sock`. |
| **`kbm`** | Wayland Input Driver | Thin wrapper over `ydotool` (mouse move/click/drag) and `vtype` (uinput virtual keyboard) with `--verify` diff checks. |

---

## Universal Desktop Actions Reference (`flow`)

The `flow` pipeline supports full desktop interaction across arbitrary applications:

| Action Specifier | Description | Real Desktop Example |
| :--- | :--- | :--- |
| `launch:<command>` | Launches desktop application detached in background | `launch:thunar`, `launch:alacritty`, `launch:code` |
| `focus:<query>` | Focuses window by ID (`W1`, `W2`) or title substring | `focus:OpenCode`, `focus:Terminal`, `focus:W1` |
| `click:<text/coords>` | Left clicks target element or `x,y` | `click:text=Save`, `click:x=500,y=300` |
| `right_click:<text/coords>` | Right clicks to trigger desktop context menus | `right_click:text=File.txt`, `right_click:x=400,y=200` |
| `double_click:<text/coords>` | Double clicks to open folders, launch desktop icons | `double_click:text=Documents`, `double_click:x=200,y=150` |
| `drag:x1,y1->x2,y2` | Drags mouse from source to destination | `drag:200,150->500,400` (drag files or sliders) |
| `hotkey:<combo>` | Sends system hotkey combinations | `hotkey:ctrl+shift+p`, `hotkey:alt+f4`, `hotkey:super` |
| `type:<text>` | Types text (optional `enter=true`, `verify=true`) | `type:cargo build,enter=true`, `type:nano file.txt` |
| `wait:<text>` | Polls local frame until target text appears | `wait:text=Build finished,timeout=10.0` |
| `wait_change` | Polls local frame until screen pixels change | `wait_change:timeout=5.0` (waits for dialog to open) |
| `assert:<text>` | Verifies required text exists, fails flow if not | `assert:text=Success` |

---

## Standard Workflows & Command Recipes

### A. Controlling Native Desktop Software (IDEs, File Managers, Terminals)

```bash
# In the v6 project directory:
cd /home/nabeel/Documents/projects/visionmodelreplacer

# 1. Open VS Code / OpenCode command palette and run a command:
uv run flow \
  --step "focus:OpenCode" \
  --step "hotkey:ctrl+shift+p" \
  --step "wait:text=Type a command,timeout=2.0" \
  --step "type:File: Open Folder...,enter=true"

# 2. Launch File Manager, navigate directory, right-click file:
uv run flow \
  --step "launch:thunar /home/nabeel/Documents" \
  --step "wait:text=Documents,timeout=3.0" \
  --step "double_click:text=projects" \
  --step "wait:text=visionmodelreplacer,timeout=2.0" \
  --step "right_click:text=visionmodelreplacer" \
  --step "wait:text=Properties,timeout=2.0" \
  --step "click:text=Properties"

# 3. Launch Terminal & Run Shell Command:
uv run flow \
  --step "launch:alacritty" \
  --step "wait:text=nabeel@arch,timeout=2.0" \
  --step "type:uname -r && uptime,enter=true"
```

### B. High-Speed Sub-Region Perception (`screendump`)
```bash
# Window-Scoped Dump (0.59s - 3.3s on 1080p, 5x-20x speedup):
uv run screendump -s --window W3 -j            # Dumps only window W3
uv run screendump -s --window W1               # Dumps only terminal window W1

# Smart Auto Mode (Default): multi-window boundary aware, zero bleed, high-signal widget OCR
uv run screendump -s -j

# Fast layout sweep (2x downsampled OCR, 3x-5x faster):
uv run screendump -s --mode fast
```

### C. Pinpointing Native UI Elements & Spatial Queries (`locater`)
```bash
# Search within a specific window (W1, W2, or window title):
uv run locater -s --window W1 --text "bash" -j

# Spatial query: find an input box to the right of an "Email" label:
uv run locater -s --window W3 --right-of "Email" --shape rect --type "me@example.com" --verify

# Spatial query: find button or dropdown below a section header:
uv run locater -s --below "Password" --type "pass123"

# Auto-scroll if element is off-screen (scrolls mouse wheel up to N times):
uv run locater -s --text "Save Changes" --scroll-until 4 --scroll-dy -5 --click
```

---

## Why v6 Beats Cloud Vision Models

1. **Zero Turn Overhead**: Cloud vision models require conversational round-trips for every action. `v6 flow` executes the whole chain in **1 turn**.
2. **Sub-Second Crop Latency**: 20.5× speedup (0.59s vs 12.28s) by cropping regions before OCR.
3. **Dirty-Rectangle Detection**: 13 milliseconds (`cv2.absdiff`) to find exactly where UI changes occurred.
4. **Wayland & Linux Native**: Directly injects hardware events at the Linux `/dev/uinput` level, allowing full control of native Wayland and X11 apps.
5. **100% Privacy & Zero Cost**: Runs entirely local on the host CPU without any cloud APIs, token costs, or rate limits.
