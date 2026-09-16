---
name: visionmodelreplacer
description: Local, vision-model-free desktop UI perception and automation tool stack using screendump, locater, kbm, visiond, and flow. Use whenever the user asks to inspect the desktop, parse UI layouts into text/ASCII/JSON, find pixel coordinates of UI elements (by text, color, or shape), or automate keyboard and mouse actions without using vision models.
---

# VisionModelReplacer Skill (v6 - OmniFlow High-Speed Stack)

A lightning-fast, local, vision-model-free desktop UI automation stack located at `/home/nabeel/Documents/projects/visionmodelreplacer-v6`. Engineered to outperform cloud vision models in both execution speed and conversational turn efficiency.

---

## Core Tool Suite

| Tool | Primary Role | Key Capabilities |
|---|---|---|
| **`flow`** | **Autonomous Action Pipeline** | **Single-turn multi-step macro runner**. Executes chained navigation, local polling waits, clicks, and typing in 1 request. |
| **`screendump`** | Screen Perception | Multi-window detection, 2D columns (0 bleed), sub-region crop OCR (0.59s), and window scoping (`--window W1`). |
| **`locater`** | Target Pinpointer & Actuator | Pixel-exact coordinates by text, color, shape, window scoping, or spatial relation (`--right-of`, `--below`). |
| **`visiond`** | Background Daemon | Keeps OpenCV, OCR, and Wayland automation warm in RAM. Sub-100ms IPC via `/run/user/<uid>/visiond.sock`. |
| **`kbm`** | Wayland Input Driver | Thin wrapper over `ydotool` (mouse move/click) and `vtype` (uinput virtual keyboard) with `--verify` diff checks. |

---

## The Golden Rule for AI Agents: Always Prefer `flow` Over Turn-by-Turn Steps

Instead of spending 8–10 conversation turns clicking and waiting, dispatch an entire workflow in **ONE single command**:

```bash
# In the v6 project directory:
cd /home/nabeel/Documents/projects/visionmodelreplacer-v6

# 1-Turn End-to-End Workflow:
uv run flow \
  --window W3 \
  --step "nav:https://www.linkedin.com/feed/" \
  --step "wait:text=Start a post,timeout=6.0,interval=0.15" \
  --step "click:text=Start a post,settle=0.2" \
  --step "type:Autonomous workflow executed in v6!,verify=true" \
  --json
```

---

## Standard Workflows & Command Recipes

### A. Chained Multi-Step Flows (`flow`)
```bash
# URL navigation + search + submit:
uv run flow \
  --window W3 \
  --step "nav:https://google.com" \
  --step "wait:text=Search,timeout=5.0" \
  --step "click:text=Search" \
  --step "type:DeepMind Antigravity,enter=true" \
  --json

# Multi-step button clicks with polling wait:
uv run flow \
  --step "click:text=File" \
  --step "wait:text=Save As,timeout=2.0" \
  --step "click:text=Save As"
```

### B. High-Speed Sub-Region Perception (`screendump`)
```bash
# Window-Scoped Dump (0.59s - 3.3s on 1080p, 5x-20x speedup):
uv run screendump -s --window W3 -j            # Dumps only browser window W3
uv run screendump -s --window W1               # Dumps only terminal window W1

# Smart Auto Mode (Default): multi-window boundary aware, zero bleed, high-signal widget OCR
uv run screendump -s -j

# Fast layout sweep (2x downsampled OCR, 3x-5x faster):
uv run screendump -s --mode fast

# Targeted layout parse (only OCRs interactive widgets, 65% leaner payload):
uv run screendump -s --mode targeted
```

### C. Locating UI Elements & Spatial Relational Queries
```bash
# Window-scoped text search (2x-10x faster):
uv run locater -s --window W3 --text "Hackathon" -j

# Window-scoped spatial query (restricts candidates to target window):
uv run locater -s --window W3 --right-of "Email" --shape rect --type "me@example.com" --verify

# Spatial query: find button or dropdown below a section header:
uv run locater -s --below "Password" --type "pass123"

# Auto-scroll if element is off-screen (scrolls mouse wheel up to N times):
uv run locater -s --text "Save Changes" --scroll-until 4 --scroll-dy -5 --click
```

---

## Why v6 Beats Vision Models

1. **Zero Turn Overhead**: Cloud vision models require conversational round-trips for every action. `v6 flow` executes the whole chain in **1 turn**.
2. **Sub-Second Crop Latency**: 20.5× speedup (0.59s vs 12.28s) by cropping regions before OCR.
3. **Dirty-Rectangle Detection**: 13 milliseconds (`cv2.absdiff`) to find exactly where UI changes occurred.
4. **100% Privacy & Zero Cost**: Runs entirely local on the host CPU without any cloud APIs, token costs, or rate limits.
