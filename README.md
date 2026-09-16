# screendump

Convert a screenshot into a structured, text-based UI representation.

It takes an image, runs OCR (tesseract) plus basic computer vision
(OpenCV: edge/contour detection, region classification), merges the results
into a layout tree, and renders an approximate Markdown/ASCII sketch of the
screen: what elements exist and roughly where they are.

```
┌─ WINDOW ─────────────────────────────────────────────────────────────┐
├──────────────────────────────────────────────────────────────────────┤
│  Firefox                                                              │
│  [icon]  [icon]  [input] https://example.com                         │
│  ────────────────────────────────────────────────────────────────     │
│                       Example Website                                 │
│                      [input] Search                                   │
│                      [button] Login   [button] Settings               │
│┌─ SIDEBAR ─────────┐ ┌─ MAIN ──────────────────────────────────────┐  │
││  Files            │ │  Welcome back                               │  │
│└───────────────────┘ └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

Regions are labeled frames (`WINDOW`, `SIDEBAR`, `MAIN`, `TERMINAL`,
`TABBAR`, `STATUS`...), widgets carry their type tag (`[button]`,
`[input]`, `[tab]`, `[heading]`, `[icon]`, `[box]`), and sibling panels are
laid out side by side so horizontal relationships survive.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- `tesseract-ocr` (system binary)
- A screen capture tool (e.g., `grim`, `scrot`, `maim`, `import`, or GNOME portal)

```bash
# Install system binary for OCR
# Debian / Ubuntu
sudo apt-get install -y tesseract-ocr
# Arch Linux
sudo pacman -S tesseract
```

## Setup

Sync the virtual environment and install all dependencies using `uv`:

```bash
uv sync
```

## Usage

Run commands with `uv run`:

```bash
# screendump: capture screen or read image to 2D ASCII/structured layout
uv run screendump screenshot.png
uv run screendump -s                         # capture the screen
uv run screendump                            # same: capture screen if no image
uv run screendump -j screenshot.png          # structured JSON with windows array
uv run screendump screenshot.png --ascii
uv run screendump screenshot.png --psm 6 --min-conf 40

# locater: precise pixel search by window or region + text/color/shape hints
uv run locater screenshot.png --window W1 --text "pytest" -j
uv run locater screenshot.png --window W2 --text "Google" -j
uv run locater screenshot.png --region 200,150,450,250 --text "Search" -j
uv run locater screenshot.png --color "#FF7A00" --shape circle -j
```

### Multi-Window Support (v2)

When multiple applications share the screen (e.g. tiling window managers, snapped windows, side-by-side apps):
1. **Seam & Boundary Detection**: `screendump` detects continuous vertical and horizontal split dividers and top/bottom system status bars.
2. **Strict Text Isolation**: OCR words never bleed across window boundaries.
3. **2D Side-by-Side Rendering**: Sibling root windows are rendered side-by-side in parallel ASCII columns with aligned bottom borders.
4. **Window Scoping**: Every window receives an ID (`[W1]`, `[W2]`) and inferred title (`Terminal`, `Browser`). `locater` can target windows directly via `--window W1` or `--window <title>`.

### Agent Acceleration & Optimistic Automation (v3)

1. **`visiond` Background Daemon**:
   Keeps OpenCV, OCR pipelines, and Wayland automation warm in memory. Eliminates python startup and module import latency:
   ```bash
   uv run visiond start          # launch background daemon
   uv run visiond status         # check daemon health
   uv run visiond stop           # stop background daemon
   ```
   When `visiond` is running, `screendump` and `locater` automatically route through its fast Unix domain socket.

2. **Single-Round-Trip Act + Perceive**:
   Eliminates separate click and screen refresh cycles:
   ```bash
   # Match button, click it, settle, and return updated UI layout in one command
   uv run locater -s --window W2 --text "Start a post" --click --dump

   # Match input field, click, type text, and verify
   uv run locater -s --window W2 --text "Search" --type "python" --verify --dump
   ```

3. **Input Verification via Visual Diffing**:
   Detects whether keystrokes were dropped or accepted by checking grayscale pixel changes in the target field:
   ```bash
   kbm type --verify "Hello world"
   ```

4. **Fast OCR & Downscaled Layout (`--fast`)**:
   Downsamples image by 2x for global text parsing, cutting OCR latency by 3x–5x while preserving UI structure:
   ```bash
   uv run screendump -s --fast
   uv run locater -s --window W2 --text "Submit" --fast
   ```

### Unified Autonomous UI Stack (v5)

Version 5 unifies the strengths of all previous iterations into a cohesive, high-performance toolkit:

1. **Window-Scoped Perception (`screendump --window <ID_OR_NAME>`)**:
   Instead of dumping the entire 1080p display, `screendump` can crop directly to a specific window (`W1`, `W2`, `Browser`, `Terminal`), accelerating capture time from 17s down to **3.3s (5.2× speedup)**:
   ```bash
   uv run screendump -j --window W3               # dumps only browser window in ~3.3s
   uv run screendump --window W1                  # renders ASCII layout of terminal alone
   ```

2. **Adaptive Mode Selection (`--mode auto|targeted|fast|full`)**:
   - `auto` *(default)*: Smart combination of window partition detection, 0 line bleeding, and targeted box OCR on actionable elements.
   - `targeted`: Pure OpenCV candidate box OCR (fastest full-screen dump at 7.5s, 65% leaner payload).
   - `fast`: 2× downsampled global sweep for instant structural layout.
   - `full`: High-resolution deep OCR pass.

3. **Window-Scoped Spatial Queries**:
   Combines `--window` scoping with spatial relational queries (`--right-of`, `--below`, `--left-of`, `--above`). Restricts candidates strictly inside the target window, eliminating false matches in neighboring apps and accelerating query latency from 16.3s to **9.2s**:
   ```bash
   uv run locater -s --window W3 --right-of "Hackathon" --click
   ```

4. **Zero-Turn Interaction & Fast Entry (`--type-and-enter`)**:
   Enables one-shot URL navigation, search box querying, and terminal execution:
   ```bash
   # Match search box, click, type query, hit enter, and return updated UI in 1 turn
   uv run locater -s --window W3 --text "Search" --type-and-enter "google search" --dump
   ```

5. **Bidirectional Actuate + Perceive**:
   Both `screendump` and `locater` support full action pipelines with visual diff verification (`--verify`):
   ```bash
   uv run screendump -s --click 500 300 --settle 200 -j
   ```


Options:

| flag            | meaning                                        |
|-----------------|------------------------------------------------|
| `-s/--screenshot` | capture the screen instead of reading an image |
| `-j/--json`     | structured JSON with `windows`, `regions`, `elements` |
| `--ascii`       | pure ASCII box characters (`+-|`)              |
| `--psm`         | tesseract page segmentation mode (default 11)  |
| `--min-conf`    | drop OCR words below this confidence (default 30) |
| `--out FILE`    | write output to a file                         |
| `--debug PFX`   | dump detected regions JSON + annotated image   |

`-j` emits `screen` dimensions, `windows` (each with `id`, `title`, `bbox`, `elements`),
`regions` (nested labeled containers with name + bounding box) and `elements` (each with
`type`, `text`, `bbox` and OCR `confidence`) — a compact machine-readable view for LLM tooling.
OCR `confidence`) — a compact machine-readable view for LLM tooling.

The output width always fills the terminal width. For screen capture the
first available tool is used: `gnome-screenshot`, `grim`, `spectacle`,
`scrot`, `maim`, ImageMagick `import`, or finally a GNOME capture through
`xdg-desktop-portal` (no install needed; the shot is also saved to
`~/Pictures/` by GNOME, as usual). Set `SCREENDUMP_CAPTURE_CMD` to a
command with `{out}` as the output path to force a specific tool.

## How it works

1. **vision.py** — Canny edges → dilation → contours → bounding rects.
   Rectangles are classified as window / panel / button / input / icon by
   their size and interior fill ratio. Separator lines are found with
   morphological opening.
2. **ocr.py** — runs the `tesseract` CLI (TSV output), filters by
   confidence, and groups words into text lines with bounding boxes.
3. **layout.py** — each OCR line is attached to the smallest containing
   region; regions form a tree by containment; children are sorted into
   rows by y-overlap.
4. **semantic.py** — classifies containers (sidebar / main / terminal /
   tabbar / toolbar / statusbar / menu / form) and widgets (tab / heading /
   ...) from geometry, interior fill and content; serializes the tree to
   JSON (`-j`).
5. **render.py** — walks the classified tree: labeled frames, tagged
   widgets (`[button] Login`), summary chrome lines (`TABS:`, `STATUS:`),
   side-by-side panels, positioned proportionally to the original image.

## Testing
 
Run tests across the entire test suite with `uv`:
 
```bash
uv run pytest
```
 
Or test on a generated sample:
 
```bash
uv run python locater/tests/make_sample.py sample.png
uv run screendump sample.png
```