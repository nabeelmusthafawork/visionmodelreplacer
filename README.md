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
- `tesseract-ocr` (binary)
- `python3-opencv`, `python3-numpy` (no pip needed; install via your distro)

```
apt-get install -y python3 python3-numpy python3-opencv python3-pil tesseract-ocr
```

## Usage

```
python3 -m screendump screenshot.png
python3 -m screendump -s          # capture the screen
python3 -m screendump             # same: no image means capture the screen
python3 -m screendump -j screenshot.png   # structured JSON
python3 -m screendump screenshot.png --ascii
python3 -m screendump screenshot.png --psm 6 --min-conf 40
```

Options:

| flag            | meaning                                        |
|-----------------|------------------------------------------------|
| `-s/--screenshot` | capture the screen instead of reading an image |
| `-j/--json`     | structured JSON instead of the ASCII map       |
| `--ascii`       | pure ASCII box characters (`+-|`)              |
| `--psm`         | tesseract page segmentation mode (default 11)  |
| `--min-conf`    | drop OCR words below this confidence (default 30) |
| `--out FILE`    | write output to a file                         |
| `--debug PFX`   | dump detected regions JSON + annotated image   |

`-j` emits `screen` dimensions, `regions` (nested labeled containers with
name + bounding box) and `elements` (each with `type`, `text`, `bbox` and
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

```
python3 tests/make_sample.py tests/sample.png
python3 -m screendump tests/sample.png
```