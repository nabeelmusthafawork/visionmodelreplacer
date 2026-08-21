# Prompt: Build the `locater` tool

You are building a CLI tool called `locater` that complements the existing `screendump`
package. Read the code below first, then implement the tool. Do not ask questions —
all decisions are specified here. If something is underspecified, pick the most
reasonable option consistent with the existing `screendump` code style.

## Background: what already exists

Repo: `/home/nabeel/filesync-shared/dumbstuff` — a Python package `screendump`
(editable-installed). Relevant modules you MUST reuse:

- `screendump/capture.py` — `capture_screen()` returns PNG bytes of the whole display.
  Uses grim/scrot/maim/import/gnome portal, honors `SCREENDUMP_CAPTURE_CMD`.
- `screendump/ocr.py` — `ocr_words(png_bytes, psm, min_conf) -> list[Word]`,
  `group_lines(words) -> list[Line]`. Words/lines have `x, y, w, h, conf` and
  `right/bottom` properties. Raises `ocr.TesseractError` if tesseract is missing.
- `screendump/cli.py` — the CLI style to mirror: argparse, positional `image` path
  or `-s/--screenshot`, `-j/--json`, `--out`, `--debug PREFIX`. Image bytes are
  decoded with `cv2.imdecode`. Bboxes are pixel coords `[x, y, w, h]`.
- `pyproject.toml` — console script `screendump = "screendump.cli:main"`,
  `[tool.setuptools] packages = ["screendump"]`.

## Goal / workflow

An AI agent automates a desktop. It first runs `screendump` to get a text/ASCII
representation of the screen (including pixel bboxes of UI elements). Then it calls
`locater` to **squar down** a search: the AI provides a rough pixel region plus hints
(text, color, shape) about the element it wants, and `locater` crops the screenshot
to that region, looks for the described element, and returns **precise absolute pixel
coordinates** (center + bbox) of every good match, ranked.

Example: AI wants to click the Brave browser icon. It calls
`locater --region 50,850,400,1050 --color #FF7A00 --shape circle` and gets back the
icon's exact center so it can move the mouse and click.

## Location and packaging

Create a new package folder `locater/` next to `screendump/`:

- `locater/__init__.py`
- `locater/cli.py` — `main(argv=None) -> int`, like `screendump.cli`
- `locater/match.py` — detection logic (importable, unit-testable)

Update `pyproject.toml`:

```toml
[project.scripts]
screendump = "screendump.cli:main"
locater = "locater.cli:main"

[tool.setuptools]
packages = ["screendump", "locater"]
```

Then run `pip install -e /home/nabeel/filesync-shared/dumbstuff` so the `locater`
command becomes available. Keep dependencies at zero (like screendump — it relies on
system tesseract and imports cv2/numpy lazily). The `locater` module itself may
import cv2/numpy at top level only inside functions if that keeps import light;
follow `screendump` conventions (it imports cv2/numpy at module top — match that).

## CLI interface

```
usage: locater [image] [-s] [--region X1,Y1,X2,Y2] [--text TEXT] [--color HEX]
               [--shape {circle,square,rect,triangle}] [--tol TOL] [--fuzzy RATIO]
               [--psm PSM] [--min-conf MIN_CONF] [--max N] [-j] [--out FILE]
               [--debug PREFIX]
```

- `image` (optional positional) and `-s/--screenshot`: exactly like `screendump`
  (mutually exclusive; omit both -> capture screen; error if capture fails).
- `--region X1,Y1,X2,Y2`: pixel crop region. Default: full screen. Out-of-bounds
  values must be clamped to the image, and an invalid/empty region is an error.
- Criteria — at least one of `--text`, `--color`, `--shape` is required:
  - `--text STR`: fuzzy text search inside the region (see matching).
  - `--color #RRGGBB`: hex color; matching blob must contain pixels within tolerance.
  - `--shape`: `circle | square | rect | triangle`. When given together with
    `--color`, evaluate the shape on the color-masked blobs; when given alone,
    evaluate on filled/strongly-colored blobs detected in the region.
- `--tol INT`: per-channel color tolerance, default 40.
- `--fuzzy RATIO`: minimum fuzzy text similarity, default 0.75.
- `--psm`, `--min-conf`: passed to OCR, same defaults as screendump (11, 30.0).
- `--max N`: cap the number of returned matches, default no cap.
- `-j/--json`, `--out FILE`, `--debug PREFIX`: same semantics as screendump.
  `--debug` writes a `PREFIX.json` (internal details) and `PREFIX.png` (region
  annotated with every candidate bbox + center dot).

## Matching logic (`locater/match.py`)

Work on the cropped region first, then translate all bboxes/centers back to
**absolute screen pixels** before returning.

### Color blobs
- Parse hex to RGB; convert target and crop to HSV.
- `cv2.inRange` with tolerance. **Handle hue wrap-around at 180** (either build the
  mask in two passes or use an H radius helper).
- Clean the mask: `cv2.morphologyEx` open then close with a small kernel so noise
  and antialiasing don't create spurious blobs.
- `cv2.findContours` (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) on the mask.
- For each contour: drop blobs smaller than ~8x8 px; record bbox, center, area,
  color match ratio (fraction of blob pixels inside the mask), and perimeter.
- Multiple blobs are separate candidates.

### Shape classification (per blob)
- `circularity = 4*pi*area/perimeter^2`.
- `circle`: circularity >= 0.8.
- Otherwise `approxPolyDP` (epsilon ~ 2% of perimeter): 4 vertices -> square if
  aspect ratio (w/h) in [0.85, 1.18] else `rect`; 3 vertices -> `triangle`.
- Shape score = 1.0 on exact match, 0.5 on "near match" (e.g. square requested but
  rounded-square/circle-ish found), 0.0 otherwise.

### Text match
- Run `ocr.ocr_words` on the crop (then `group_lines`).
- Normalize: lowercase, strip punctuation.
- Score = `difflib.SequenceMatcher` ratio between target and each line; also accept
  if target is a fuzzy substring of the line (score = ratio in that case).
- Candidate bbox = the line's bbox (absolute pixels), confidence = OCR conf.
- A candidate only counts if score >= `--fuzzy`.

### Scoring and ranking
- Overall score in [0, 1]:
  - Single criterion: the criterion's score (text: fuzzy ratio × (conf/100) weight;
    color: color match ratio; shape: shape score).
  - Multiple criteria: weighted combination, PLUS an overlap bonus when the criteria
    match spatially overlapping candidates (e.g. color blob overlaps text line).
    Merge candidates by bbox overlap (>50% IoU) so a single element isn't reported
    twice; merge should keep the higher score and union the criteria info.
- Sort descending by score; drop candidates with score 0; apply `--max`.

## Output format

`-j` JSON (the important format for agents):

```json
{
  "screen": {"width": 1920, "height": 1080},
  "region": [x1, y1, x2, y2],
  "matches": [
    {
      "score": 0.92,
      "bbox": [x, y, w, h],
      "center": [cx, cy],
      "confidence": 78.5,
      "criteria": {"color": "#FF7A00", "shape": "circle", "text": "Brave"}
    }
  ]
}
```

`center` and `bbox` are **absolute screen pixels** — the agent clicks `center`.
When no matches are found: `"matches": []` (exit code 0) plus a JSON `"summary"`
field and a human-readable note explaining the failure mode (no blobs / no text
match / below threshold).

Human-readable output (no `-j`): a few ranked lines, each like
`0.92 circle #FF7A00 at (410, 920) [400, 900, 60, 50]`.

## Error handling

- Capture errors: same behavior as screendump (`error: ...`, exit 1).
- No criteria given: `error: specify at least one of --text, --color, --shape`,
  exit 1.
- Bad hex color / malformed `--region`: clear `error:` message, exit 1.
- OCR failure inside `--text` matching: warn on stderr and continue with the other
  criteria (mirror screendump's `warning: OCR failed:` handling).

## Tests

Add `tests/test_locater_match.py` mirroring the existing test style. Build a
synthetic image with PIL (like `tests/make_sample.py`): e.g. a light-grey 1280x720
background with (1) an orange (#FF7A00) filled circle at a known location, (2) a
blue rounded square, (3) a green rectangle, (4) black text "Hello World" nearby.
Assert, using `match.py` functions directly with a `--region` that isolates parts
of the image:

- color-only search finds the orange blob with the correct absolute center/bbox;
- color+shape `circle` finds only the orange circle (not the rounded square);
- text search for "hlo" (fuzzy) finds the "Hello World" line bbox;
- out-of-bounds region is clamped without crashing;
- a `--region` that contains none of the target yields an empty match list.

Keep tests dependency-light (PIL only, no real screen, no tesseract needed for the
color/shape tests; guard any OCR-dependent test to skip if tesseract is absent).

## Definition of done

- `locater` runs from a terminal: `locater -s --region 0,0,500,500 --color #FF7A00 -j`
  prints valid JSON with absolute-pixel matches.
- `screendump` still works unchanged.
- `python -m pytest tests/` (or however the repo runs tests) passes, including the
  new test file.
- No comments in the code unless truly necessary; follow the existing code style
  (dataclasses, type hints, `from __future__ import annotations`).
