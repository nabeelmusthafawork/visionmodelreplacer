"""Targeted / Lazy OCR: extracts text only from detected interactive boxes.

Instead of running full-screen OCR on 2M pixels (8-10s), targeted OCR:
1. Filters detected regions for text-bearing candidates (buttons, inputs, tabs, headings).
2. Stacks the cropped regions vertically into a single consolidated image strip.
3. Runs a single fast OCR pass on the small composite strip (~50-80ms).
4. Remaps the detected words and lines back to their original screen coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np

from screendump import ocr, vision


def extract_targeted_lines(
    bgr: np.ndarray,
    regions: list[vision.Region],
    min_conf: float = 20.0,
    psm: int = 11,
) -> list[ocr.Line]:
    """Perform targeted OCR only within candidate bounding boxes."""
    img_h, img_w = bgr.shape[:2]

    # Filter regions that are candidates for containing labels/text
    candidates = [
        r for r in regions
        if r.kind in ("button", "input", "tab", "panel", "window", "box")
        and 10 <= r.h <= 300
        and 15 <= r.w <= 1400
    ]

    if not candidates:
        return []

    # Sort candidates top-to-bottom
    candidates.sort(key=lambda r: (r.y, r.x))

    # Build stacked strip
    pad = 4
    strip_pad = 8
    crops: list[tuple[vision.Region, int, int, int, int]] = []
    total_h = 0
    max_w = 0

    for r in candidates:
        inset = 5 if r.w >= 30 and r.h >= 20 else 0
        x0 = max(0, r.x + inset)
        y0 = max(0, r.y + inset)
        x1 = min(img_w, r.right - inset)
        y1 = min(img_h, r.bottom - inset)
        w = x1 - x0
        h = y1 - y0
        if w < 10 or h < 8:
            continue
        crop = bgr[y0:y1, x0:x1]
        # Quick variance filter: skip solid flat color boxes with no text/edges
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 15.0:
            continue

        crops.append((r, x0, y0, w, h))
        total_h += h + strip_pad
        max_w = max(max_w, w)

    if not crops or total_h == 0 or max_w == 0:
        return []

    # Assemble stacked composite image
    composite = np.full((total_h, max_w, 3), 255, dtype=np.uint8)
    curr_y = 0
    crop_mappings: list[tuple[int, int, int, int, int]] = []  # (comp_y, orig_x0, orig_y0, w, h)

    for r, x0, y0, w, h in crops:
        composite[curr_y : curr_y + h, 0:w] = bgr[y0 : y0 + h, x0 : x0 + w]
        crop_mappings.append((curr_y, x0, y0, w, h))
        curr_y += h + strip_pad

    # Add border margin and upscale 2x so Tesseract sees standard DPI glyphs
    margin = 30
    padded = cv2.copyMakeBorder(
        composite, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    big = cv2.resize(padded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    ok, buf = cv2.imencode(".png", big)
    if not ok:
        return []

    raw_words = ocr.ocr_words(buf.tobytes(), psm=psm, min_conf=min_conf)
    if not raw_words:
        return []

    # Remap words from composite space back to original screen space
    remapped_words: list[ocr.Word] = []
    for w in raw_words:
        comp_x = (w.x - margin * 2) // 2
        comp_y = (w.y - margin * 2) // 2
        word_w = max(1, w.w // 2)
        word_h = max(1, w.h // 2)

        # Find which crop contains this word
        for cy, orig_x0, orig_y0, cw, ch in crop_mappings:
            if cy - 4 <= comp_y < cy + ch + 8:
                local_y = comp_y - cy
                local_x = comp_x
                remapped_words.append(
                    ocr.Word(
                        text=w.text,
                        x=orig_x0 + local_x,
                        y=orig_y0 + local_y,
                        w=word_w,
                        h=word_h,
                        conf=w.conf,
                    )
                )
                break

    splits = [r.x for r in regions if r.kind == "window" and r.x > 0]
    return ocr.group_lines(remapped_words, splits=splits)
