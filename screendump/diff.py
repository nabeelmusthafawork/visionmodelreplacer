"""Visual diffing and input verification utilities.

Provides pixel-level change detection between screen states to confirm whether
user inputs (clicks, typing, shortcut keys) had a visual effect on the screen.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_region_diff(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    threshold: int = 20,
) -> dict:
    """Compare two frames within an optional bounding region [x, y, w, h] or (x1, y1, x2, y2).

    Returns a dict with:
      - changed: bool (True if changed pixels exceed threshold)
      - changed_pixels: int (count of pixels whose absolute difference > threshold)
      - change_ratio: float (ratio of changed pixels to total region pixels)
      - diff_score: float (mean absolute pixel difference across the region)
    """
    if before_bgr.shape != after_bgr.shape:
        raise ValueError(
            f"Image dimensions do not match: {before_bgr.shape} vs {after_bgr.shape}"
        )

    h, w = before_bgr.shape[:2]
    if region is not None:
        if len(region) == 4:
            # Handle both [x, y, w, h] and (x1, y1, x2, y2)
            # If 3rd/4th values look like coordinates (x2 > x1 and y2 > y1 and region[2] > w/2):
            # To be unambiguous, we support standard bbox [x, y, w, h].
            rx, ry, rw, rh = region
            x0 = max(0, min(w, rx))
            y0 = max(0, min(h, ry))
            x1 = max(0, min(w, rx + rw))
            y1 = max(0, min(h, ry + rh))
        else:
            x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    crop_before = before_bgr[y0:y1, x0:x1]
    crop_after = after_bgr[y0:y1, x0:x1]

    if crop_before.size == 0 or crop_after.size == 0:
        return {
            "changed": False,
            "changed_pixels": 0,
            "change_ratio": 0.0,
            "diff_score": 0.0,
            "region": [x0, y0, x1 - x0, y1 - y0],
        }

    # Convert to grayscale for robust intensity diff
    gray_before = cv2.cvtColor(crop_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(crop_after, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_before, gray_after)
    changed_mask = diff > threshold
    changed_pixels = int(np.count_nonzero(changed_mask))
    total_pixels = int(crop_before.shape[0] * crop_before.shape[1])
    change_ratio = float(changed_pixels / total_pixels) if total_pixels > 0 else 0.0
    diff_score = float(np.mean(diff))

    # A region is considered changed if at least 6 pixels or 0.1% of pixels changed
    is_changed = changed_pixels >= 6 and (change_ratio >= 0.001 or diff_score >= 1.5)

    return {
        "changed": is_changed,
        "changed_pixels": changed_pixels,
        "change_ratio": change_ratio,
        "diff_score": diff_score,
        "region": [x0, y0, x1 - x0, y1 - y0],
    }


def verify_text_input(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    input_bbox: tuple[int, int, int, int] | list[int],
    cursor_pos: tuple[int, int] | None = None,
) -> dict:
    """Verify whether keystrokes were successfully rendered into an input field.

    If input_bbox is provided, evaluates diff inside the bounding box.
    If cursor_pos is provided, evaluates diff within a 200x50 padding around the cursor.
    """
    if input_bbox is not None:
        x, y, w, h = input_bbox
        # Add 4px padding around the input box
        pad = 4
        region = (x - pad, y - pad, w + 2 * pad, h + 2 * pad)
    elif cursor_pos is not None:
        cx, cy = cursor_pos
        region = (cx - 100, cy - 25, 200, 50)
    else:
        region = None

    diff_res = compute_region_diff(before_bgr, after_bgr, region=region, threshold=15)
    return {
        "verified": diff_res["changed"],
        "details": diff_res,
    }


def find_dirty_rects(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    min_area: int = 100,
    threshold: int = 25,
) -> list[tuple[int, int, int, int]]:
    """Detect bounding boxes of visual changes between two frames (dirty rectangles).

    Useful for tracking popups, dropdowns, or notifications that appeared
    after an action, without re-scanning the entire screen.
    """
    if before_bgr.shape != after_bgr.shape:
        raise ValueError("Image dimensions must match to compute dirty rects")

    gray_before = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_before, gray_after)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Dilate to connect nearby character/component changes into a unified bounding box
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        rects.append((x, y, w, h))

    # Sort largest area first
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    return rects
