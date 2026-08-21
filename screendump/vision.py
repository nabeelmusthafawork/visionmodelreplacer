"""Computer-vision helpers: find UI boxes, separators and filled regions.

All detection is done with pure OpenCV on the input image. The result is a
list of ``Region`` rects which the layout stage turns into a tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DILATE_REACH = 4  # px the detected bbox is expanded past the true border


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    kind: str = "box"  # box | separator | window | button | input | icon | panel
    fill_ratio: float = 0.0  # fraction of interior pixels that are "filled"
    border_ratio: float = 0.0  # fraction of border pixels that are dark
    text: str = ""  # filled in by the layout stage

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def _adaptive_kernel(size: int) -> int:
    """Dilation kernel size roughly proportional to image size."""
    return max(3, size // 400 * 2 + 1)


def _background_level(gray: np.ndarray) -> int:
    """Most common gray value in the image (the page background)."""
    hist = np.bincount(gray.ravel(), minlength=256)
    return int(hist.argmax())


def detect_regions(bgr: np.ndarray, debug: dict | None = None) -> list[Region]:
    """Detect rectangular UI regions (windows, cards, buttons, inputs...).

    Strategy:
      1. Canny edges, then dilate so neighbouring strokes merge into blobs.
      2. Find all contours (RETR_LIST: nested widgets too) -> bounding rects.
      3. Keep rects with plausible UI sizes, drop irregular text blobs and
         near-duplicate detections.
      4. Classify each rect: window/panel (large), button/input (small,
         bordered), icon (tiny) - measured on the ORIGINAL image so
         brightness semantics survive.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.equalizeHist(gray), 40, 120)
    kernel = _adaptive_kernel(w)
    dilated = cv2.dilate(edges, np.ones((kernel, kernel), np.uint8), iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    min_dim = max(8, w // 200)
    min_area = min_dim * min_dim * 4
    rects: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_dim or ch < min_dim:
            continue
        if cw * ch < min_area:
            continue
        if x <= 2 and y <= 2 and cw >= w - 4 and ch >= h - 4:
            continue  # the whole image
        peri = cv2.arcLength(c, True)
        rect_peri = 2 * (cw + ch)
        if peri > rect_peri * 1.35:
            continue  # irregular blob (text glyphs etc.), not a UI box
        rects.append((x, y, cw, ch))

    # Sort by area descending; drop near-duplicates (nearly same size AND
    # contained in a bigger kept rect) but KEEP genuinely nested widgets.
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for r in rects:
        rx, ry, rw, rh = r
        duplicate = False
        for k in kept:
            kx, ky, kw, kh = k
            if (
                rx >= kx
                and ry >= ky
                and rx + rw <= kx + kw
                and ry + rh <= ky + kh
                and rw >= 0.85 * kw
                and rh >= 0.8 * kh
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(r)

    # A "window" must be an outermost region (not contained in another).
    windows: set[int] = set()
    for i, (x, y, rw, rh) in enumerate(kept):
        if any(
            x >= kx + 3 and y >= ky + 3 and x + rw <= kx + kw - 3 and y + rh <= ky + kh - 3
            for kx, ky, kw, kh in kept
        ):
            continue
        windows.add(i)

    regions: list[Region] = []
    bg = _background_level(gray)
    for i, (x, y, rw, rh) in enumerate(kept):
        region = _classify(bgr, gray, x, y, rw, rh, bg, i in windows)
        if region is not None:
            regions.append(region)

    regions.extend(_detect_separators(gray, w, h, regions))

    if debug is not None:
        debug["regions"] = [vars(r) for r in regions]
        annotated = bgr.copy()
        for r in regions:
            color = {
                "box": (0, 255, 0),
                "separator": (255, 0, 255),
                "button": (0, 255, 255),
                "input": (255, 255, 0),
                "icon": (255, 0, 0),
                "panel": (0, 165, 255),
                "window": (0, 255, 0),
            }.get(r.kind, (0, 255, 0))
            cv2.rectangle(annotated, (r.x, r.y), (r.right, r.bottom), color, 2)
        debug["annotated"] = annotated

    return regions


def _border_sides(
    gray: np.ndarray, x: int, y: int, w: int, h: int, thresh: int, reach: int = 6
) -> int:
    """Count box border sides by scanning for near-continuous dark lines.

    The true border sits 0..reach px inside the dilated bbox; scan each
    offset and count how many of the four sides have a long dark run.
    Text glyphs never produce such runs, so this cleanly separates real
    boxes from text blobs.
    """
    sides = 0
    min_run = 0.8
    for offset in range(reach):
        if y + offset >= y + h or x + offset >= x + w:
            break
        if np.mean(gray[y + offset, x : x + w] < thresh) > min_run:
            sides |= 1
            break
    for offset in range(reach):
        if y + h - 1 - offset <= y:
            break
        if np.mean(gray[y + h - 1 - offset, x : x + w] < thresh) > min_run:
            sides |= 2
            break
    for offset in range(reach):
        if np.mean(gray[y : y + h, x + offset] < thresh) > min_run:
            sides |= 4
            break
    for offset in range(reach):
        if np.mean(gray[y : y + h, x + w - 1 - offset] < thresh) > min_run:
            sides |= 8
            break
    return sides


def _classify(
    bgr: np.ndarray,
    gray: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg: int,
    is_window: bool,
) -> Region | None:
    """Decide what kind of widget a detected rect is (or drop it)."""
    thresh = min(200, max(60, bg - 60))

    inset = DILATE_REACH + 2
    ix = x + inset
    iy = y + inset
    iw = w - 2 * inset
    ih = h - 2 * inset
    if iw <= 4 or ih <= 4:
        return None

    sides = _border_sides(gray, x, y, w, h, thresh)

    interior = gray[iy : iy + ih, ix : ix + iw]
    fill_ratio = float(np.mean(interior < thresh))

    hsv = cv2.cvtColor(bgr[iy : iy + ih, ix : ix + iw], cv2.COLOR_BGR2HSV)
    sat_ratio = float(np.mean(hsv[:, :, 1] > 60))

    region = Region(x=x, y=y, w=w, h=h, fill_ratio=fill_ratio, border_ratio=sides / 4.0)

    area_ratio = (iw * ih) / max(1, gray.shape[0] * gray.shape[1])

    if is_window and area_ratio > 0.18:
        region.kind = "window"
    elif iw < 48 and ih < 48:
        if sides < 2:
            return None  # text glyph blob, not a widget
        region.kind = "icon"
    elif sides < 2:
        return None  # text blob / irregular blob, not a UI box
    elif fill_ratio > 0.45 or sat_ratio > 0.15:
        region.kind = "button"
    elif area_ratio > 0.04 and ih > 48:
        region.kind = "panel"
    elif iw < 0.7 * gray.shape[1]:
        region.kind = "input"
    else:
        region.kind = "panel"  # full-width band: toolbar / menu / status bar
    return region


def _detect_separators(
    gray: np.ndarray, w: int, h: int, regions: list[Region]
) -> list[Region]:
    """Find thin horizontal/vertical divider lines (menus, table headers...).

    Box borders are long straight lines too, so any separator candidate that
    hugs the edge of a detected region is discarded.
    """
    results: list[Region] = []
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15
    )
    min_len = max(40, w // 6)
    tol = max(4, w // 400 * 2 + 1 + 2 * 2 + 2)  # dilation reach + slack

    horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    )
    contours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_len or ch > 6:
            continue
        if any(
            r.kind != "separator"
            and cw >= r.w * 0.6
            and x >= r.x - tol
            and x + cw <= r.right + tol
            and (abs(y - r.y) <= tol or abs(y + ch - r.bottom) <= tol)
            for r in regions
        ):
            continue
        results.append(Region(x=x, y=y, w=cw, h=ch, kind="separator"))

    vertical = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    )
    contours, _ = cv2.findContours(
        vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < min_len or cw > 6:
            continue
        if any(
            r.kind != "separator"
            and ch >= r.h * 0.6
            and y >= r.y - tol
            and y + ch <= r.bottom + tol
            and (abs(x - r.x) <= tol or abs(x + cw - r.right) <= tol)
            for r in regions
        ):
            continue
        results.append(Region(x=x, y=y, w=cw, h=ch, kind="separator"))
    return results