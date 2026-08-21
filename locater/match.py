"""Detection logic for locater: color blobs, shape classification, text matching.

All returned bboxes and centers are absolute screen pixel coordinates
(translated back from the cropped region by ``locate``).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import cv2
import numpy as np

from screendump import ocr

MIN_BLOB = 8
NEAR_SHAPES = {
    ("circle", "square"),
    ("square", "circle"),
    ("square", "rect"),
    ("rect", "square"),
}


@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int
    area: float
    perimeter: float
    contour: np.ndarray
    color_ratio: float = 0.0

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class Candidate:
    score: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float = 0.0
    criteria: dict[str, str] = field(default_factory=dict)
    blob: Blob | None = None

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "bbox": list(self.bbox),
            "center": list(self.center),
            "confidence": round(self.confidence, 2),
            "criteria": dict(self.criteria),
        }


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse #RRGGBB into an (r, g, b) tuple; raise ValueError on bad input."""
    if not value.startswith("#") or len(value) != 7:
        raise ValueError(f"bad color {value!r} (expected #RRGGBB)")
    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except ValueError:
        raise ValueError(f"bad color {value!r} (expected #RRGGBB)") from None


def clamp_region(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Clamp a region into the image; raise ValueError if it becomes empty."""
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty region after clamping: {x1},{y1},{x2},{y2}")
    return x1, y1, x2, y2


def _color_mask(hsv: np.ndarray, target_hsv: np.ndarray, tol: int) -> np.ndarray:
    """Boolean mask of pixels within tolerance of the target HSV color.

    Hue is circular (0-180): the distance is computed with a wrap-around
    helper so red at the 0/180 seam still matches.
    """
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    hc, sc, vc = (float(x) for x in target_hsv)
    hue_tol = tol * 180.0 / 255.0
    dh = np.abs((h - hc + 90.0) % 180.0 - 90.0) <= hue_tol
    ds = np.abs(s - sc) <= tol
    dv = np.abs(v - vc) <= tol
    return (dh & ds & dv).astype(np.uint8) * 255


def _strong_color_mask(hsv: np.ndarray) -> np.ndarray:
    """Mask of filled, strongly-colored blobs (used when only --shape given)."""
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    return ((s > 60) & (v > 80)).astype(np.uint8) * 255


def _blobs_from_mask(mask: np.ndarray, color_ratio: bool = True) -> list[Blob]:
    """Clean the mask and extract contour blobs (min ~8x8 px)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
        cv2.MORPH_CLOSE,
        kernel,
    )
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[Blob] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < MIN_BLOB or h < MIN_BLOB:
            continue
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        ratio = 1.0
        if color_ratio:
            fill = np.zeros_like(mask)
            cv2.drawContours(fill, [c], -1, 255, thickness=cv2.FILLED)
            roi_mask = mask[y : y + h, x : x + w]
            roi_fill = fill[y : y + h, x : x + w]
            n = np.count_nonzero(roi_fill)
            ratio = float(np.count_nonzero(roi_mask & roi_fill)) / n if n else 1.0
        blobs.append(
            Blob(
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                perimeter=float(cv2.arcLength(c, True)),
                contour=c,
                color_ratio=ratio,
            )
        )
    return blobs


def _shape_of(blob: Blob) -> str:
    """Classify a blob as circle | square | rect | triangle | unknown."""
    circularity = 4 * np.pi * blob.area / (blob.perimeter**2)
    if circularity >= 0.8:
        return "circle"
    approx = cv2.approxPolyDP(blob.contour, 0.02 * blob.perimeter, True)
    n = len(approx)
    if n == 3:
        return "triangle"
    aspect = blob.w / blob.h if blob.h else 0.0
    if n == 4 or blob.area >= 0.6 * blob.w * blob.h:
        return "square" if 0.85 <= aspect <= 1.18 else "rect"
    return "unknown"


def shape_score(found: str, requested: str) -> float:
    """1.0 on exact match, 0.5 on near match (rounded/square/circle-ish), else 0."""
    if found == requested:
        return 1.0
    if (found, requested) in NEAR_SHAPES:
        return 0.5
    return 0.0


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()


def fuzzy_score(target: str, line: str) -> float:
    """SequenceMatcher ratio between target and line, or its best substring."""
    t = _normalize(target)
    ln = _normalize(line)
    if not t or not ln:
        return 0.0
    best = difflib.SequenceMatcher(None, t, ln).ratio()
    for i in range(len(ln) - len(t) + 1):
        best = max(best, difflib.SequenceMatcher(None, t, ln[i : i + len(t)]).ratio())
    return best


def _candidate(
    blob: Blob,
    ox: int,
    oy: int,
    score: float,
    confidence: float,
    criteria: dict[str, str],
) -> Candidate:
    return Candidate(
        score=score,
        bbox=(blob.x + ox, blob.y + oy, blob.w, blob.h),
        center=(int(round(blob.cx + ox)), int(round(blob.cy + oy))),
        confidence=confidence,
        criteria=criteria,
        blob=blob,
    )


def text_candidates(
    crop_bgr: np.ndarray,
    offset: tuple[int, int],
    target: str,
    psm: int,
    min_conf: float,
    fuzzy: float,
) -> list[Candidate]:
    """OCR the crop and return text candidates with score >= fuzzy."""
    ok, buf = cv2.imencode(".png", crop_bgr)
    if not ok:
        return []
    words = ocr.ocr_words(buf.tobytes(), psm=psm, min_conf=min_conf)
    ox, oy = offset
    out: list[Candidate] = []
    for line in ocr.group_lines(words):
        ratio = fuzzy_score(target, line.text)
        if ratio < fuzzy:
            continue
        out.append(
            Candidate(
                score=ratio * line.conf / 100.0,
                bbox=(line.x + ox, line.y + oy, line.w, line.h),
                center=(int(round(line.cx + ox)), int(round(line.cy + oy))),
                confidence=line.conf,
                criteria={"text": target},
            )
        )
    return out


def _iou(a: Candidate, b: Candidate) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    oy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ox * oy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _combine(a: Candidate, b: Candidate) -> Candidate:
    x = min(a.bbox[0], b.bbox[0])
    y = min(a.bbox[1], b.bbox[1])
    right = max(a.bbox[0] + a.bbox[2], b.bbox[0] + b.bbox[2])
    bottom = max(a.bbox[1] + a.bbox[3], b.bbox[1] + b.bbox[3])
    bbox = (x, y, right - x, bottom - y)
    score = min(1.0, (a.score + b.score) / 2.0 + 0.15)
    return Candidate(
        score=score,
        bbox=bbox,
        center=(int(round((x + right) / 2)), int(round((y + bottom) / 2))),
        confidence=max(a.confidence, b.confidence),
        criteria={**a.criteria, **b.criteria},
    )


def _merge_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Merge candidates whose bboxes overlap >50% IoU (same element)."""
    work = list(candidates)
    while True:
        merged: list[Candidate] = []
        used: set[int] = set()
        for i, a in enumerate(work):
            if i in used:
                continue
            partner = None
            for j, b in enumerate(work):
                if j <= i or j in used:
                    continue
                if _iou(a, b) > 0.5:
                    partner = j
                    break
            if partner is not None:
                merged.append(_combine(a, work[partner]))
                used.add(i)
                used.add(partner)
            else:
                merged.append(a)
        if len(merged) == len(work):
            return merged
        work = merged


def _blob_dict(b: Blob) -> dict:
    return {
        "bbox": [b.x, b.y, b.w, b.h],
        "area": round(b.area, 1),
        "perimeter": round(b.perimeter, 1),
        "color_ratio": round(b.color_ratio, 3),
        "shape": _shape_of(b),
    }


def locate(
    bgr: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    text: str | None = None,
    color: str | None = None,
    shape: str | None = None,
    tol: int = 40,
    fuzzy: float = 0.75,
    psm: int = 11,
    min_conf: float = 30.0,
    max_n: int | None = None,
) -> tuple[list[Candidate], dict]:
    """Find matches for the given criteria inside the (clamped) region.

    Returns (matches sorted by score desc, info dict with internals for
    debug/summary purposes).
    """
    img_h, img_w = bgr.shape[:2]
    if region is None:
        x1, y1, x2, y2 = 0, 0, img_w, img_h
    else:
        x1, y1, x2, y2 = clamp_region(*region, img_w, img_h)
    crop = bgr[y1:y2, x1:x2]
    info: dict = {
        "region": [x1, y1, x2, y2],
        "criteria": {},
        "blobs": [],
        "ocr_error": None,
    }
    if crop.size == 0:
        return [], info
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    candidates: list[Candidate] = []
    color_blobs: list[Blob] = []

    if color is not None:
        target_hsv = cv2.cvtColor(np.uint8([[hex_to_rgb(color)]]), cv2.COLOR_RGB2HSV)[0, 0]
        mask = _color_mask(hsv, target_hsv, tol)
        color_blobs = _blobs_from_mask(mask)
        info["criteria"]["color"] = color
        info["blobs"] = [_blob_dict(b) for b in color_blobs]
        for b in color_blobs:
            candidates.append(
                _candidate(b, x1, y1, b.color_ratio, b.color_ratio * 100.0, {"color": color})
            )

    if shape is not None:
        info["criteria"]["shape"] = shape
        if color is not None:
            blobs = color_blobs
        else:
            blobs = _blobs_from_mask(_strong_color_mask(hsv), color_ratio=False)
            info["blobs"] = [_blob_dict(b) for b in blobs]
        for b in blobs:
            found = _shape_of(b)
            score = shape_score(found, shape)
            if score <= 0:
                continue
            candidates.append(_candidate(b, x1, y1, score, score * 100.0, {"shape": shape}))

    if text is not None:
        info["criteria"]["text"] = text
        try:
            candidates.extend(text_candidates(crop, (x1, y1), text, psm, min_conf, fuzzy))
        except ocr.TesseractError as exc:
            info["ocr_error"] = str(exc)

    matches = [c for c in _merge_candidates(candidates) if c.score > 0]
    matches.sort(key=lambda c: c.score, reverse=True)
    if max_n is not None:
        matches = matches[:max_n]
    return matches, info
