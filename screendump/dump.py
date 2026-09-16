"""Core screendump processing pipeline.

Encapsulates screen capture, vision analysis, OCR extraction, semantic tree
classification, and rendering into a unified reusable function.
"""

from __future__ import annotations

import json
import shutil
from typing import Any

import cv2
import numpy as np

from screendump import layout, ocr, render, semantic, vision


def dump_image(
    bgr: np.ndarray,
    is_json: bool = False,
    fast: bool = False,
    targeted: bool = False,
    ascii_chars: bool = False,
    terminal_width: int | None = None,
    psm: int = 11,
    min_conf: float = 30.0,
    debug: dict | None = None,
    window: str | None = None,
    mode: str | None = None,
) -> str | dict[str, Any]:
    """Run the unified screendump pipeline on a BGR image and return text or dict.

    Supports:
    - window scoping: crops directly to requested window (W1, W2, or window title)
    - modes: 'auto' (smart targeted + window aware), 'targeted', 'fast', 'full'
    """
    img_h, img_w = bgr.shape[:2]

    # Resolve mode flag compatibility
    resolved_mode = mode.lower() if mode else None
    if resolved_mode is None:
        if targeted:
            resolved_mode = "targeted"
        elif fast:
            resolved_mode = "fast"
        else:
            resolved_mode = "auto"

    # Window Scoping: if a specific window is requested, find and crop to it
    target_window_obj = None
    offset_x, offset_y = 0, 0
    work_bgr = bgr

    if window:
        initial_regions = vision.detect_regions(bgr, debug=debug)
        roots = layout.build_tree(initial_regions, [])
        win_candidates = [r for r in roots if r.kind == "window"]
        w_query = window.strip().lower()
        for w in win_candidates:
            if (
                w.window_id.lower() == w_query
                or w.window_id.lower() == f"w{w_query}"
                or w_query in (w.title or "").lower()
            ):
                target_window_obj = w
                break

        if target_window_obj is not None:
            offset_x, offset_y = target_window_obj.x, target_window_obj.y
            crop_x1 = min(img_w, target_window_obj.right)
            crop_y1 = min(img_h, target_window_obj.bottom)
            work_bgr = bgr[offset_y:crop_y1, offset_x:crop_x1]
            if work_bgr.size == 0:
                work_bgr = bgr
                offset_x, offset_y = 0, 0

    cur_h, cur_w = work_bgr.shape[:2]

    # 1. Vision: partition detection and UI contour detection
    regions = vision.detect_regions(work_bgr, debug=debug)

    # 2. OCR: text extraction based on resolved mode
    if resolved_mode in ("targeted", "auto"):
        from screendump import targeted_ocr
        lines = targeted_ocr.extract_targeted_lines(work_bgr, regions, min_conf=min_conf, psm=6)
    elif resolved_mode == "fast":
        # Fast mode: downsample 2x for global OCR (4x reduction in pixel area)
        small = cv2.resize(work_bgr, (max(1, cur_w // 2), max(1, cur_h // 2)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", small)
        raw_words = ocr.ocr_words(buf.tobytes(), psm=psm, min_conf=min_conf * 0.9)
        rescaled_words = [
            ocr.Word(text=w.text, x=w.x * 2, y=w.y * 2, w=w.w * 2, h=w.h * 2, conf=w.conf)
            for w in raw_words
        ]
        splits = [r.x for r in regions if r.kind == "window" and r.x > 0]
        lines = ocr.group_lines(rescaled_words, splits=splits)
    else:
        # Full mode: standard deep multi-pass OCR
        splits = [r.x for r in regions if r.kind == "window" and r.x > 0]
        ok, buf = cv2.imencode(".png", work_bgr)
        words = ocr.ocr_words(buf.tobytes(), psm=psm, min_conf=min_conf)
        lines = ocr.group_lines(words, splits=splits)

    # 3. Layout & Semantic tree
    roots = layout.build_tree(regions, lines)
    semantic.classify(roots)

    # If scoped to a window, remap coordinates back to global desktop space for JSON output
    if target_window_obj is not None and is_json:
        res_dict = semantic.to_dict(roots, cur_w, cur_h)
        # Remap coordinates in elements
        for el in res_dict.get("elements", []):
            bx, by, bw, bh = el.get("bbox", [0, 0, 0, 0])
            el["bbox"] = [bx + offset_x, by + offset_y, bw, bh]
        res_dict["screen"] = {"width": img_w, "height": img_h}
        res_dict["window"] = {
            "id": target_window_obj.window_id,
            "title": target_window_obj.title,
            "bbox": [offset_x, offset_y, cur_w, cur_h],
        }
        return res_dict

    # 4. Serialization / Rendering
    if is_json:
        return semantic.to_dict(roots, cur_w, cur_h)

    width = terminal_width or shutil.get_terminal_size((80, 24)).columns
    return render.render(roots, cur_w, cur_h, width=width, ascii_chars=ascii_chars)

