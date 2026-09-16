"""Spatial relationship queries for UI elements.

Allows locating input boxes, buttons, or indicators relative to text labels:
  --right-of "Username"
  --below "Password"
  --left-of "Submit"
  --above "Footer"
"""

from __future__ import annotations

import cv2
import numpy as np

from locater import match
from screendump import vision


def find_relative_element(
    bgr: np.ndarray,
    relation: str,
    anchor_text: str,
    target_shape: str | None = None,
    target_color: str | None = None,
    region: tuple[int, int, int, int] | None = None,
    max_distance: int = 400,
    tol: int = 40,
    fuzzy: float = 0.7,
) -> list[match.Candidate]:
    """Find elements positioned spatially relative to a text anchor."""
    img_h, img_w = bgr.shape[:2]

    # 1. Locate the anchor element
    anchor_matches, _ = match.locate(
        bgr,
        region=region,
        text=anchor_text,
        fuzzy=fuzzy,
        max_n=1,
    )
    if not anchor_matches:
        return []

    anchor = anchor_matches[0]
    ax, ay, aw, ah = anchor.bbox
    aright = ax + aw
    abottom = ay + ah
    acx, acy = anchor.center

    # 2. Detect visual element regions in the vicinity
    all_regions = vision.detect_regions(bgr)

    candidates: list[tuple[float, vision.Region]] = []

    rel = relation.lower().replace("-", "_")

    for r in all_regions:
        # Avoid matching the anchor itself
        if abs(r.x - ax) < 5 and abs(r.y - ay) < 5 and abs(r.w - aw) < 10 and abs(r.h - ah) < 10:
            continue
        if r.kind in ("window", "separator"):
            continue

        # Enforce region boundary if provided (e.g. from --window or --region)
        if region is not None:
            rx1, ry1, rx2, ry2 = region
            if r.x < rx1 or r.right > rx2 or r.y < ry1 or r.bottom > ry2:
                continue

        # Optional shape filtering
        if target_shape:
            ts = target_shape.lower()
            if ts == "circle" and r.kind != "circle":
                continue
            elif ts in ("rect", "square", "button", "input") and r.kind not in ("rect", "square", "button", "input", "box"):
                continue

        # Optional color filtering
        if target_color:
            try:
                target_rgb = match.hex_to_rgb(target_color)
                # Crop region from bgr and check mean color
                crop_patch = bgr[r.y : r.bottom, r.x : r.right]
                if crop_patch.size > 0:
                    mean_bgr = crop_patch.mean(axis=(0, 1))
                    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
                    dist = sum(abs(a - b) for a, b in zip(mean_rgb, target_rgb)) / 3.0
                    if dist > tol:
                        continue
            except Exception:
                pass

        rcx, rcy = r.x + r.w // 2, r.y + r.h // 2

        if rel in ("right_of", "right"):
            # Candidate must be to the right and vertically aligned
            dx = r.x - aright
            dy = abs(rcy - acy)
            if -5 <= dx <= max_distance and dy <= max(35, ah * 1.5):
                score = float(dx + dy * 0.5)
                candidates.append((score, r))

        elif rel in ("left_of", "left"):
            # Candidate must be to the left and vertically aligned
            dx = ax - r.right
            dy = abs(rcy - acy)
            if -5 <= dx <= max_distance and dy <= max(35, ah * 1.5):
                score = float(dx + dy * 0.5)
                candidates.append((score, r))

        elif rel in ("below", "under"):
            # Candidate must be below and horizontally overlapping/aligned
            dy = r.y - abottom
            dx = abs(rcx - acx)
            if -5 <= dy <= max_distance and dx <= max(120, aw * 1.5):
                score = float(dy + dx * 0.5)
                candidates.append((score, r))

        elif rel in ("above", "over"):
            # Candidate must be above and horizontally overlapping/aligned
            dy = ay - r.bottom
            dx = abs(rcx - acx)
            if -5 <= dy <= max_distance and dx <= max(120, aw * 1.5):
                score = float(dy + dx * 0.5)
                candidates.append((score, r))

    if not candidates:
        return []

    # Sort by spatial proximity
    candidates.sort(key=lambda item: item[0])

    # Convert best candidates to match.Candidate
    matches: list[match.Candidate] = []
    for dist, r in candidates[:5]:
        # Calculate confidence inversely proportional to distance
        conf = max(50.0, 100.0 - (dist / max_distance) * 50.0)
        criteria = {
            "spatial": f"{relation} '{anchor_text}'",
            "kind": r.kind,
        }
        matches.append(
            match.Candidate(
                bbox=(r.x, r.y, r.w, r.h),
                center=(r.x + r.w // 2, r.y + r.h // 2),
                confidence=conf,
                score=conf / 100.0,
                criteria=criteria,
            )
        )

    return matches
