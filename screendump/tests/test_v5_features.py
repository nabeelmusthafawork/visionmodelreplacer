"""Tests for v5 unified features:
- Screendump window scoping
- Screendump mode selection (auto, targeted, fast, full)
- Locater spatial window confinement
- Type-and-enter action integration
"""

import cv2
import numpy as np
import pytest

from locater.spatial import find_relative_element
from screendump.dump import dump_image
from screendump import vision, layout


def test_screendump_window_scoping():
    # Synthetic desktop with two windows side-by-side: W1 (left), W2 (right)
    img = np.full((600, 1000, 3), 240, dtype=np.uint8)

    # Window 1: x in [20, 480], y in [30, 570]
    cv2.rectangle(img, (20, 30), (480, 570), (40, 40, 40), 2)
    # Put a button inside W1
    cv2.rectangle(img, (50, 100), (180, 140), (80, 80, 80), 2)
    cv2.putText(img, "Button1", (60, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Window 2: x in [520, 980], y in [30, 570]
    cv2.rectangle(img, (520, 30), (980, 570), (40, 40, 40), 2)
    # Put a button inside W2
    cv2.rectangle(img, (550, 100), (680, 140), (80, 80, 80), 2)
    cv2.putText(img, "Button2", (560, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # 1. Dump full desktop
    full_res = dump_image(img, is_json=True, mode="auto")
    assert isinstance(full_res, dict)
    assert len(full_res.get("elements", [])) >= 2

    # 2. Dump scoped to W1
    w1_res = dump_image(img, is_json=True, window="W1", mode="auto")
    assert isinstance(w1_res, dict)
    assert "window" in w1_res
    # All elements must have coordinates inside W1 (x < 500)
    for el in w1_res.get("elements", []):
        bx, by, bw, bh = el["bbox"]
        assert bx < 500, f"Element leaked out of W1: {el}"


def test_screendump_modes():
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 100), (50, 50, 50), 2)
    cv2.putText(img, "Submit", (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Test auto mode
    res_auto = dump_image(img, is_json=True, mode="auto")
    assert isinstance(res_auto, dict)

    # Test fast mode
    res_fast = dump_image(img, is_json=True, mode="fast")
    assert isinstance(res_fast, dict)

    # Test targeted mode
    res_targeted = dump_image(img, is_json=True, mode="targeted")
    assert isinstance(res_targeted, dict)

    # Test full mode
    res_full = dump_image(img, is_json=True, mode="full")
    assert isinstance(res_full, dict)


def test_spatial_window_confinement():
    img = np.full((400, 800, 3), 255, dtype=np.uint8)

    # Label inside Window 1: x in [0, 380]
    cv2.putText(img, "Target", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # Box inside Window 1 to the right of label
    cv2.rectangle(img, (150, 80), (250, 120), (100, 100, 100), 2)

    # Box inside Window 2: x in [420, 780]
    cv2.rectangle(img, (450, 80), (550, 120), (100, 100, 100), 2)

    # Search with region restricted to Window 1: [0, 0, 380, 400]
    matches = find_relative_element(
        img,
        relation="right_of",
        anchor_text="Target",
        region=(0, 0, 380, 400),
    )
    assert len(matches) >= 1
    # Matches must be inside Window 1
    for m in matches:
        rx, ry, rw, rh = m.bbox
        assert rx + rw <= 380
