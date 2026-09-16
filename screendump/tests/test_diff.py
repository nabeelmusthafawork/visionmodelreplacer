"""Tests for diff.py change detection and input verification."""

import cv2
import numpy as np
import pytest
from screendump.diff import compute_region_diff, verify_text_input


def test_compute_region_diff_identical():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = compute_region_diff(img, img)
    assert not res["changed"]
    assert res["changed_pixels"] == 0
    assert res["diff_score"] == 0.0


def test_compute_region_diff_changed():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = img1.copy()
    # Draw a line/text
    cv2.putText(img2, "Hello", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    res = compute_region_diff(img1, img2, region=(5, 30, 80, 40))
    assert res["changed"]
    assert res["changed_pixels"] > 10
    assert res["diff_score"] > 5.0


def test_verify_text_input():
    img1 = np.ones((200, 400, 3), dtype=np.uint8) * 240
    img2 = img1.copy()

    # Case 1: Typing failed / no change
    res_fail = verify_text_input(img1, img2, input_bbox=(50, 50, 200, 40))
    assert not res_fail["verified"]

    # Case 2: Text rendered
    cv2.putText(img2, "Test text", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    res_ok = verify_text_input(img1, img2, input_bbox=(50, 50, 200, 40))
    assert res_ok["verified"]


def test_find_dirty_rects():
    from screendump.diff import find_dirty_rects
    img1 = np.full((400, 400, 3), 255, dtype=np.uint8)
    img2 = img1.copy()
    # Draw a popup/dropdown at (150, 100, 120, 80)
    cv2.rectangle(img2, (150, 100), (270, 180), (50, 50, 50), -1)

    rects = find_dirty_rects(img1, img2, min_area=200)
    assert len(rects) >= 1
    rx, ry, rw, rh = rects[0]
    # Check that the dirty rect tightly covers the popup
    assert abs(rx - 150) <= 20
    assert abs(ry - 100) <= 20
    assert rw >= 100 and rh >= 70
