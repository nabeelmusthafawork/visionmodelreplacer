"""Tests for locater action dispatch and verification."""

import cv2
import numpy as np
import pytest
from locater import action
from screendump.diff import compute_region_diff


def test_action_buttons_mapping():
    assert action.BUTTONS["left"] == "0xC0"
    assert action.BUTTONS["right"] == "0xC1"
    assert action.BUTTONS["middle"] == "0xC2"


def test_visual_diff_typing_simulation():
    # Simulate an input box before and after typing
    box = (50, 100, 300, 40)
    img_before = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(img_before, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (180, 180, 180), 1)

    img_after = img_before.copy()
    cv2.putText(img_after, "user typed text", (box[0] + 10, box[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    res = compute_region_diff(img_before, img_after, region=box)
    assert res["changed"] is True
    assert res["changed_pixels"] > 0
