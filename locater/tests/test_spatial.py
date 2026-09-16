"""Tests for spatial relationship resolver (right-of, below, etc.)."""

import cv2
import numpy as np
import pytest
from locater.spatial import find_relative_element


def test_spatial_right_of():
    img = np.full((300, 600, 3), 255, dtype=np.uint8)

    # Draw label: "Email" at (50, 100)
    cv2.putText(img, "Email", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Draw an empty input rectangle to the right: (180, 100, 200, 35)
    cv2.rectangle(img, (180, 100), (380, 135), (100, 100, 100), 2)

    matches = find_relative_element(img, "right_of", "Email")
    assert len(matches) >= 1
    m = matches[0]
    rx, ry, rw, rh = m.bbox
    assert rx >= 170
    assert abs(ry - 100) <= 25


def test_spatial_below():
    img = np.full((400, 500, 3), 255, dtype=np.uint8)

    # Draw label: "Search" at (100, 50)
    cv2.putText(img, "Search", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Draw a box below: (100, 110, 250, 40)
    cv2.rectangle(img, (100, 110), (350, 150), (80, 80, 80), 2)

    matches = find_relative_element(img, "below", "Search")
    assert len(matches) >= 1
    m = matches[0]
    rx, ry, rw, rh = m.bbox
    assert ry >= 100
    assert abs(rx - 100) <= 30
