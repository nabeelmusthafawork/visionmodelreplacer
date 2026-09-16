"""Tests for screendump dump pipeline including fast mode."""

import cv2
import numpy as np
import pytest
from screendump.dump import dump_image


def test_dump_image_basic():
    img = np.full((300, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (200, 80), (200, 200, 200), -1)
    cv2.putText(img, "Submit", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Standard JSON dump
    data = dump_image(img, is_json=True, fast=False)
    assert isinstance(data, dict)
    assert "screen" in data
    assert data["screen"]["width"] == 600

    # Fast mode dump
    data_fast = dump_image(img, is_json=True, fast=True)
    assert isinstance(data_fast, dict)
    assert "screen" in data_fast
    assert data_fast["screen"]["width"] == 600
