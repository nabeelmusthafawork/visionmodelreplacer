"""Tests for targeted / lazy OCR pipeline."""

import cv2
import numpy as np
import pytest
from screendump import targeted_ocr, vision


def test_targeted_ocr_extraction():
    img = np.full((300, 600, 3), 255, dtype=np.uint8)

    # Draw two buttons with text
    cv2.rectangle(img, (50, 50), (180, 100), (220, 220, 220), -1)
    cv2.rectangle(img, (50, 50), (180, 100), (50, 50, 50), 2)
    cv2.putText(img, "Save", (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.rectangle(img, (250, 50), (380, 100), (220, 220, 220), -1)
    cv2.rectangle(img, (250, 50), (380, 100), (50, 50, 50), 2)
    cv2.putText(img, "Cancel", (270, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    regions = vision.detect_regions(img)
    lines = targeted_ocr.extract_targeted_lines(img, regions, min_conf=20.0)

    # Targeted lines should have found text in the boxes
    assert len(lines) >= 1
    texts = [l.text.lower() for l in lines]
    assert any("save" in t or "cancel" in t for t in texts)
