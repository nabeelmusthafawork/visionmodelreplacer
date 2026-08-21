"""Unit tests for locater.match on a synthetic image (PIL only, no real screen).

Runs under any python3 that can import cv2/numpy. OCR-dependent tests are
skipped when tesseract is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from locater import match
from screendump import ocr

W, H = 1280, 720


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def _build() -> np.ndarray:
    img = Image.new("RGB", (W, H), "#e8e8e8")
    d = ImageDraw.Draw(img)
    d.ellipse([160, 110, 240, 190], fill="#FF7A00")
    d.rounded_rectangle([400, 300, 520, 420], radius=16, fill="#3B82F6")
    d.rectangle([700, 250, 900, 320], fill="#22C55E")
    d.text((300, 550), "Hello World", font=_font(40), fill="#000000")
    return np.array(img)[:, :, ::-1].copy()


def test_color_only():
    bgr = _build()
    matches, info = match.locate(bgr, region=(100, 80, 300, 220), color="#FF7A00")
    assert matches, "expected at least one orange blob"
    m = matches[0]
    assert abs(m.center[0] - 200) <= 3 and abs(m.center[1] - 150) <= 3
    assert abs(m.bbox[0] - 160) <= 4 and abs(m.bbox[1] - 110) <= 4
    assert abs(m.bbox[2] - 80) <= 6 and abs(m.bbox[3] - 80) <= 6
    assert m.score >= 0.9
    assert info["criteria"] == {"color": "#FF7A00"}


def test_color_shape_circle():
    bgr = _build()
    matches, _ = match.locate(
        bgr, region=(100, 80, 550, 450), color="#FF7A00", shape="circle"
    )
    assert len(matches) == 1, f"expected only the orange circle, got {len(matches)}"
    m = matches[0]
    assert m.criteria == {"color": "#FF7A00", "shape": "circle"}
    assert abs(m.center[0] - 200) <= 3 and abs(m.center[1] - 150) <= 3
    assert m.score >= 0.9


def test_text_fuzzy():
    if ocr.TESSERACT_BIN is None:
        print("SKIP test_text_fuzzy (tesseract not installed)")
        return
    bgr = _build()
    matches, info = match.locate(
        bgr, region=(250, 520, 900, 620), text="hlo", fuzzy=0.5
    )
    assert matches, "expected a fuzzy text match for 'hlo'"
    m = matches[0]
    assert "text" in m.criteria
    assert 520 <= m.center[1] <= 620
    assert 250 <= m.center[0] <= 900
    assert m.confidence > 0


def test_region_clamped():
    bgr = _build()
    region = match.clamp_region(-500, -500, 99999, 99999, W, H)
    assert region == (0, 0, W, H)
    matches, info = match.locate(bgr, region=region, color="#FF7A00")
    assert matches
    assert info["region"] == [0, 0, W, H]
    m = matches[0]
    assert abs(m.center[0] - 200) <= 3 and abs(m.center[1] - 150) <= 3


def test_region_invalid():
    try:
        match.clamp_region(100, 100, 50, 200, W, H)
        assert False, "expected ValueError for an inverted region"
    except ValueError:
        pass


def test_no_target_in_region():
    bgr = _build()
    matches, info = match.locate(bgr, region=(1000, 100, 1200, 200), color="#FF7A00")
    assert matches == []
    assert info["blobs"] == []


if __name__ == "__main__":
    for name in sorted(n for n in globals() if n.startswith("test_")):
        fn = globals()[name]
        fn()
        print(f"PASS {name}")
    print("ALL TESTS PASSED")
