"""Unit test for render/layout/semantic that doesn't require cv2/tesseract.

Runs under any python3 (stdlib only) by stubbing the vision module.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import cv2  # noqa: F401
    from screendump import vision
except ImportError:
    vision = types.ModuleType("screendump.vision")
    vision.Region = types.SimpleNamespace
    sys.modules["screendump.vision"] = vision

from screendump import layout, render, semantic  # noqa: E402


def el(kind, x, y, w, h, text="", fill_ratio=0.0):
    return layout.Element(kind=kind, x=x, y=y, w=w, h=h, text=text, fill_ratio=fill_ratio)


def test_browser_window():
    window = el("window", 20, 20, 640, 380)
    window.children = [
        el("text", 38, 30, 60, 20, "Firefox"),
        el("icon", 38, 74, 24, 24),
        el("icon", 70, 74, 24, 24),
        el("input", 104, 70, 316, 26, "https://example.com"),
        el("separator", 20, 110, 640, 2),
        el("text", 260, 140, 160, 24, "Example Website"),
        el("input", 250, 190, 180, 42, "Search"),
        el("button", 250, 260, 80, 34, "Login"),
        el("button", 350, 260, 88, 34, "Settings"),
    ]
    layout._sort_and_group(window)
    semantic.classify([window])

    out = render.Renderer(680, 420, width=80).render([window])
    print(out)

    assert out.splitlines()[0].startswith("\u250c")
    assert "Firefox" in out
    assert "https://example.com" in out
    assert "[button] Login" in out
    assert "[button] Settings" in out
    assert "[input] Search" in out
    assert "separator" not in out  # separators render as plain dashes


def test_ascii_mode():
    button = el("button", 250, 260, 80, 34, "Login")
    out = render.Renderer(680, 420, width=40, ascii_chars=True).render([button])
    print(out)
    assert "+- DESKTOP" in out
    assert "[button] Login" in out


def test_side_by_side_panels():
    window = el("window", 0, 0, 1000, 600)
    sidebar = el("panel", 0, 0, 250, 600)
    sidebar.children = [
        el("text", 30, 40, 150, 18, "Files"),
        el("text", 30, 70, 150, 18, "Projects"),
        el("input", 20, 110, 200, 30, "Search files"),
    ]
    main = el("panel", 250, 0, 750, 600)
    main.children = [
        el("text", 300, 40, 300, 18, "Welcome back"),
        el("button", 300, 80, 120, 32, "Open"),
        el("button", 430, 80, 120, 32, "New"),
    ]
    window.children = [sidebar, main]
    layout._sort_and_group(window)
    semantic.classify([window])

    out = render.Renderer(1000, 600, width=100).render([window])
    print(out)

    assert "SIDEBAR" in out and "MAIN" in out
    header = next(l for l in out.splitlines() if "SIDEBAR" in l and "MAIN" in l)
    assert "\u250c\u2500 SIDEBAR" in header
    assert "\u250c\u2500 MAIN" in header  # both frames on one line: side by side


def test_semantic_classification():
    window = el("window", 0, 0, 1200, 800)
    tabs = el("panel", 0, 0, 1200, 40)
    tabs.children = [el("box", 5, 5, 150, 30, "Tab A"), el("box", 160, 5, 150, 30, "Tab B")]
    sidebar = el("panel", 0, 40, 200, 760)
    sidebar.children = [el("text", 20, 60, 100, 16, "One"), el("text", 20, 90, 100, 16, "Two")]
    term = el("panel", 200, 40, 1000, 760, fill_ratio=0.8)
    term.children = [
        el("text", 220, 60, 500, 16, "$ apt update"),
        el("text", 220, 85, 500, 16, "nabeel@arch:~$"),
    ]
    window.children = [tabs, sidebar, term]
    layout._sort_and_group(window)
    semantic.classify([window])

    by_type = {c.ui_type: c for c in window.children}
    assert by_type["tabbar"] is tabs
    assert by_type["sidebar"] is sidebar
    assert by_type["terminal"] is term


def test_json_output():
    window = el("window", 20, 20, 640, 380)
    sidebar = el("panel", 30, 60, 180, 300)
    sidebar.children = [el("text", 40, 70, 100, 16, "Files", )]
    widget = el("button", 250, 260, 80, 34, "Login")
    window.children = [sidebar, widget]
    layout._sort_and_group(window)
    semantic.classify([window])

    data = semantic.to_dict([window], 680, 420)
    print(json.dumps(data, indent=2))

    assert data["screen"] == {"width": 680, "height": 420}
    assert data["regions"][0]["type"] == "window"
    region = data["regions"][0]["children"][0]
    assert region["type"] == "sidebar"
    assert region["bbox"] == [30, 60, 180, 300]
    elem = data["elements"][1]
    assert elem["type"] in ("button", "tab")
    assert elem["text"] == "Login"
    assert elem["bbox"] == [250, 260, 80, 34]
    json.dumps(data, indent=2)  # must be serializable


def test_chrome_tabs_and_desktop():
    window = el("window", 0, 0, 900, 600)
    window.children = [
        el("text", 5, 5, 140, 24, "Terminal 2"),
        el("text", 150, 5, 140, 24, "Terminal 1"),
        el("box", 300, 5, 40, 24, "+"),
    ]
    layout._sort_and_group(window)
    semantic.classify([window])

    out = render.Renderer(900, 600, width=80).render([window])
    print(out)
    assert "TABS:" in out and "Terminal 2" in out and "Terminal 1" in out

    stray = el("text", 40, 40, 300, 20, "[nabeel@arch dumbstuff]$")
    out2 = render.Renderer(900, 600, width=80).render([window, stray])
    print(out2)
    assert "DESKTOP" in out2
    assert "[nabeel@arch dumbstuff]$" in out2


def test_no_line_cap():
    window = el("window", 0, 0, 900, 900)
    window.children = [el("text", 10, 10 + i * 30, 400, 20, f"line {i}") for i in range(40)]
    layout._sort_and_group(window)
    out = render.Renderer(900, 900, width=80).render([window])
    assert "line 39" in out  # nothing truncated or hidden
    assert "more lines" not in out


def test_dedupe_duplicate_ocr_lines():
    from screendump import ocr

    class R(types.SimpleNamespace):
        pass

    button = R(
        kind="button", x=10, y=10, w=120, h=30, fill_ratio=0.8,
        text="", right=130, bottom=40, area=3600, cx=70, cy=25,
    )
    line1 = ocr.Line("Reinstalled it now runs the code", 12, 12, 100, 20, 70.0, ())
    line2 = ocr.Line("Reinstalled it now runs", 12, 12, 100, 20, 60.0, ())

    roots = layout.build_tree([button], [line1, line2])
    assert len(roots) == 1
    text = roots[0].text
    assert text.count("Reinstalled") == 1  # inverted-pass duplicate dropped


def test_multi_window_side_by_side_rendering():
    w1 = el("window", 0, 30, 960, 1050)
    w1.children = [
        el("text", 20, 50, 150, 20, "nabeel@arch:~$"),
        el("text", 20, 90, 200, 20, "$ git status"),
    ]
    w2 = el("window", 960, 30, 960, 1050)
    w2.children = [
        el("input", 980, 70, 400, 30, "https://github.com"),
        el("button", 980, 120, 80, 30, "Search"),
    ]
    top_bar = el("panel", 0, 0, 1920, 30)
    top_bar.children = [el("text", 900, 5, 120, 20, "sep 16 2:16PM")]

    roots = [top_bar, w1, w2]
    layout._sort_and_group(w1)
    layout._sort_and_group(w2)
    layout._sort_and_group(top_bar)
    layout._tag_window_identities(roots)
    semantic.classify(roots)

    out = render.Renderer(1920, 1080, width=120).render(roots)
    print("\n" + out)

    assert "PANEL" in out or "STATUSBAR" in out
    assert "[W1]" in out and "[W2]" in out
    # Both windows must be on the same horizontal lines (side-by-side)
    window_headers = [line for line in out.splitlines() if "[W1]" in line and "[W2]" in line]
    assert len(window_headers) == 1, "Expected both [W1] and [W2] headers on the same line"
    # Both windows must have their bottom borders aligned
    bottom_borders = [line for line in out.splitlines() if line.count("\u2518") >= 2]
    assert len(bottom_borders) >= 1, "Expected side-by-side bottom borders aligned on same line"


def test_multi_window_json_structure():
    w1 = el("window", 0, 30, 960, 1050)
    w1.children = [el("text", 20, 50, 150, 20, "nabeel@arch:~$")]
    w2 = el("window", 960, 30, 960, 1050)
    w2.children = [el("button", 980, 120, 80, 30, "Login")]
    roots = [w1, w2]
    layout._sort_and_group(w1)
    layout._sort_and_group(w2)
    layout._tag_window_identities(roots)
    semantic.classify(roots)

    data = semantic.to_dict(roots, 1920, 1080)
    assert "windows" in data
    assert len(data["windows"]) == 2
    assert data["windows"][0]["id"] == "W1"
    assert data["windows"][1]["id"] == "W2"
    assert data["windows"][0]["bbox"] == [0, 30, 960, 1050]
    assert data["windows"][1]["bbox"] == [960, 30, 960, 1050]


if __name__ == "__main__":
    test_browser_window()
    test_ascii_mode()
    test_side_by_side_panels()
    test_semantic_classification()
    test_json_output()
    test_chrome_tabs_and_desktop()
    test_no_line_cap()
    test_dedupe_duplicate_ocr_lines()
    test_multi_window_side_by_side_rendering()
    test_multi_window_json_structure()
    print("ALL TESTS PASSED")