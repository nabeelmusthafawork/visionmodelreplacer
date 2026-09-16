"""Semantic classification of the UI element tree + JSON serialization.

Turns the geometric element tree from ``layout`` into a semantic UI map:
each container gets a role (sidebar, main, terminal, tabbar, toolbar,
statusbar, menu, form...) and each leaf gets a widget type (tab, heading,
button, input, ...). The same tree can be rendered as ASCII or dumped as
structured JSON for machine consumption.
"""

from __future__ import annotations

import re

from screendump.layout import Element, flatten

_CONTAINER_KINDS = ("window", "panel")

# Shell prompt / command-start heuristics for terminal detection.
_PROMPT_RE = re.compile(
    r"^\s*(?:"
    r"[$#>]\s"                                   # "$ ", "> ", "# "
    r"|[\w.\-]+@[\w.\-]+[:~][^\n]*[$#>]"         # "user@host:~$"
    r"|(?:ls|cd|sudo|apt|git|pip|python3?|rm|mkdir|cat|echo|cp|mv|vim|nano|curl|wget)\s"
    r")"
)


def classify(roots: list[Element]) -> None:
    """Assign a semantic ``ui_type`` to every element in the forest."""
    for root in roots:
        _classify_roles(root)
        _classify_widgets(root)


# -- container roles ---------------------------------------------------------

def _classify_roles(el: Element, parent: Element | None = None) -> None:
    for c in el.children:
        _classify_roles(c, el)
    if el.kind == "window":
        el.ui_type = "window"
    elif el.kind == "panel":
        el.ui_type = _panel_role(el, parent)


def _panel_role(el: Element, parent: Element | None) -> str:
    """Guess what kind of region a panel is, based on its geometry + content."""
    if parent is not None and el.w >= 0.6 * parent.w and el.h <= max(48, 0.14 * parent.h):
        # full-width horizontal strip
        if parent.bottom - el.bottom <= max(24, 0.08 * parent.h):
            return "statusbar"
        return "tabbar" if _tab_like(el) else "toolbar"

    if (
        parent is not None
        and el.w <= 0.45 * parent.w
        and el.h >= 0.5 * parent.h
        and (el.x - parent.x <= 0.08 * parent.w or parent.right - el.right <= 0.08 * parent.w)
    ):
        return "sidebar"

    if _terminal_like(el):
        return "terminal"

    if (
        parent is not None
        and el.w <= 0.35 * parent.w
        and el.h >= 0.3 * parent.h
        and sum(1 for c in el.children if c.kind == "text") >= 3
        and not any(c.kind in ("button", "input") for c in el.children)
    ):
        return "menu"

    if sum(1 for c in el.children if c.kind == "input") >= 2:
        return "form"

    siblings = [c for c in (parent.children if parent else []) if c.kind == "panel"]
    if siblings and el is max(siblings, key=lambda c: c.area):
        return "main"

    return "panel"


def _tab_like(el: Element) -> bool:
    small = [c for c in el.children if c.h <= 48 and c.text]
    return len(small) >= 2


def _terminal_like(el: Element) -> bool:
    texts = [c.text for c in el.children if c.kind == "text" and c.text]
    if any(_PROMPT_RE.match(t) for t in texts):
        return True
    return el.fill_ratio > 0.6 and bool(texts)


# -- widget classification ---------------------------------------------------

def _classify_widgets(el: Element) -> None:
    """Classify children of *el*: row of tabs near the top, headings by size."""
    top_strip = max(44, 0.10 * el.h)
    for row in el.rows:
        if row and min(c.y for c in row) - el.y <= top_strip:
            small = [c for c in row if c.kind in ("button", "input", "box") and c.h <= 44]
            if len(small) >= 2:
                for c in small:
                    c.ui_type = "tab"
                continue
            if el.kind == "window":
                texts = [c for c in row if c.kind in ("text", "box") and c.h <= 48 and len(c.text) <= 30]
                if len(texts) >= 2:
                    for c in texts:
                        c.ui_type = "tab"

    heights = [c.h for c in el.children if c.kind == "text"]
    if heights:
        median = sorted(heights)[len(heights) // 2]
        for c in el.children:
            if c.kind == "text" and c.h >= 1.6 * median:
                c.ui_type = "heading"

    for c in el.children:
        _classify_widgets(c)


# -- JSON serialization ------------------------------------------------------

def to_dict(roots: list[Element], img_w: int, img_h: int) -> dict:
    """Structured, machine-readable view of the detected screen."""
    all_elements = flatten(roots)
    windows = [
        {
            "id": r.window_id or f"W{i + 1}",
            "title": r.title or "",
            "type": r.ui_type or r.kind,
            "bbox": [r.x, r.y, r.w, r.h],
            "elements": [
                _element_dict(e) for e in flatten([r]) if e.kind not in _CONTAINER_KINDS
            ],
        }
        for i, r in enumerate(roots)
        if r.kind == "window"
    ]
    return {
        "screen": {"width": img_w, "height": img_h},
        "windows": windows,
        "regions": [_region_dict(r) for r in roots],
        "elements": [_element_dict(e) for e in all_elements if e.kind not in _CONTAINER_KINDS],
    }


def _region_dict(el: Element) -> dict:
    return {
        "type": el.ui_type or el.kind,
        "bbox": [el.x, el.y, el.w, el.h],
        "title": el.title or "",
        "elements": [
            _element_dict(c) for c in el.children if c.kind not in _CONTAINER_KINDS
        ],
        "children": [
            _region_dict(c) for c in el.children if c.kind in _CONTAINER_KINDS
        ],
    }


def _element_dict(el: Element) -> dict:
    return {
        "type": el.ui_type or el.kind,
        "text": el.text,
        "bbox": [el.x, el.y, el.w, el.h],
        "confidence": el.confidence,
    }
