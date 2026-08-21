"""Merge OCR text lines with detected regions into a structured UI tree."""

from __future__ import annotations

from dataclasses import dataclass, field

from screendump import vision
from screendump.ocr import Line


@dataclass
class Element:
    kind: str  # window | panel | button | input | icon | text | separator
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    title: str = ""
    consumed: bool = False  # rendered by its parent (e.g. title bar)
    ui_type: str = ""  # semantic: sidebar | main | terminal | tabbar | toolbar | statusbar | tab | heading | menu | form | ...
    confidence: float | None = None  # OCR confidence of the attached text
    fill_ratio: float = 0.0  # interior darkness of the detected region
    children: list["Element"] = field(default_factory=list)
    rows: list[list["Element"]] = field(default_factory=list)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> int:
        return self.w * self.h


def _contains(outer: vision.Region, inner: vision.Region | Line, margin: float = 0.03) -> bool:
    mx = margin * outer.w
    my = margin * outer.h
    return (
        inner.x >= outer.x + mx
        and inner.y >= outer.y + my
        and inner.right <= outer.right - mx
        and inner.bottom <= outer.bottom - my
    )


def _interior_fill(region: vision.Region) -> bool:
    """Regions with dense dark interiors are filled buttons, not outlines."""
    return region.fill_ratio > 0.45


def _wordy(text: str) -> bool:
    """Real OCR text has at least two characters and one letter.

    Tesseract hallucinates single glyphs ('T', '3', '-') on dark
    gradient blobs; those are noise, not buttons.
    """
    return len(text) >= 2 and any(ch.isalpha() for ch in text)


def _similar_text(a: str, b: str) -> bool:
    """Two OCR lines with the same gist (the inverted pass re-reads the same
    text as a button label, so similar lines must be deduped)."""
    if not a or not b:
        return False
    ta, tb = a.split(), b.split()
    if len(ta) >= 2 and len(tb) >= 2:
        shared = len(set(ta) & set(tb)) / min(len(set(ta)), len(set(tb)))
        if shared >= 0.6:
            return True
    return a == b or a in b or b in a


def _y_overlap(a: vision.Region | Line, b: vision.Region | Line) -> bool:
    overlap = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return overlap >= 0.5 * min(a.h, b.h)


def _smallest_containing(
    target: vision.Region | Line, regions: list[vision.Region], margin: float = 0.03
) -> vision.Region | None:
    best = None
    for r in regions:
        if r.kind == "separator":
            continue
        if _contains(r, target, margin):
            if best is None or r.area < best.area:
                best = r
    return best


def _region_of(e: Element, regions: list[vision.Region]) -> vision.Region | None:
    for r in regions:
        if (r.x, r.y, r.w, r.h) == (e.x, e.y, e.w, e.h):
            return r
    return None


def build_tree(
    regions: list[vision.Region], lines: list[Line]
) -> list[Element]:
    """Turn regions + OCR lines into a forest of Elements."""
    elements: list[Element] = []
    for r in regions:
        if r.kind is None:
            continue
        el = Element(kind=r.kind, x=r.x, y=r.y, w=r.w, h=r.h, fill_ratio=r.fill_ratio)
        if r.kind in ("button", "input") and not _interior_fill(r):
            el.kind = "input"
        elif r.kind in ("button", "input") and _interior_fill(r):
            el.kind = "button"
        elements.append(el)

    # Attach each OCR line to the smallest containing region.
    for line in lines:
        holder = _smallest_containing(line, regions, margin=0.01)
        target = None
        if holder is not None:
            target = next(
                (e for e in elements if (e.x, e.y, e.w, e.h) == (holder.x, holder.y, holder.w, holder.h)),
                None,
            )
        if target is not None and target.kind in ("button", "input", "icon", "separator"):
            # control labels become the element's single text line; skip
            # near-duplicate OCR lines from the inverted pass
            if _similar_text(target.text, line.text):
                continue
            target.text = (target.text + " " + line.text).strip()
            target.confidence = max(target.confidence or 0, line.conf)
        elif target is not None:
            # free-floating text inside a window/panel: keep as child line
            if any(
                c.kind == "text" and _y_overlap(c, line) and _similar_text(c.text, line.text)
                for c in target.children
            ):
                continue
            target.children.append(
                Element(kind="text", x=line.x, y=line.y, w=line.w, h=line.h, text=line.text, confidence=line.conf)
            )
        else:
            elements.append(
                Element(kind="text", x=line.x, y=line.y, w=line.w, h=line.h, text=line.text, confidence=line.conf)
            )

    # Promote small boxes that ended up containing text into widgets.
    for e in elements:
        if e.kind in ("box", "panel") and e.text and e.w < 400:
            region = _region_of(e, regions)
            e.kind = "button" if (region is not None and _interior_fill(region)) else "input"
    # Downgrade empty buttons/inputs to icons or plain boxes.
    for e in elements:
        if e.kind in ("button", "input") and not e.text:
            e.kind = "icon" if e.w < 60 and e.h < 60 else "box"

    # Build parent/child relationships by containment.
    for e in elements:
        parent = _smallest_element_containing(e, elements)
        if parent is not None:
            parent.children.append(e)
            e._parent = parent

    roots = [e for e in elements if not hasattr(e, "_parent")]
    for e in elements:
        if hasattr(e, "_parent"):
            del e._parent

    _extract_titles(roots)
    for root in roots:
        _sort_and_group(root)
    return roots


def _extract_titles(roots: list[Element]) -> None:
    """Full-width strips near the top of a window become its title bar."""
    for root in roots:
        if root.kind != "window":
            continue
        for child in list(root.children):
            if child.kind not in ("input", "box", "panel"):
                continue
            if child.w < 0.85 * root.w:
                continue
            if child.h > max(40, 0.15 * root.h):
                continue
            if child.y - root.y > max(40, 0.2 * root.h):
                continue
            texts = [c for c in child.children if c.kind == "text"]
            if texts:
                root.title = texts[0].text
                child.consumed = True
                root.children.remove(child)


def _smallest_element_containing(
    target: Element, elements: list[Element]
) -> Element | None:
    best: Element | None = None
    for e in elements:
        if e is target:
            continue
        if e.kind in ("text", "separator"):
            continue
        if e.x < target.x and e.y < target.y and e.right > target.right and e.bottom > target.bottom:
            if best is None or e.area < best.area:
                best = e
    return best


def _sort_and_group(el: Element) -> None:
    """Sort children top-to-bottom, group into rows, sort within rows."""
    el.children.sort(key=lambda c: (c.y, c.x))
    rows: list[list[Element]] = []
    for child in el.children:
        placed = False
        for row in rows:
            anchor = row[0]
            if _y_overlap(child, anchor):
                row.append(child)
                placed = True
                break
        if not placed:
            rows.append([child])
    for row in rows:
        row.sort(key=lambda c: c.x)
    el.rows = rows
    for child in el.children:
        _sort_and_group(child)


def _y_overlap(a: Element, b: Element) -> bool:
    overlap = min(a.bottom, b.bottom) - max(a.y, b.y)
    return overlap >= 0.5 * min(a.h, b.h)


def flatten(roots: list[Element]) -> list[Element]:
    out: list[Element] = []

    def walk(e: Element) -> None:
        out.append(e)
        for c in e.children:
            walk(c)

    for r in roots:
        walk(r)
    return out