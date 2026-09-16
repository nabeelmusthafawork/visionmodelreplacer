"""Render the semantic UI tree as a compact labeled map.

The renderer draws regions as labeled frames (``┌─ SIDEBAR ──┐``), tags
widgets by type (``[button] Settings``, ``[input] Search``, ``[tab] 1``,
``[heading] Title``) and lays sibling panels out side by side so horizontal
relationships survive (sidebar on the left, terminal on the right).
"""

from __future__ import annotations

from screendump.layout import Element, _sort_and_group, _wordy

_BOX = dict(
    tl="┌", tr="┐", bl="└", br="┘",
    h="─", v="│", lt="├", rt="┤",
)
_ASCII_BOX = dict(
    tl="+", tr="+", bl="+", br="+",
    h="-", v="|", lt="+", rt="+",
)

_CHROME = ("tabbar", "toolbar", "statusbar")
_GUTTER = 1  # blank column between side-by-side panels


def _noise(el: Element) -> bool:
    """Textless boxes/icons add no information; drop them."""
    if el.children:
        return False
    if el.kind in ("box", "icon") and not el.text:
        return True
    return bool(el.text) and not _wordy(el.text)


def _looks_like_tab_strip(text: str) -> list[str] | None:
    """A single long OCR line of short tokens is usually a merged tab bar."""
    tokens = [t for t in text.split() if _word_like(t)]
    if len(tokens) >= 6 and len(text) >= 30 and max(len(t) for t in tokens) <= 24:
        return tokens
    return None


def _word_like(tok: str) -> bool:
    """Keep real words; drop OCR junk like '<<C', '@', '0O', '::54'."""
    if len(tok) < 2 or not any(ch.isalpha() for ch in tok) or not tok[0].isalnum():
        return False
    # 2-char tokens built only from look-alike glyphs ('0O', '1l', '||')
    # are tab-icon residue, not words.
    if len(tok) <= 2 and all(ch in "0Ol1I|" for ch in tok):
        return False
    return True


class Renderer:
    def __init__(self, img_w: int, img_h: int, width: int = 80, ascii_chars: bool = False):
        self.img_w = max(1, img_w)
        self.img_h = max(1, img_h)
        self.width = max(20, width)
        self.box = _ASCII_BOX if ascii_chars else _BOX

    # -- coordinate mapping ------------------------------------------------
    def col(self, x: float) -> int:
        return int(x / self.img_w * (self.width - 2))

    def render(self, roots: list[Element]) -> str:
        lines: list[str] = []
        frames = [r for r in roots if r.kind in ("window", "panel")]
        strays = [r for r in roots if r.kind not in ("window", "panel") and not _noise(r)]

        # Check if desktop has top/bottom full-width bars (e.g. status bar / dock)
        top_bars = [
            r for r in frames
            if r.w >= 0.75 * self.img_w and r.h <= max(64, int(0.15 * self.img_h)) and r.y <= 0.15 * self.img_h
        ]
        bottom_bars = [
            r for r in frames
            if r.w >= 0.75 * self.img_w and r.h <= max(64, int(0.15 * self.img_h)) and r.bottom >= 0.85 * self.img_h
        ]
        app_frames = [r for r in frames if r not in top_bars and r not in bottom_bars]

        for bar in top_bars:
            if lines:
                lines.append("")
            lines.extend(self._render_element(bar))

        if len(app_frames) == 1:
            if lines:
                lines.append("")
            lines.extend(self._render_element(app_frames[0]))
        elif len(app_frames) > 1:
            sorted_apps = sorted(app_frames, key=lambda e: (e.y, e.x))
            rows: list[list[Element]] = []
            for app in sorted_apps:
                placed = False
                for row in rows:
                    if self._y_overlap_app(app, row[0]):
                        row.append(app)
                        placed = True
                        break
                if not placed:
                    rows.append([app])
            for row in rows:
                if lines:
                    lines.append("")
                if len(row) == 1:
                    lines.extend(self._render_element(row[0]))
                else:
                    lines.extend(self._render_columns(row, self.width))

        for bar in bottom_bars:
            if lines:
                lines.append("")
            lines.extend(self._render_element(bar))

        if (frames or top_bars or app_frames) and strays:
            lines.append("")
        if strays:
            desktop = Element(
                kind="window", x=0, y=0, w=self.img_w, h=self.img_h, title="DESKTOP"
            )
            desktop.children = sorted(strays, key=lambda e: (e.y, e.x))
            _sort_and_group(desktop)
            lines.extend(self._render_element(desktop))
        return "\n".join(lines)

    @staticmethod
    def _y_overlap_app(a: Element, b: Element) -> bool:
        overlap = min(a.bottom, b.bottom) - max(a.y, b.y)
        return overlap >= 0.3 * min(a.h, b.h)

    # -- elements ------------------------------------------------------------
    def _render_element(self, el: Element) -> list[str]:
        if _noise(el):
            return []
        if el.kind == "text":
            return [self._pad_line(self._text_str(el), self.col(el.x))]
        if el.kind in ("button", "input", "icon", "box", "separator", "tab"):
            return [self._pad_line(self._widget_str(el), self.col(el.x))]
        if el.kind in ("window", "panel"):
            return self._render_frame(el)
        return [self._pad_line(el.text or "?", self.col(el.x))]

    # -- frames --------------------------------------------------------------
    def _label(self, el: Element) -> str:
        if el.kind == "window":
            return el.title or "WINDOW"
        return (el.ui_type or "panel").upper()

    def _render_frame(self, el: Element, inner_w: int | None = None) -> list[str]:
        b = self.box
        inner_w = inner_w if inner_w is not None else self.width - 2
        label = f"{b['h']} {self._label(el)} {b['h']}"
        if len(label) > inner_w - 2:
            label = label[: inner_w - 2]
        out: list[str] = [f"{b['tl']}{label}{b['h'] * (inner_w - len(label))}{b['tr']}"]
        if el.kind == "window":
            out.append(f"{b['lt']}{b['h'] * inner_w}{b['rt']}")

        content = self._render_content(el, inner_w)
        for line in content:
            line = line[:inner_w]
            out.append(f"{b['v']}{line}{' ' * (inner_w - len(line))}{b['v']}")

        out.append(f"{b['bl']}{b['h'] * inner_w}{b['br']}")
        return out

    # -- content -------------------------------------------------------------
    def _render_content(self, el: Element, inner_w: int) -> list[str]:
        out: list[str] = []
        chrome: list[str] = []
        containers: list[Element] = []
        flat_rows: list[list[Element]] = []

        for row in el.rows:
            row = [c for c in row if not c.consumed]
            if not row:
                continue
            if len(row) == 1 and row[0].ui_type in _CHROME:
                chrome.append(self._chrome_line(row[0], inner_w))
                continue
            if el.kind == "window":
                top_strip = max(44, 0.10 * el.h)
                if min(c.y for c in row) - el.y <= top_strip:
                    flats = [c for c in row if c.kind not in ("window", "panel") and not _noise(c)]
                    if flats and all(c.ui_type == "tab" for c in flats):
                        chrome.append(self._tabs_line([c.text for c in flats], inner_w))
                        continue
                    if all(c.kind == "text" for c in flats):
                        tokens = _looks_like_tab_strip(" ".join(c.text for c in flats))
                        if tokens:
                            chrome.append(self._tabs_line(tokens, inner_w))
                            continue
            conts = [c for c in row if c.kind in ("window", "panel")]
            flats = [c for c in row if c.kind not in ("window", "panel") and not _noise(c)]
            if flats:
                flat_rows.append(flats)
            containers.extend(conts)

        out.extend(chrome)
        for flats in flat_rows:
            out.append(self._render_row(flats, el, inner_w))
        out.extend(self._render_columns(containers, inner_w))
        return out

    def _render_columns(self, containers: list[Element], inner_w: int) -> list[str]:
        if not containers:
            return []
        columns = self._group_columns(containers)
        available = inner_w - _GUTTER * (len(columns) - 1)
        if available <= 0:
            return []
        widths = [max(m.w for m in col) for col in columns]
        total = sum(widths)
        widths = [max(6, round(available * w / total)) for w in widths]
        # fix overflow from rounding, shaving the widest columns first
        while sum(widths) > available:
            widths[widths.index(max(widths))] -= 1

        rendered = []
        for col, w in zip(columns, widths):
            col = sorted(col, key=lambda c: (c.y, c.x))
            lines: list[str] = []
            for c in col:
                sub = Renderer(c.w, c.h, width=w, ascii_chars=self.box is _ASCII_BOX)
                lines.extend(sub._render_frame(c))
            rendered.append(lines)
        return self._join_columns(rendered, inner_w)

    def _group_columns(self, containers: list[Element]) -> list[list[Element]]:
        """Group sibling containers into x-columns; panels side by side land
        in different columns, panels stacked vertically share one."""
        columns: list[list[Element]] = []
        for c in sorted(containers, key=lambda e: (e.x, e.y)):
            for col in columns:
                if any(self._x_overlap(c, m) for m in col):
                    col.append(c)
                    break
            else:
                columns.append([c])
        return columns

    @staticmethod
    def _x_overlap(a: Element, b: Element) -> bool:
        overlap = min(a.right, b.right) - max(a.x, b.x)
        return overlap >= 0.5 * min(a.w, b.w)

    def _join_columns(self, rendered: list[list[str]], inner_w: int) -> list[str]:
        height = max(len(r) for r in rendered)
        padded: list[list[str]] = []
        for lines in rendered:
            if not lines:
                continue
            width = len(lines[0])
            blank = f"{self.box['v']}{' ' * max(0, width - 2)}{self.box['v']}" if width >= 2 else " " * width
            needed = height - len(lines)
            if needed > 0 and len(lines) >= 2:
                padded.append(lines[:-1] + [blank] * needed + [lines[-1]])
            else:
                padded.append(lines + [blank] * needed)
        return [
            " ".join(lines).strip()[:inner_w]
            for lines in zip(*padded)
        ]

    # -- chrome strips ---------------------------------------------------------
    def _tabs_line(self, tokens: list[str], inner_w: int) -> str:
        tokens = [t for t in tokens if t]
        # OCR often doubles a tab strip (re-read by the inverted pass):
        # drop consecutive repeats and whole-sequence repeats.
        out: list[str] = []
        for t in tokens:
            if out and t == out[-1]:
                continue
            out.append(t)
        tokens = out
        if len(tokens) >= 4 and len(tokens) % 2 == 0:
            half = len(tokens) // 2
            if tokens[:half] == tokens[half:]:
                tokens = tokens[:half]
        return self._trunc("TABS: " + " | ".join(tokens), inner_w)

    def _chrome_line(self, el: Element, inner_w: int) -> str:
        labels = [c.text for c in el.children if _wordy(c.text)]
        if el.ui_type == "statusbar":
            return self._trunc(f"STATUS: {' '.join(labels)}", inner_w)
        if el.ui_type == "tabbar":
            return self._trunc("TABS: " + (" | ".join(labels) if labels else "\u2014"), inner_w)
        parts = [self._widget_str(c) for c in el.children if not _noise(c)]
        return self._trunc("TOOLBAR: " + " ".join(parts), inner_w)

    # -- rows and widgets -------------------------------------------------------
    def _render_row(self, row: list[Element], parent: Element, inner_w: int) -> str:
        cursor = 0
        parts: list[str] = []
        for child in row:
            if _noise(child):
                continue
            target = self.col(child.x)
            if target <= cursor:
                target = cursor + 1
            parts.append(" " * (target - cursor))
            parts.append(self._widget_str(child))
            cursor = target + len(parts[-1])
        return "".join(parts)[:inner_w]

    def _text_str(self, el: Element) -> str:
        if el.ui_type == "heading":
            return f"[heading] {el.text}"
        return el.text

    def _widget_str(self, el: Element) -> str:
        """Render one widget with its type tag (positioning is external)."""
        b = self.box

        if el.kind == "text":
            return self._text_str(el)

        if el.ui_type == "tab":
            return f"[{el.text}]" if el.text else "[tab]"

        if el.kind == "icon":
            return "[icon]"

        if el.kind == "button":
            return f"[button] {el.text}" if el.text else "[button]"

        if el.kind == "input":
            return f"[input] {el.text}" if el.text else "[input]"

        if el.kind == "box":
            return f"[box] {el.text}" if el.text else "[box]"

        if el.kind == "separator":
            return b["h"] * max(3, self.col(el.w))

        return el.text or "?"

    def _pad_line(self, text: str, col: int) -> str:
        if len(text) + col >= self.width - 1:
            return text[: self.width - 1]
        return " " * col + text

    @staticmethod
    def _trunc(text: str, width: int) -> str:
        if len(text) > width:
            return text[: width - 1] + "\u2026"
        return text


def render(roots: list[Element], img_w: int, img_h: int, width: int = 80, ascii_chars: bool = False) -> str:
    return Renderer(img_w, img_h, width=width, ascii_chars=ascii_chars).render(roots)