"""OCR via the tesseract CLI (no python wrapper dependency needed)."""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable

TESSERACT_BIN = os.environ.get("TESSERACT_BIN") or shutil.which("tesseract")


@dataclass(frozen=True)
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class Line:
    """A logical text line: words merged by tesseract's block/par/line ids."""

    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float
    words: tuple[Word, ...]

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


class TesseractError(RuntimeError):
    pass


def _tesseract_cmd(png_bytes: bytes, extra_args: Iterable[str]) -> bytes:
    if not TESSERACT_BIN:
        raise TesseractError(
            "tesseract binary not found; install it or set TESSERACT_BIN"
        )
    proc = subprocess.run(
        [TESSERACT_BIN, "stdin", "stdout", *extra_args],
        input=png_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise TesseractError(
            f"tesseract failed ({proc.returncode}): "
            + proc.stderr.decode("utf-8", "replace")[:500]
        )
    return proc.stdout


def _normalize(text: str) -> str:
    """Clean tesseract artifacts: typographic quotes, stray bytes, spaces."""
    out = []
    for ch in text:
        if ch in "\u2018\u2019\u201a\u2032":
            ch = "'"
        elif ch in "\u201c\u201d\u201e\u2033":
            ch = '"'
        elif ch == "\u00a0":
            ch = " "
        if ch.isprintable() and (ch == " " or not ch.isspace()):
            out.append(ch)
    return " ".join("".join(out).split())


def ocr_words(png_bytes: bytes, psm: int = 11, min_conf: float = 30.0) -> list[Word]:
    """Run tesseract with TSV output; return word-level results with geometry."""
    tsv = _tesseract_cmd(
        png_bytes, ["--psm", str(psm), "tsv"]
    ).decode("utf-8", "replace")
    words: list[Word] = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        text = _normalize(row.get("text") or "")
        if not text:
            continue
        try:
            conf = float(row["conf"])
            x, y, w, h = (int(row[k]) for k in ("left", "top", "width", "height"))
        except (KeyError, ValueError):
            continue
        if conf < min_conf or w <= 0 or h <= 0:
            continue
        words.append(Word(text=text, x=x, y=y, w=w, h=h, conf=conf))
    return words


def group_lines(words: list[Word], splits: Iterable[int] | None = None) -> list[Line]:
    """Group word boxes into lines using tesseract's block/par/line grouping.

    Words arrive in reading order from tesseract; consecutive words that
    belong to the same line (overlap in y, reasonable x gap, and not crossing
    window splits) are merged.
    """
    if not words:
        return []
    split_set = tuple(splits) if splits else ()
    lines: list[Line] = []
    current: list[Word] = [words[0]]
    prev = words[0]

    def flush() -> None:
        if not current:
            return
        x = min(w.x for w in current)
        y = min(w.y for w in current)
        right = max(w.right for w in current)
        bottom = max(w.bottom for w in current)
        conf = sum(w.conf for w in current) / len(current)
        text = " ".join(w.text for w in current)
        lines.append(
            Line(
                text=text,
                x=x,
                y=y,
                w=right - x,
                h=bottom - y,
                conf=conf,
                words=tuple(current),
            )
        )

    for word in words[1:]:
        y_overlap = min(prev.bottom, word.bottom) - max(prev.y, word.y)
        max_gap = max(32, int(2.5 * max(prev.h, word.h)))
        x_gap = word.x - prev.right
        crosses_split = any(prev.right <= s <= word.x for s in split_set)
        same_line = (
            y_overlap >= 0.3 * min(prev.h, word.h)
            and abs(word.y - prev.y) <= 0.6 * max(prev.h, word.h)
            and 0 <= x_gap <= max_gap
            and not crosses_split
        )
        if same_line:
            current.append(word)
        else:
            flush()
            current = [word]
        prev = word
    flush()
    return lines


def ocr_lines(png_bytes: bytes, psm: int = 11, min_conf: float = 30.0) -> list[Line]:
    return group_lines(ocr_words(png_bytes, psm=psm, min_conf=min_conf))