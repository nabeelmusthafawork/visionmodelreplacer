"""screendump: convert a screenshot into a structured text UI representation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from screendump import capture, layout, ocr, render, semantic, vision


def _decode(data: bytes):
    arr = np.frombuffer(data, dtype="uint8")
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to encode PNG")
    return buf.tobytes()


def _ocr_all(
    bgr: np.ndarray, regions: list[vision.Region], psm: int, min_conf: float
) -> list[ocr.Line]:
    """OCR the full image, plus an inverted pass for dark filled buttons."""
    lines = ocr.ocr_lines(_png_bytes(bgr), psm=psm, min_conf=min_conf)

    for r in regions:
        if r.kind not in ("button",):
            continue
        if r.fill_ratio < 0.45:
            # light buttons: labels are read fine by the global pass
            continue
        if any(_overlap(line, r) for line in lines):
            # the global pass already read this region; don't re-read it
            continue
        pad = 4
        x0, y0 = max(0, r.x - pad), max(0, r.y - pad)
        x1, y1 = r.right + pad, r.bottom + pad
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        inverted = cv2.bitwise_not(crop)
        extra = ocr.ocr_lines(_png_bytes(inverted), psm=psm, min_conf=min_conf * 0.8)
        for line in extra:
            lines.append(
                ocr.Line(
                    text=line.text,
                    x=line.x + x0,
                    y=line.y + y0,
                    w=line.w,
                    h=line.h,
                    conf=line.conf,
                    words=line.words,
                )
            )

    # Focused pass: browser tab strips are small (~11px) text that
    # sparse-text mode often misses; upscale and re-OCR them as a
    # single line. Browser chrome sits ABOVE the detected window box
    # (the strip is outside it), so when the window starts near the
    # screen top, crop the band above it instead.
    img_h = bgr.shape[0]
    for w in regions:
        if w.kind != "window":
            continue
        strip_h = min(64, max(24, int(w.h * 0.06)))
        near_top = 0 < w.y < 0.15 * img_h
        if near_top:
            y0 = max(0, w.y - 60)
            y1 = y0 + 56
        else:
            y0, y1 = w.y, w.y + strip_h
        crop = bgr[y0:y1, w.x : w.right]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        extra = ocr.ocr_lines(_png_bytes(big), psm=7, min_conf=20)
        for line in extra:
            lines.append(
                ocr.Line(
                    text=line.text,
                    x=w.x + line.x // 2,
                    # pin the line inside the window top so it lands in
                    # the window's tab strip chrome line
                    y=w.y + 2 if near_top else y0 + line.y // 2,
                    w=line.w // 2,
                    h=line.h // 2,
                    conf=line.conf,
                    words=(),
                )
            )

    # Dark-on-dark desktops often yield no window region at all (the
    # terminal has no visible border). The tab bar is then never read,
    # so fall back to OCRing the full-width top band ourselves. Tab
    # text on dark themes is light-on-dark: try the inverted pass if
    # the normal one finds nothing word-like.
    if not any(r.kind == "window" for r in regions):
        top = bgr[0:64, :]
        if top.size:
            best: list[ocr.Line] = []
            for crop in (top, cv2.bitwise_not(top)):
                big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                extra = ocr.ocr_lines(_png_bytes(big), psm=7, min_conf=18)
                if sum(layout._wordy(l.text) for l in extra) > sum(layout._wordy(l.text) for l in best):
                    best = extra
            for line in best:
                lines.append(
                    ocr.Line(
                        text=line.text,
                        x=line.x // 2,
                        y=line.y // 2,
                        w=line.w // 2,
                        h=line.h // 2,
                        conf=line.conf,
                        words=(),
                    )
                )

    # The crop passes re-read the same visual text; keep only the
    # highest-confidence copy of any overlapping, similar-sounding line.
    # Lines with no real word content are tesseract noise, not UI text.
    deduped: list[ocr.Line] = []
    for line in sorted(lines, key=lambda l: l.conf, reverse=True):
        if not layout._wordy(line.text):
            continue
        if any(layout._similar_text(line.text, prev.text) and _overlap(line, prev) for prev in deduped):
            continue
        deduped.append(line)
    return deduped


def _overlap(a, b) -> bool:
    """Line vs line, or line vs region rect (b has x/y/w/h attrs)."""
    y = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    x = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    return y >= 0.5 * min(a.h, b.h) and x >= 0.3 * min(a.w, b.w)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="screendump",
        description="Convert a screenshot into a text-based UI representation",
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="path to a screenshot (png/jpg); omit to capture the screen",
    )
    parser.add_argument(
        "-s",
        "--screenshot",
        action="store_true",
        help="capture the screen instead of reading an image",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="output structured JSON instead of the ASCII map",
    )
    parser.add_argument("--ascii", action="store_true", help="use pure ASCII box characters")
    parser.add_argument("--psm", type=int, default=11, help="tesseract page segmentation mode (default: 11)")
    parser.add_argument("--min-conf", type=float, default=30.0, help="minimum OCR confidence (default: 30)")
    parser.add_argument("--out", help="write the text representation to a file instead of stdout")
    parser.add_argument("--debug", help="write detected boxes JSON + annotated image (base name)", metavar="PREFIX")
    args = parser.parse_args(argv)

    if args.screenshot and args.image:
        raise SystemExit("error: give either --screenshot or an image path, not both")
    if args.image:
        data = Path(args.image).read_bytes()
    else:
        try:
            data = capture.capture_screen()
        except capture.CaptureError as exc:
            raise SystemExit(f"error: {exc}")
    bgr = _decode(data)
    if bgr is None:
        raise SystemExit(f"error: could not decode image: {args.image}")
    img_h, img_w = bgr.shape[:2]

    debug: dict | None = {} if args.debug else None
    regions = vision.detect_regions(bgr, debug=debug)
    try:
        lines = _ocr_all(bgr, regions, psm=args.psm, min_conf=args.min_conf)
    except ocr.TesseractError as exc:
        print(f"warning: OCR failed: {exc}", file=sys.stderr)
        lines = []
    roots = layout.build_tree(regions, lines)
    semantic.classify(roots)

    if args.debug:
        json_debug = {k: v for k, v in debug.items() if k != "annotated"}
        Path(args.debug + ".json").write_text(json.dumps(json_debug, indent=2))
        cv2.imwrite(args.debug + ".png", debug["annotated"])

    if args.json:
        output = json.dumps(semantic.to_dict(roots, img_w, img_h), indent=2)
    else:
        width = shutil.get_terminal_size((80, 24)).columns
        output = render.render(roots, img_w, img_h, width=width, ascii_chars=args.ascii)
    if args.out:
        Path(args.out).write_text(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())