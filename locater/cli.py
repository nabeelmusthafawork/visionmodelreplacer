"""locater: precise pixel-location search inside a screenshot region."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from locater import match
from locater.capture import capture_screen


def _decode(data: bytes):
    arr = np.frombuffer(data, dtype="uint8")
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _parse_region(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(f"malformed --region {value!r} (expected X1,Y1,X2,Y2)")
    try:
        x1, y1, x2, y2 = (int(p.strip()) for p in parts)
    except ValueError:
        raise ValueError(f"malformed --region {value!r} (expected X1,Y1,X2,Y2)") from None
    return x1, y1, x2, y2


def _summary(info: dict) -> str:
    reasons: list[str] = []
    criteria = info.get("criteria", {})
    if not info.get("blobs"):
        reasons.append("no blobs")
    if "text" in criteria:
        reasons.append("OCR failed" if info.get("ocr_error") else "no text match")
    if not reasons:
        reasons.append("below threshold")
    return "; ".join(reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="locater",
        description="Find precise pixel coordinates of an element described by "
        "text/color/shape hints inside a screenshot region",
    )
    parser.add_argument("image", nargs="?", help="path to a screenshot (png/jpg)")
    parser.add_argument(
        "-s", "--screenshot", action="store_true", help="capture the whole screen"
    )
    parser.add_argument(
        "--region", metavar="X1,Y1,X2,Y2", help="pixel crop region (default: full image)"
    )
    parser.add_argument("--text", help="fuzzy text to search for")
    parser.add_argument("--color", metavar="HEX", help="hex color to match (#RRGGBB)")
    parser.add_argument(
        "--shape", choices=["circle", "square", "rect", "triangle"], help="shape to match"
    )
    parser.add_argument(
        "--tol", type=int, default=40, help="per-channel color tolerance (default: 40)"
    )
    parser.add_argument(
        "--fuzzy", type=float, default=0.75, help="minimum fuzzy text similarity (default: 0.75)"
    )
    parser.add_argument("--psm", type=int, default=11, help="tesseract page segmentation mode (default: 11)")
    parser.add_argument("--min-conf", type=float, default=30.0, help="minimum OCR confidence (default: 30)")
    parser.add_argument("--max", type=int, help="cap the number of returned matches")
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    parser.add_argument("--out", help="write output to a file instead of stdout")
    parser.add_argument("--debug", metavar="PREFIX", help="write debug JSON + annotated PNG (base name)")
    args = parser.parse_args(argv)

    if args.image and args.screenshot:
        raise SystemExit("error: image and --screenshot are mutually exclusive")
    if not (args.text or args.color or args.shape):
        raise SystemExit("error: specify at least one of --text, --color, --shape")

    if args.image:
        data = Path(args.image).read_bytes()
        bgr = _decode(data)
        if bgr is None:
            raise SystemExit(f"error: could not decode image: {args.image}")
    else:
        try:
            bgr = _decode(capture_screen())
        except RuntimeError as exc:
            raise SystemExit(f"error: screen capture failed: {exc}")
        if bgr is None:
            raise SystemExit("error: screen capture produced an undecodable image")
    img_h, img_w = bgr.shape[:2]

    region = None
    if args.region:
        try:
            region = match.clamp_region(*_parse_region(args.region), img_w, img_h)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
    if args.color:
        try:
            match.hex_to_rgb(args.color)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")

    matches, info = match.locate(
        bgr,
        region=region,
        text=args.text,
        color=args.color,
        shape=args.shape,
        tol=args.tol,
        fuzzy=args.fuzzy,
        psm=args.psm,
        min_conf=args.min_conf,
        max_n=args.max,
    )
    if info.get("ocr_error"):
        print(f"warning: OCR failed: {info['ocr_error']}", file=sys.stderr)

    if args.debug:
        debug = {
            "screen": {"width": img_w, "height": img_h},
            "region": info["region"],
            "criteria": info["criteria"],
            "blobs": info["blobs"],
            "ocr_error": info["ocr_error"],
            "matches": [m.to_dict() for m in matches],
        }
        Path(args.debug + ".json").write_text(json.dumps(debug, indent=2))
        annotated = bgr.copy()
        x1, y1, x2, y2 = info["region"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 1)
        for m in matches:
            x, y, w, h = m.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(annotated, m.center, 4, (0, 0, 255), -1)
        cv2.imwrite(args.debug + ".png", annotated)

    summary = _summary(info) if not matches else None
    if args.json:
        out = {
            "screen": {"width": img_w, "height": img_h},
            "region": info["region"],
            "matches": [m.to_dict() for m in matches],
        }
        if summary is not None:
            out["summary"] = summary
        output = json.dumps(out, indent=2)
    elif matches:
        lines = []
        for m in matches:
            parts = []
            for key in ("shape", "color", "text"):
                if key in m.criteria:
                    parts.append(m.criteria[key])
            x, y, w, h = m.bbox
            cx, cy = m.center
            lines.append(f"{m.score:.2f} {' '.join(parts)} at ({cx}, {cy}) [{x}, {y}, {w}, {h}]")
        output = "\n".join(lines)
    else:
        output = f"no matches: {summary}"

    if args.out:
        Path(args.out).write_text(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
