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
    parser.add_argument(
        "--window",
        metavar="ID_OR_NAME",
        help="scope search to a window by ID (e.g. W1, W2) or title (e.g. Terminal, Browser)",
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
    parser.add_argument("--right-of", dest="right_of", help="find element situated to the right of text label")
    parser.add_argument("--below", dest="below", help="find element situated directly below text label")
    parser.add_argument("--left-of", dest="left_of", help="find element situated to the left of text label")
    parser.add_argument("--above", dest="above", help="find element situated directly above text label")
    parser.add_argument("--scroll-until", type=int, default=0, help="maximum scroll attempts if target not found (default: 0)")
    parser.add_argument("--scroll-dy", type=int, default=-5, help="wheel ticks per scroll attempt (default: -5)")
    parser.add_argument("--click", action="store_true", help="click the top matched element")
    parser.add_argument("--double-click", action="store_true", help="double click the top matched element")
    parser.add_argument("--button", default="left", choices=["left", "right", "middle"], help="mouse button (default: left)")
    parser.add_argument("--type", dest="type_text", help="click the top matched element and type text")
    parser.add_argument("--type-and-enter", dest="type_and_enter", help="click top element, type text, and press enter")
    parser.add_argument("--key", dest="key_combo", help="press key combination after matching (e.g. 'enter')")
    parser.add_argument("--verify", action="store_true", help="verify visual change after typing or clicking")
    parser.add_argument("--dump", action="store_true", help="output post-action screen representation")
    parser.add_argument("--settle", type=int, default=200, help="ms to sleep before post-action dump (default: 200)")
    parser.add_argument(
        "--mode",
        choices=["auto", "targeted", "fast", "full"],
        default=None,
        help="perception mode for post-action dump",
    )
    parser.add_argument("--fast", action="store_true", help="use fast layout mode for post-action dump")
    parser.add_argument("--targeted", action="store_true", help="use targeted box OCR mode for post-action dump")
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    parser.add_argument("--out", help="write output to a file instead of stdout")
    parser.add_argument("--debug", metavar="PREFIX", help="write debug JSON + annotated PNG (base name)")
    args = parser.parse_args(argv)

    if args.image and args.screenshot:
        raise SystemExit("error: image and --screenshot are mutually exclusive")
    if not (
        args.text
        or args.color
        or args.shape
        or args.right_of
        or args.below
        or args.left_of
        or args.above
    ):
        raise SystemExit(
            "error: specify at least one search criterion (--text, --color, --shape, --right-of, --below, --left-of, --above)"
        )

    # Fast-path: query background daemon if running and no debug/image/window scoping given
    if not args.image and not args.debug and not args.window:
        from screendump.daemon import send_daemon_request
        act = None
        if args.double_click:
            act = "double_click"
        elif args.click:
            act = "click"
        elif args.type_text:
            act = "type"

        req = {
            "cmd": "locate",
            "region": list(_parse_region(args.region)) if args.region else None,
            "text": args.text,
            "color": args.color,
            "shape": args.shape,
            "tol": args.tol,
            "fuzzy": args.fuzzy,
            "psm": args.psm,
            "min_conf": args.min_conf,
            "max": args.max,
            "action": act,
            "button": args.button,
            "text_to_type": args.type_text,
            "verify": args.verify,
            "dump": args.dump,
            "settle": args.settle,
            "dump_json": args.json,
            "fast": args.fast,
        }
        resp = send_daemon_request(req)
        if resp and resp.get("ok"):
            if args.json:
                out_str = json.dumps(resp, indent=2)
            else:
                matches = resp.get("matches", [])
                lines = []
                for m in matches:
                    parts = [m.get("criteria", {}).get(k, "") for k in ("shape", "color", "text") if k in m.get("criteria", {})]
                    x, y, w, h = m["bbox"]
                    cx, cy = m["center"]
                    lines.append(f"{m['score']:.2f} {' '.join(filter(None, parts))} at ({cx}, {cy}) [{x}, {y}, {w}, {h}]")
                if resp.get("action"):
                    lines.append(f"-> action: {resp['action']}")
                if resp.get("dump"):
                    lines.append("\n" + str(resp["dump"]))
                out_str = "\n".join(lines) if lines else "no matches"
            if args.out:
                Path(args.out).write_text(out_str + "\n")
            else:
                print(out_str)
            return 0

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

    target_window = None
    if args.window:
        from screendump import layout, vision
        regs = vision.detect_regions(bgr)
        roots = layout.build_tree(regs, [])
        windows = [r for r in roots if r.kind == "window"]
        w_query = args.window.strip().lower()
        for w in windows:
            if (
                w.window_id.lower() == w_query
                or w.window_id.lower() == f"w{w_query}"
                or w_query in (w.title or "").lower()
            ):
                target_window = w
                break
        if target_window is None:
            avail = ", ".join(f"{w.window_id} ({w.title})" for w in windows) or "none detected"
            raise SystemExit(f"error: window {args.window!r} not found. Available windows: {avail}")

    region = None
    if target_window is not None:
        region = (
            target_window.x,
            target_window.y,
            target_window.x + target_window.w,
            target_window.y + target_window.h,
        )

    if args.region:
        try:
            user_region = match.clamp_region(*_parse_region(args.region), img_w, img_h)
            if region is not None:
                rx1, ry1, rx2, ry2 = region
                ux1, uy1, ux2, uy2 = user_region
                region = (max(rx1, ux1), max(ry1, uy1), min(rx2, ux2), min(ry2, uy2))
            else:
                region = user_region
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
    if args.color:
        try:
            match.hex_to_rgb(args.color)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")

    info = {"region": region or [0, 0, img_w, img_h], "criteria": {}}
    spatial_relation = None
    anchor_text = None
    if args.right_of:
        spatial_relation, anchor_text = "right_of", args.right_of
    elif args.below:
        spatial_relation, anchor_text = "below", args.below
    elif args.left_of:
        spatial_relation, anchor_text = "left_of", args.left_of
    elif args.above:
        spatial_relation, anchor_text = "above", args.above

    if spatial_relation and anchor_text:
        from locater import spatial
        matches = spatial.find_relative_element(
            bgr,
            relation=spatial_relation,
            anchor_text=anchor_text,
            target_shape=args.shape,
            target_color=args.color,
            region=region,
            tol=args.tol,
            fuzzy=args.fuzzy,
        )
        info["criteria"]["spatial"] = f"{spatial_relation} '{anchor_text}'"
    else:
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

    # Auto-scroll loop if not found and scroll_until requested
    if not matches and args.scroll_until > 0 and not args.image:
        from locater import action
        for _ in range(args.scroll_until):
            action.scroll(args.scroll_dy)
            import time
            time.sleep(0.3)
            new_raw = capture_screen()
            bgr = _decode(new_raw)
            if bgr is None:
                break
            if spatial_relation and anchor_text:
                matches = spatial.find_relative_element(
                    bgr,
                    relation=spatial_relation,
                    anchor_text=anchor_text,
                    target_shape=args.shape,
                    target_color=args.color,
                    region=region,
                    tol=args.tol,
                    fuzzy=args.fuzzy,
                )
            else:
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
            if matches:
                break
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

    if target_window is not None:
        info["window"] = {
            "id": target_window.window_id,
            "title": target_window.title,
            "bbox": [target_window.x, target_window.y, target_window.w, target_window.h],
        }

    action_result = None
    has_act = (
        args.click
        or args.double_click
        or args.type_text
        or args.type_and_enter
        or args.key_combo
    )
    if matches and has_act:
        from locater import action
        top = matches[0]
        tx, ty = top.center
        action_info = {}
        if args.double_click:
            action.double_click(tx, ty)
            action_info["action"] = "double_click"
            action_info["target"] = [tx, ty]
        elif args.click or args.type_text or args.type_and_enter:
            action.click(tx, ty, button=args.button)
            action_info["action"] = "click"
            action_info["button"] = args.button
            action_info["target"] = [tx, ty]

        if args.type_text:
            import time
            time.sleep(0.08)
            type_res = action.type_text(args.type_text, target_bbox=top.bbox, verify=args.verify)
            action_info["type"] = type_res

        if args.type_and_enter:
            import time
            time.sleep(0.08)
            type_res = action.type_text(args.type_and_enter, target_bbox=top.bbox, verify=args.verify)
            action_info["type"] = type_res
            time.sleep(0.05)
            action.press_key("enter")
            action_info["key"] = "enter"

        if args.key_combo:
            import time
            time.sleep(0.05)
            action.press_key(args.key_combo)
            action_info["key"] = args.key_combo

        action_result = action_info

    post_dump = None
    if args.dump:
        import time
        time.sleep(max(0, args.settle) / 1000.0)
        from screendump.capture import capture_screen as _cap
        from screendump.dump import dump_image
        post_raw = _cap()
        post_bgr = _decode(post_raw)
        if post_bgr is not None:
            post_dump = dump_image(
                post_bgr,
                is_json=args.json,
                fast=args.fast,
                targeted=args.targeted,
                mode=args.mode,
                window=args.window,
            )

    summary = _summary(info) if not matches else None
    if args.json:
        out = {
            "screen": {"width": img_w, "height": img_h},
            "region": info["region"],
            "matches": [m.to_dict() for m in matches],
        }
        if "window" in info:
            out["window"] = info["window"]
        if action_result is not None:
            out["action"] = action_result
        if post_dump is not None:
            out["dump"] = post_dump
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
        if action_result is not None:
            lines.append(f"-> action: {action_result}")
        if post_dump is not None:
            lines.append("\n" + str(post_dump))
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
