"""flow.py: Autonomous Multi-Step Action Pipeline for v6.

Executes end-to-end multi-step agent plans in a single local execution loop,
collapsing multi-turn conversational back-and-forth into a single sub-3-second call.

Supported Step Types:
  - nav: URL (focuses URL bar, types URL, hits enter)
  - click: query (text, coordinates, or spatial)
  - type: text (with optional enter=true, verify=true)
  - wait: query (polls local frame until target appears or timeout)
  - key: key combination (ctrl+l, enter, esc, etc.)
  - scroll: ticks (dy=-5)
  - dump: captures final screen representation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import cv2
import numpy as np

from locater import action, match, spatial
from locater.capture import capture_screen
from screendump import dump, layout, vision
from screendump.diff import compute_region_diff, find_dirty_rects


def _decode(raw: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


class FlowExecutor:
    """High-speed chained action pipeline."""

    def __init__(self, default_window: str | None = None, verbose: bool = False):
        self.default_window = default_window
        self.verbose = verbose
        self.last_frame: np.ndarray | None = None
        self.execution_log: list[dict[str, Any]] = []

    def get_frame(self) -> np.ndarray:
        raw = capture_screen()
        bgr = _decode(raw)
        if bgr is None:
            raise RuntimeError("Failed to capture and decode screen frame")
        self.last_frame = bgr
        return bgr

    def get_window_region(self, bgr: np.ndarray, window_id: str) -> tuple[int, int, int, int] | None:
        regs = vision.detect_regions(bgr)
        roots = layout.build_tree(regs, [])
        windows = [r for r in roots if r.kind == "window"]
        w_query = window_id.strip().lower()
        for w in windows:
            if (
                w.window_id.lower() == w_query
                or w.window_id.lower() == f"w{w_query}"
                or w_query in (w.title or "").lower()
            ):
                return (w.x, w.y, w.right, w.bottom)
        return None

    def execute_step(self, step: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        act = step.get("action", "").lower()
        step_window = step.get("window", self.default_window)
        res: dict[str, Any] = {"action": act, "status": "ok"}

        if act == "nav":
            # Fast navigation: focuses address bar, writes URL, sends enter
            url = step.get("url") or step.get("target")
            if not url:
                raise ValueError("nav step requires 'url'")
            # If target window is given, click near top of window to focus it
            bgr = self.get_frame()
            if step_window:
                w_reg = self.get_window_region(bgr, step_window)
                if w_reg:
                    # Click address bar area in that window (typically ~50px down from top, centered)
                    wx1, wy1, wx2, wy2 = w_reg
                    addr_x = wx1 + (wx2 - wx1) // 3
                    addr_y = wy1 + 50
                    action.click(addr_x, addr_y)
                    time.sleep(0.08)
            action.press_key("ctrl+l")
            time.sleep(0.08)
            action.press_key("ctrl+a")
            time.sleep(0.05)
            action.type_text(url)
            time.sleep(0.05)
            action.press_key("enter")
            res["navigated_to"] = url

        elif act == "click":
            x = step.get("x")
            y = step.get("y")
            text = step.get("text")
            settle = float(step.get("settle", 0.15))
            button = step.get("button", "left")

            if x is not None and y is not None:
                action.click(int(x), int(y), button=button)
                res["clicked_coords"] = [int(x), int(y)]
            elif text:
                bgr = self.get_frame()
                region = self.get_window_region(bgr, step_window) if step_window else None
                matches, _ = match.locate(
                    bgr,
                    region=region,
                    text=text,
                    fuzzy=float(step.get("fuzzy", 0.65)),
                    max_n=1,
                )
                if not matches:
                    raise RuntimeError(f"click failed: element with text {text!r} not found")
                top = matches[0]
                cx, cy = top.center
                action.click(cx, cy, button=button)
                res["clicked_text"] = text
                res["target"] = [cx, cy]
                res["bbox"] = list(top.bbox)
            else:
                raise ValueError("click step requires either [x, y] or 'text'")
            time.sleep(settle)

        elif act == "type":
            text = step.get("text", "")
            x = step.get("x")
            y = step.get("y")
            press_enter = bool(step.get("enter", False))
            verify = bool(step.get("verify", False))

            if x is not None and y is not None:
                action.click(int(x), int(y))
                time.sleep(0.08)

            t_info = action.type_text(text, verify=verify)
            res["type_result"] = t_info
            if press_enter:
                time.sleep(0.05)
                action.press_key("enter")
                res["pressed_enter"] = True

        elif act == "wait":
            # Ultra-fast local polling loop
            text = step.get("text")
            timeout = float(step.get("timeout", 6.0))
            poll_interval = float(step.get("interval", 0.15))
            start_t = time.perf_counter()
            found = False

            while time.perf_counter() - start_t < timeout:
                bgr = self.get_frame()
                region = self.get_window_region(bgr, step_window) if step_window else None
                matches, _ = match.locate(
                    bgr,
                    region=region,
                    text=text,
                    fuzzy=float(step.get("fuzzy", 0.65)),
                    max_n=1,
                )
                if matches:
                    found = True
                    res["found"] = True
                    res["wait_time_seconds"] = round(time.perf_counter() - start_t, 3)
                    res["target"] = list(matches[0].center)
                    res["bbox"] = list(matches[0].bbox)
                    break
                time.sleep(poll_interval)

            if not found:
                raise TimeoutError(f"wait timed out after {timeout}s: element {text!r} not found")

        elif act == "key":
            combo = step.get("combo") or step.get("key")
            if not combo:
                raise ValueError("key step requires 'combo' or 'key'")
            action.press_key(combo)
            res["pressed"] = combo

        elif act == "scroll":
            dy = int(step.get("dy", -5))
            action.scroll(dy)
            res["scrolled_ticks"] = dy

        elif act == "sleep":
            sec = float(step.get("duration", 0.5))
            time.sleep(sec)
            res["slept_seconds"] = sec

        res["duration_seconds"] = round(time.perf_counter() - t0, 3)
        self.execution_log.append(res)
        return res

    def run_flow(self, steps: list[dict[str, Any]], post_dump: bool = True) -> dict[str, Any]:
        t0 = time.perf_counter()
        results: list[dict[str, Any]] = []
        error: str | None = None

        for idx, step in enumerate(steps):
            try:
                out = self.execute_step(step)
                results.append(out)
            except Exception as exc:
                error = f"Step {idx} ({step.get('action')}): {exc}"
                results.append({"step_index": idx, "action": step.get("action"), "error": str(exc), "status": "failed"})
                break

        final_dump = None
        if post_dump and error is None:
            bgr = self.get_frame()
            final_dump = dump.dump_image(
                bgr,
                is_json=True,
                window=self.default_window,
                mode="targeted",
            )

        total_elapsed = round(time.perf_counter() - t0, 3)
        return {
            "ok": error is None,
            "total_elapsed_seconds": total_elapsed,
            "steps_executed": len(results),
            "step_results": results,
            "error": error,
            "final_dump": final_dump,
        }


def parse_step_string(s: str) -> dict[str, Any]:
    """Parse concise step format, e.g.:

    'nav:https://example.com'
    'click:text=Start a post,settle=0.2'
    'type:Hello world!,enter=true'
    'wait:text=Submit,timeout=5'
    """
    if ":" not in s:
        return {"action": s.strip()}
    act, rest = s.split(":", 1)
    act = act.strip().lower()
    step: dict[str, Any] = {"action": act}

    if act == "nav" and "=" not in rest:
        step["url"] = rest.strip()
        return step

    if act == "type":
        text_val = rest
        if ",enter=true" in text_val:
            step["enter"] = True
            text_val = text_val.replace(",enter=true", "")
        if ",verify=true" in text_val:
            step["verify"] = True
            text_val = text_val.replace(",verify=true", "")
        if text_val.startswith("text="):
            text_val = text_val[5:]
        step["text"] = text_val.strip()
        return step

    # Key-value options separated by comma
    parts = rest.split(",")
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k, v = k.strip(), v.strip()
            if v.lower() == "true":
                step[k] = True
            elif v.lower() == "false":
                step[k] = False
            else:
                try:
                    step[k] = int(v) if v.isdigit() else float(v)
                except ValueError:
                    step[k] = v
        else:
            # Positional fallback
            if act in ("click", "wait", "type") and "text" not in step:
                step["text"] = p.strip()
    return step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="flow",
        description="Autonomous Multi-Step Action Pipeline (v6)",
    )
    parser.add_argument(
        "--step",
        dest="steps",
        action="append",
        help="action step specification (e.g. 'nav:https://...', 'click:text=Post', 'type:Hello')",
    )
    parser.add_argument("--json-flow", help="JSON string or path to JSON file containing step list")
    parser.add_argument("--window", help="default window scope (e.g. W1, W2, W3)")
    parser.add_argument("--no-dump", action="store_true", help="do not dump final screen state")
    parser.add_argument("-j", "--json", action="store_true", help="output structured JSON")
    args = parser.parse_args(argv)

    steps: list[dict[str, Any]] = []
    if args.json_flow:
        raw_str = args.json_flow
        if os.path.exists(raw_str):
            raw_str = open(raw_str).read()
        loaded = json.loads(raw_str)
        steps = loaded if isinstance(loaded, list) else loaded.get("steps", [])
    elif args.steps:
        for s in args.steps:
            steps.append(parse_step_string(s))
    else:
        raise SystemExit("error: specify at least one --step or a --json-flow file")

    executor = FlowExecutor(default_window=args.window)
    result = executor.run_flow(steps, post_dump=not args.no_dump)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Flow {'COMPLETED' if result['ok'] else 'FAILED'} in {result['total_elapsed_seconds']}s ({result['steps_executed']} steps)")
        for idx, res in enumerate(result.get("step_results", [])):
            print(f"  Step {idx+1}: {res.get('action')} -> {res.get('status')} ({res.get('duration_seconds', 0)}s)")
        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            return 1
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
