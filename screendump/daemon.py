"""visiond: high-performance background daemon for screendump & locater.

Keeps OpenCV, OCR pipelines, and Wayland automation warm in memory to eliminate
process startup latency and module import overhead. Listens on a Unix domain socket.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from locater import action, match
from screendump import capture, dump


def get_socket_path() -> Path:
    env = os.environ.get("VISIOND_SOCKET")
    if env:
        return Path(env)
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and Path(runtime_dir).exists():
        return Path(runtime_dir) / "visiond.sock"
    return Path(f"/tmp/visiond-{os.getuid()}.sock")


def _decode(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype="uint8")
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class VisionDaemon:
    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self.running = False
        self.sock: socket.socket | None = None

    def start(self, foreground: bool = True) -> None:
        if self.socket_path.exists():
            # Check if an existing daemon is alive
            if self.ping():
                print(f"visiond: already running on {self.socket_path}")
                return
            self.socket_path.unlink(missing_ok=True)

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(self.socket_path))
        self.sock.listen(16)
        self.running = True

        import threading
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._handle_signal)
                signal.signal(signal.SIGTERM, self._handle_signal)
            except ValueError:
                pass

        print(f"visiond: listening on {self.socket_path}")

        while self.running:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                break
            with conn:
                try:
                    data = conn.recv(65536).decode("utf-8")
                    if not data:
                        continue
                    req = json.loads(data.strip())
                    resp = self.handle_request(req)
                    conn.sendall(json.dumps(resp).encode("utf-8") + b"\n")
                except Exception as exc:
                    err_resp = {"ok": False, "error": str(exc)}
                    try:
                        conn.sendall(json.dumps(err_resp).encode("utf-8") + b"\n")
                    except Exception:
                        pass

    def _handle_signal(self, signum, frame):
        self.stop()

    def stop(self) -> None:
        self.running = False
        if self.sock:
            self.sock.close()
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)
        print("visiond: stopped")

    @classmethod
    def ping(cls, socket_path: Path | None = None) -> bool:
        sp = socket_path or get_socket_path()
        if not sp.exists():
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(str(sp))
                s.sendall(json.dumps({"cmd": "ping"}).encode("utf-8") + b"\n")
                res = s.recv(1024).decode("utf-8")
                return json.loads(res).get("ok") is True
        except Exception:
            return False

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any]:
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "time": time.time()}

        if cmd == "dump":
            raw = capture.capture_screen()
            bgr = _decode(raw)
            if bgr is None:
                return {"ok": False, "error": "failed to decode screenshot"}
            is_json = req.get("json", False)
            fast = req.get("fast", False)
            targeted = req.get("targeted", False)
            mode = req.get("mode")
            window = req.get("window")
            ascii_chars = req.get("ascii", False)
            psm = req.get("psm", 11)
            min_conf = req.get("min_conf", 30.0)
            res = dump.dump_image(
                bgr,
                is_json=is_json,
                fast=fast,
                targeted=targeted,
                mode=mode,
                window=window,
                ascii_chars=ascii_chars,
                psm=psm,
                min_conf=min_conf,
            )
            return {"ok": True, "result": res}

        if cmd == "locate":
            raw = capture.capture_screen()
            bgr = _decode(raw)
            if bgr is None:
                return {"ok": False, "error": "failed to decode screenshot"}
            img_h, img_w = bgr.shape[:2]
            region = req.get("region")
            target_window = None

            if req.get("window"):
                from screendump import layout, vision
                regs = vision.detect_regions(bgr)
                roots = layout.build_tree(regs, [])
                win_candidates = [r for r in roots if r.kind == "window"]
                w_query = str(req["window"]).strip().lower()
                for w in win_candidates:
                    if (
                        w.window_id.lower() == w_query
                        or w.window_id.lower() == f"w{w_query}"
                        or w_query in (w.title or "").lower()
                    ):
                        target_window = w
                        break
                if target_window is not None:
                    region = (target_window.x, target_window.y, target_window.right, target_window.bottom)

            spatial_relation = None
            anchor_text = None
            for k in ("right_of", "below", "left_of", "above"):
                if req.get(k):
                    spatial_relation, anchor_text = k, req[k]
                    break

            if spatial_relation and anchor_text:
                from locater import spatial
                matches = spatial.find_relative_element(
                    bgr,
                    relation=spatial_relation,
                    anchor_text=anchor_text,
                    target_shape=req.get("shape"),
                    target_color=req.get("color"),
                    region=region,
                    tol=req.get("tol", 40),
                    fuzzy=req.get("fuzzy", 0.75),
                )
                info = {"region": region or [0, 0, img_w, img_h], "criteria": {"spatial": f"{spatial_relation} '{anchor_text}'"}}
            else:
                matches, info = match.locate(
                    bgr,
                    region=region,
                    text=req.get("text"),
                    color=req.get("color"),
                    shape=req.get("shape"),
                    tol=req.get("tol", 40),
                    fuzzy=req.get("fuzzy", 0.75),
                    psm=req.get("psm", 11),
                    min_conf=req.get("min_conf", 30.0),
                    max_n=req.get("max"),
                )

            action_res = None
            if matches and req.get("action"):
                act = req["action"]
                top = matches[0]
                tx, ty = top.center
                if act == "click":
                    action.click(tx, ty, button=req.get("button", "left"))
                    action_res = {"action": "click", "target": [tx, ty]}
                elif act == "double_click":
                    action.double_click(tx, ty)
                    action_res = {"action": "double_click", "target": [tx, ty]}
                elif act in ("type", "type_and_enter"):
                    type_str = req.get("text_to_type", "")
                    action.click(tx, ty, button="left")
                    time.sleep(0.08)
                    t_info = action.type_text(type_str, target_bbox=top.bbox, verify=req.get("verify", False))
                    action_res = {"action": act, "target": [tx, ty], "details": t_info}
                    if act == "type_and_enter":
                        time.sleep(0.05)
                        action.press_key("enter")

            post_dump = None
            if req.get("dump"):
                time.sleep(max(0, req.get("settle", 200)) / 1000.0)
                post_raw = capture.capture_screen()
                post_bgr = _decode(post_raw)
                if post_bgr is not None:
                    post_dump = dump.dump_image(
                        post_bgr,
                        is_json=req.get("dump_json", True),
                        fast=req.get("fast", False),
                        targeted=req.get("targeted", False),
                        mode=req.get("mode"),
                        window=req.get("window"),
                    )

            return {
                "ok": True,
                "screen": {"width": img_w, "height": img_h},
                "region": info["region"],
                "matches": [m.to_dict() for m in matches],
                "action": action_res,
                "dump": post_dump,
            }

        if cmd == "act":
            act = req.get("action")
            if act == "click":
                action.click(req.get("x"), req.get("y"), button=req.get("button", "left"))
                return {"ok": True, "action": "click"}
            if act == "scroll":
                action.scroll(req.get("dy", -5))
                return {"ok": True, "action": "scroll"}
            if act == "type":
                res = action.type_text(
                    req.get("text", ""),
                    target_bbox=req.get("bbox"),
                    verify=req.get("verify", False),
                )
                return {"ok": True, "action": "type", "details": res}
            if act == "key":
                action.press_key(req.get("combo", ""))
                return {"ok": True, "action": "key"}
            return {"ok": False, "error": f"unknown action {act!r}"}

        if cmd == "dirty_rects":
            from screendump.diff import find_dirty_rects
            raw = capture.capture_screen()
            bgr = _decode(raw)
            if bgr is None:
                return {"ok": False, "error": "failed to decode screenshot"}
            last = getattr(self, "_last_bgr", None)
            self._last_bgr = bgr
            if last is None or last.shape != bgr.shape:
                return {"ok": True, "dirty_rects": []}
            rects = find_dirty_rects(last, bgr, min_area=req.get("min_area", 100))
            return {"ok": True, "dirty_rects": rects}

        return {"ok": False, "error": f"unknown cmd {cmd!r}"}


def send_daemon_request(req: dict[str, Any], timeout: float = 30.0) -> dict[str, Any] | None:
    """Send request to running daemon; returns response dict or None if daemon unreachable."""
    sp = get_socket_path()
    if not sp.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(sp))
            s.sendall(json.dumps(req).encode("utf-8") + b"\n")
            chunks = []
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if chunk.endswith(b"\n"):
                    break
            raw = b"".join(chunks).decode("utf-8")
            return json.loads(raw.strip())
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="visiond",
        description="visionmodelreplacer background daemon service",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start", help="start the daemon in foreground")
    sub.add_parser("stop", help="stop the running daemon")
    sub.add_parser("status", help="check daemon status")
    sub.add_parser("ping", help="ping daemon")
    args = parser.parse_args(argv)

    sp = get_socket_path()

    if args.command in (None, "start"):
        daemon = VisionDaemon(sp)
        try:
            daemon.start()
        except KeyboardInterrupt:
            daemon.stop()
        return 0

    if args.command == "stop":
        if not sp.exists():
            print("visiond: not running")
            return 0
        resp = send_daemon_request({"cmd": "ping"})
        if resp:
            # We can unlink the socket or send a kill signal
            sp.unlink(missing_ok=True)
            print("visiond: stopped socket")
        else:
            sp.unlink(missing_ok=True)
            print("visiond: cleaned up stale socket")
        return 0

    if args.command in ("status", "ping"):
        alive = VisionDaemon.ping(sp)
        if alive:
            print(f"visiond: active and running on {sp}")
            return 0
        else:
            print("visiond: inactive")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
