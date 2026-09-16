"""Action dispatcher for UI automation (mouse clicks, typing, key combinations).

Interacts directly with ydotool (for cursor movement and clicks) and vtype
(for uinput-level keystrokes), with optional pre/post visual verification.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

from screendump.capture import capture_screen
from screendump.diff import verify_text_input

_DIV = 2  # ydotool coordinates scaling factor on Wayland
_VTYPE = os.path.expanduser("~/.local/bin/vtype")
_SOCKET = os.environ.get("YDOTOOL_SOCKET", "/run/ydotool/.ydotool_socket")

BUTTONS = {
    "left": "0xC0",
    "right": "0xC1",
    "middle": "0xC2",
}


def _ydotool(*args: str) -> None:
    env = dict(os.environ)
    env["YDOTOOL_SOCKET"] = _SOCKET
    try:
        subprocess.run(["ydotool", *args], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ydotool failed ({exc.returncode}); is ydotoold running at {_SOCKET}?") from exc
    except FileNotFoundError:
        raise RuntimeError("ydotool executable not found on PATH") from None


def move(x: int, y: int) -> None:
    """Move mouse pointer to logical display coordinates."""
    _ydotool("mousemove", "-a", "-x", str(int(x) // _DIV), "-y", str(int(y) // _DIV))


def click(x: int | None = None, y: int | None = None, button: str = "left") -> None:
    """Move (optional) and click mouse button (left, right, middle)."""
    if x is not None and y is not None:
        move(x, y)
        time.sleep(0.05)
    btn_code = BUTTONS.get(button.lower(), "0xC0")
    _ydotool("click", btn_code)


def double_click(x: int | None = None, y: int | None = None) -> None:
    """Perform a rapid double click."""
    click(x, y, button="left")
    time.sleep(0.08)
    click(None, None, button="left")


def scroll(dy: int = -5) -> None:
    """Scroll mouse wheel (e.g. dy=-5 for scroll down, dy=+5 for scroll up)."""
    _ydotool("mousemove", "-w", "-y", str(dy))


def type_text(
    text: str,
    target_bbox: tuple[int, int, int, int] | list[int] | None = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Type string via uinput virtual keyboard.

    If verify is True and target_bbox is provided, captures the screen before
    and after typing to confirm that text rendered into the input area.
    """
    before_bgr = None
    if verify and target_bbox:
        try:
            import cv2
            import numpy as np
            raw_before = capture_screen()
            before_bgr = cv2.imdecode(np.frombuffer(raw_before, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            before_bgr = None

    cmd = ["sudo", "-n", "python3", _VTYPE, "type", text] if os.path.exists(_VTYPE) else ["kbm", "type", text]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"typing failed ({exc.returncode}): {exc.stderr}") from exc

    result: dict[str, Any] = {"text": text, "status": "typed"}
    if before_bgr is not None and target_bbox is not None:
        try:
            import cv2
            import numpy as np
            time.sleep(0.1)
            raw_after = capture_screen()
            after_bgr = cv2.imdecode(np.frombuffer(raw_after, dtype=np.uint8), cv2.IMREAD_COLOR)
            diff_info = verify_text_input(before_bgr, after_bgr, input_bbox=target_bbox)
            result["input_verified"] = diff_info["verified"]
            result["change_details"] = diff_info["details"]
        except Exception as exc:
            result["input_verified"] = None
            result["verify_error"] = str(exc)

    return result


def press_key(combo: str) -> None:
    """Send key combination (e.g. 'ctrl+l', 'enter', 'esc')."""
    cmd = ["sudo", "-n", "python3", _VTYPE, "key", combo] if os.path.exists(_VTYPE) else ["kbm", "key", combo]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"key combo failed ({exc.returncode}): {exc.stderr}") from exc
