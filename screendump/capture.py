"""Screen capture: grab the current display as PNG bytes.

Shells out to whatever screenshot tool the system has (grim, scrot, maim,
ImageMagick import, gnome-screenshot, spectacle, xdg-desktop-portal...).
Set ``SCREENDUMP_CAPTURE_CMD`` to a command template with ``{out}`` as the
output path to force a specific tool.
"""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# name -> command builder, each writing a PNG to the given path
_CANDIDATES: list[tuple[str, list[str]]] = [
    ("gnome-screenshot", ["gnome-screenshot", "-f", "{out}"]),
    ("grim", ["grim", "{out}"]),
    ("spectacle", ["spectacle", "-b", "-n", "-o", "{out}"]),
    ("scrot", ["scrot", "{out}"]),
    ("maim", ["maim", "{out}"]),
    ("import", ["import", "-window", "root", "{out}"]),
    ("gnome-shell portal", ["gdbus"]),  # handled by _portal_capture below
]


class CaptureError(RuntimeError):
    pass


def _available_commands() -> list[tuple[str, list[str]]]:
    env = os.environ.get("SCREENDUMP_CAPTURE_CMD")
    commands = []
    if env:
        commands.append((env, shlex.split(env)))
    for name, cmd in _CANDIDATES:
        if shutil.which(cmd[0]):
            commands.append((name, cmd))
    # On GNOME the portal is the only reliable path; try it first.
    if os.environ.get("XDG_CURRENT_DESKTOP", "").upper() == "GNOME":
        commands = [c for c in commands if c[0] == "gnome-shell portal"] + [
            c for c in commands if c[0] != "gnome-shell portal"
        ]
    return commands


def _run(cmd: list[str], timeout: float = 8) -> bool:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _portal_capture(out: str, deadline: float) -> bool:
    """GNOME: request a screenshot through xdg-desktop-portal.

    gnome-shell captures the screen and saves it as ``~/Pictures/
    Screenshot-N.png`` (auto-incremented). For callers without stored
    screenshot permission the portal never delivers its response signal,
    so watch for the new file instead.

    The shell writes the file after a variable delay (1-9s) and rejects
    a new request while a previous one is still in flight ("ongoing
    operation"), so on no-show we settle briefly and retry.
    """
    def newest_pic() -> tuple[float, str] | None:
        best = None
        for path in glob.glob(str(Path.home() / "Pictures" / "Screenshot-*.png")):
            key = (Path(path).stat().st_mtime, path)
            if best is None or key > best:
                best = key
        return best

    before = newest_pic()
    while time.monotonic() < deadline:
        try:
            subprocess.run(
                ["gdbus", "call", "--session", "--dest", "org.freedesktop.portal.Desktop",
                 "--object-path", "/org/freedesktop/portal/desktop",
                 "--method", "org.freedesktop.portal.Screenshot.Screenshot",
                 "", "{'handle_token': <'screendump'>, 'interactive': <false>}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

        wait_until = min(time.monotonic() + 8, deadline)
        prev_size = -1
        while time.monotonic() < wait_until:
            now = newest_pic()
            if now is not None and now != before:
                size = Path(now[1]).stat().st_size
                # the shell creates the file empty and fills it in; only
                # accept it once its size has settled to a nonzero value
                if size > 0 and size == prev_size:
                    try:
                        shutil.copy2(now[1], out)
                    except OSError:
                        return False
                    return True
                prev_size = size
            time.sleep(0.3)
        time.sleep(5)  # let any in-flight capture finish before retrying
    return False


def capture_screen() -> bytes:
    """Capture the whole screen and return PNG bytes (first tool that works)."""
    commands = _available_commands()
    if not commands:
        raise CaptureError(
            "no screenshot tool found; install one of: grim, scrot, maim, "
            "import, gnome-screenshot, spectacle"
        )
    with tempfile.TemporaryDirectory(prefix="screendump-") as tmp:
        out = str(Path(tmp) / "shot.png")
        for name, cmd in commands:
            if name == "gnome-shell portal":
                ok = _portal_capture(out, time.monotonic() + 30)
            else:
                resolved = [part.replace("{out}", out) for part in cmd]
                ok = _run(resolved)
            if ok and Path(out).exists() and Path(out).stat().st_size > 0:
                return Path(out).read_bytes()
    raise CaptureError(
        "all screenshot tools failed; make sure the desktop session is "
        "unlocked and active, then run with an image path instead"
    )
