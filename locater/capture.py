"""Screen capture for locater (reuses the screendump capture conventions).

Returns PNG bytes of the whole display. Honors ``SCREENDUMP_CAPTURE_CMD``;
otherwise tries grim (wayland), scrot/maim (x11), ImageMagick import, then
the gnome-screenshot fallback.

Tools that only support writing to a file (import, gnome-screenshot) go
through a temp file; stdout-capable tools are read directly.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

# name -> command template writing a PNG to {out}
_FILE_TOOLS = {
    "grim": ["grim", "{out}"],
    "scrot": ["scrot", "{out}"],
    "maim": ["maim", "{out}"],
    "import": ["import", "-window", "root", "{out}"],
    "gnome-screenshot": ["gnome-screenshot", "-f", "{out}"],
}


def _run(cmd: list[str], timeout: float = 15) -> bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed ({proc.returncode}): "
            + proc.stderr.decode("utf-8", "replace")[:300]
        )
    if not proc.stdout:
        raise RuntimeError(f"{cmd[0]} produced empty output")
    return proc.stdout


def _capture_file(cmd: list[str]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        proc = subprocess.run(
            [part.replace("{out}", path) for part in cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{cmd[0]} failed ({proc.returncode}): "
                + proc.stderr.decode("utf-8", "replace")[:300]
            )
        data = Path(path).read_bytes()
        if not data:
            raise RuntimeError(f"{cmd[0]} produced empty output")
        return data
    finally:
        Path(path).unlink(missing_ok=True)


def capture_screen() -> bytes:
    """Capture the whole display as PNG bytes.

    Raises RuntimeError if no capture tool works.
    """
    cmd = os.environ.get("SCREENDUMP_CAPTURE_CMD")
    if cmd:
        try:
            return _capture_file(shlex.split(cmd))
        except RuntimeError as exc:
            raise RuntimeError(f"SCREENDUMP_CAPTURE_CMD failed: {exc}")
    for name, template in _FILE_TOOLS.items():
        if not shutil.which(name):
            continue
        try:
            return _capture_file(template)
        except (RuntimeError, subprocess.TimeoutExpired):
            continue
    try:
        from screendump import capture as _sd_capture

        return _sd_capture.capture_screen()
    except Exception:
        pass
    raise RuntimeError(
        "no capture tool found; install grim/scrot/maim/import or set SCREENDUMP_CAPTURE_CMD"
    )
