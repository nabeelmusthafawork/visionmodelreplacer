"""Generate a synthetic screenshot that mimics a desktop browser window.

Used to test the screendump pipeline without a real screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 680, 420


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(out: str = "sample.png") -> None:
    img = Image.new("RGB", (W, H), "#f0f0f0")
    d = ImageDraw.Draw(img)

    # window frame
    d.rectangle([20, 20, W - 20, H - 20], outline="#333333", width=2)
    d.rectangle([20, 20, W - 20, 56], fill="#e8e8e8", outline="#333333", width=2)

    # title
    f_title = _font(18)
    d.text((38, 30), "Firefox", font=f_title, fill="#111111")

    # toolbar: back / forward / address bar
    y = 74
    f_tool = _font(15)
    d.rectangle([38, y - 4, 62, y + 22], outline="#333333", width=2, fill="#ffffff")
    d.text((45, y + 2), "<-", font=f_tool, fill="#111111")
    d.rectangle([70, y - 4, 94, y + 22], outline="#333333", width=2, fill="#ffffff")
    d.text((77, y + 2), "->", font=f_tool, fill="#111111")
    d.rectangle([104, y - 4, 420, y + 22], outline="#333333", width=2, fill="#ffffff")
    d.text((114, y + 2), "https://example.com", font=f_tool, fill="#555555")

    # divider
    d.line([20, 110, W - 20, 110], fill="#999999", width=1)

    # content: heading, search box, buttons
    f_head = _font(20)
    d.text((260, 140), "Example Website", font=f_head, fill="#222222")

    d.rectangle([250, 190, 430, 232], outline="#333333", width=2, fill="#ffffff")
    d.text((262, 200), "Search", font=f_tool, fill="#555555")

    d.rectangle([250, 260, 330, 294], outline="#333333", width=2, fill="#4a6fa5")
    d.text((263, 268), "Login", font=f_tool, fill="#ffffff")
    d.rectangle([350, 260, 438, 294], outline="#333333", width=2, fill="#ffffff")
    d.text((361, 268), "Settings", font=f_tool, fill="#222222")

    # footer text
    d.text((250, 340), "All rights reserved 2026", font=f_tool, fill="#666666")

    img.save(out)
    print(f"wrote {out} ({W}x{H})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "sample.png"))