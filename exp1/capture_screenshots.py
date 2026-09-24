"""Run the verification script and save each command's real output as PNG."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "assets"
CAPTURES = {
    "pytorch": "01_pytorch_verification.png",
    "opencv": "02_opencv_verification.png",
    "sklearn": "03_sklearn_verification.png",
    "all": "04_all_packages_verification.png",
}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    fonts_dir = Path(r"C:\Windows\Fonts")
    for font_name in ("CascadiaMono.ttf", "consola.ttf", "cour.ttf"):
        font_path = fonts_dir / font_name
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def render_output(command: str, output: str, destination: Path) -> None:
    font = load_font(20)
    small_font = load_font(16)
    lines = [f"PS> {command}", *output.rstrip().splitlines()]
    line_height = 32
    width = 1280
    height = max(230, 110 + line_height * len(lines))
    image = Image.new("RGB", (width, height), "#0b1220")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, 52), fill="#172033")
    for x, color in ((24, "#fb7185"), (48, "#fbbf24"), (72, "#34d399")):
        draw.ellipse((x, 19, x + 12, 31), fill=color)
    draw.text((104, 16), "Python verification output", font=small_font, fill="#cbd5e1")

    y = 78
    for index, line in enumerate(lines):
        color = "#93c5fd" if index == 0 else "#e2e8f0"
        if "PASS" in line or "available: True" in line:
            color = "#86efac"
        draw.text((36, y), line, font=font, fill=color)
        y += line_height

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for check, filename in CAPTURES.items():
        command = ["python", "exp1/verify_environment.py", "--check", check]
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr or f"{check} check failed")
        render_output(
            " ".join(command),
            result.stdout,
            OUTPUT_DIR / filename,
        )
        print(f"Saved {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
