import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def humanize(name: str) -> str:
    base = Path(name).stem
    return base.replace('_', ' ').strip().capitalize()


def load_and_scale(img_path: Path, target_height_px: int) -> Image.Image:
    img = Image.open(img_path).convert("RGBA")
    if img.height == 0:
        return img
    scale = target_height_px / img.height
    new_w = max(1, int(round(img.width * scale)))
    return img.resize((new_w, target_height_px), Image.LANCZOS)


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def render_table(
    images_dir: Path,
    out_path: Path,
    target_row_height_px: int = 64,
    left_col_min_width_px: int = 260,
    margin_px: int = 24,
    gap_px: int = 16,
    row_gap_px: int = 8,
    font_path: str = "",
    font_size: int = 18,
    title: str = "",
    transparent: bool = False,
    title_font_size: int | None = None,
) -> None:
    # Only include per-formula PNGs; ignore previously rendered table images
    pngs = sorted(p for p in images_dir.glob("*.png") if "table" not in p.stem.lower())
    if not pngs:
        raise SystemExit(f"No PNGs found in {images_dir}")

    # Font
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Preload scaled images and measure labels
    scaled: List[Tuple[str, Image.Image]] = []
    labels: List[str] = []
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left_col_width = left_col_min_width_px
    for p in pngs:
        label = humanize(p.name)
        labels.append(label)
        w_text, h_text = measure_text(tmp_draw, label, font)
        left_col_width = max(left_col_width, w_text)
        img = load_and_scale(p, target_row_height_px)
        scaled.append((label, img))

    # Compute total size
    max_img_width = max(im.width for _, im in scaled)
    row_height = max(target_row_height_px, font_size + 8)
    total_width = margin_px + left_col_width + gap_px + max_img_width + margin_px
    title_h = 0
    if title:
        # Build a title font (larger than row font if requested)
        tf_size = title_font_size if title_font_size is not None else max(font_size + 6, font_size)
        if font_path:
            try:
                title_font = ImageFont.truetype(font_path, tf_size)
            except Exception:
                title_font = ImageFont.load_default()
        else:
            try:
                title_font = ImageFont.truetype("arial.ttf", tf_size)
            except Exception:
                title_font = ImageFont.load_default()
        title_w, title_h = measure_text(tmp_draw, title, title_font)
        title_h = max(title_h, tf_size + 4)
    total_height = (
        margin_px + (title_h + 8 if title else 0) +
        sum(row_height + row_gap_px for _ in scaled) - row_gap_px + margin_px
    )

    # Create canvas and draw
    bg_mode = "RGBA" if transparent else "RGB"
    bg_color = (255, 255, 255, 0) if transparent else "white"
    canvas = Image.new(bg_mode, (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Draw title
    y = margin_px
    if title:
        # Center the title horizontally
        x_title = (total_width - title_w) // 2
        draw.text((x_title, y), title, fill="black", font=title_font)
        y += title_h + 8
    for label, img in scaled:
        # Draw label
        draw.text((margin_px, y + (row_height - font_size) // 2), label, fill="black", font=font)
        # Paste image
        x_img = margin_px + left_col_width + gap_px
        y_img = y + (row_height - img.height) // 2
        canvas.paste(img, (x_img, y_img), img)
        # Row separator (optional light line)
        draw.line((margin_px, y + row_height, total_width - margin_px, y + row_height), fill=(230, 230, 230), width=1)
        y += row_height + row_gap_px

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote PNG table: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Render a PNG table of formula images and labels (no LaTeX)")
    ap.add_argument("--images-dir", default="figures/formulas", help="Directory containing formula PNGs")
    ap.add_argument("--out", default="figures/formulas/engineered_feature_formulas_table.png", help="Output PNG path")
    ap.add_argument("--row-height", type=int, default=64, help="Target row image height in pixels")
    ap.add_argument("--left-col-min", type=int, default=260, help="Minimum width of label column in pixels")
    ap.add_argument("--font", default="", help="Optional path to a .ttf font")
    ap.add_argument("--font-size", type=int, default=18, help="Font size for labels")
    ap.add_argument("--title", default="", help="Optional title text at top of image")
    ap.add_argument("--transparent", action="store_true", help="Make background transparent")
    args = ap.parse_args()

    render_table(
        images_dir=Path(args.images_dir),
        out_path=Path(args.out),
        target_row_height_px=args.row_height,
        left_col_min_width_px=args.left_col_min,
        font_path=args.font,
        font_size=args.font_size,
        title=args.title,
        transparent=args.transparent,
    )


if __name__ == "__main__":
    main()


