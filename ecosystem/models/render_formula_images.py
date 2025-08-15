import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


ITEM_RE = re.compile(r"^\s*\\item\[(.*?)\]\s*(.*)$")
INLINE_MATH_RE = re.compile(r"\$(.*?)\$", re.DOTALL)
DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_")
    return slug.lower() or "formula"


def parse_items(tex_path: Path) -> List[Tuple[str, str]]:
    """Parse \item[label] blocks and collect their text bodies.

    Returns list of (label, body_text).
    """
    lines = tex_path.read_text(encoding="utf-8").splitlines()
    items: List[Tuple[str, List[str]]] = []
    current_label: str = ""
    current_buf: List[str] = []

    def flush():
        nonlocal current_label, current_buf
        if current_label:
            items.append((current_label, current_buf))
        current_label = ""
        current_buf = []

    for line in lines:
        m = ITEM_RE.match(line)
        if m:
            # Start of a new item
            flush()
            current_label = m.group(1).strip()
            rest = m.group(2)
            current_buf.append(rest)
        else:
            if current_label:
                current_buf.append(line)

        if "\\end{description}" in line:
            break

    flush()
    # Join buffers
    return [(label, "\n".join(buf)) for (label, buf) in items]


def extract_math(body_text: str) -> List[str]:
    """Extract math expressions from an item's body.

    Preference order: explicit display blocks \\[...\\], then inline $...$.
    Returns list of one or more math strings.
    """
    math_blocks: List[str] = []
    # Display first
    for dm in DISPLAY_MATH_RE.finditer(body_text):
        expr = dm.group(1).strip()
        if expr:
            math_blocks.append(expr)
    # Inline next
    for im in INLINE_MATH_RE.finditer(body_text):
        expr = im.group(1).strip()
        if expr:
            math_blocks.append(expr)
    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for expr in math_blocks:
        key = re.sub(r"\s+", " ", expr)
        if key not in seen:
            seen.add(key)
            unique.append(expr)
    return unique


def make_standalone_tex(label: str, math_blocks: List[str]) -> str:
    lines: List[str] = [
        r"\documentclass[preview,border=2pt]{standalone}",
        r"\usepackage{amsmath,amssymb}",
        r"\begin{document}",
    ]
    # Wrap each block in display math
    for i, expr in enumerate(math_blocks):
        lines.append(r"\[" + "\n" + expr + "\n" + r"\]")
        if i < len(math_blocks) - 1:
            lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def run(cmd: List[str], cwd: Path) -> None:
    """Run a command, printing stdout/stderr on failure."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        cmd_str = " ".join(cmd)
        print(f"\nCommand failed (exit {proc.returncode}): {cmd_str}")
        if proc.stdout:
            print("--- stdout ---")
            print(proc.stdout)
        if proc.stderr:
            print("--- stderr ---")
            print(proc.stderr)
        raise RuntimeError(f"Command failed: {cmd_str}")


def build_images(tex_file: Path, to_svg: bool, to_png: bool) -> None:
    # Build PDF first
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name], cwd=tex_file.parent)
    pdf_path = tex_file.with_suffix(".pdf")
    if to_svg:
        run(["dvisvgm", "--pdf", pdf_path.name, "-n", "-a", "-o", tex_file.with_suffix(".svg").name], cwd=tex_file.parent)
    if to_png:
        # Transparent background PNG at decent density
        run(["magick", "-density", "300", "-background", "none", pdf_path.name, tex_file.with_suffix(".png").name], cwd=tex_file.parent)


def export_formulas(
    tex_input: Path,
    out_dir: Path,
    build: bool = True,
    svg: bool = True,
    png: bool = True,
) -> Dict[str, List[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = parse_items(tex_input)

    used_names: Dict[str, int] = {}
    summary: Dict[str, List[str]] = {}

    for label, body in items:
        blocks = extract_math(body)
        if not blocks:
            continue
        base = slugify(label)
        idx = used_names.get(base, 0)
        used_names[base] = idx + 1
        if idx > 0:
            base = f"{base}_{idx+1}"

        tex_content = make_standalone_tex(label, blocks)
        tex_path = out_dir / f"{base}.tex"
        tex_path.write_text(tex_content, encoding="utf-8")

        if build:
            try:
                build_images(tex_path, to_svg=svg, to_png=png)
            except Exception as e:
                print(f"Build failed for '{label}' ({base}). See messages above. Skipping.")

        summary[base] = blocks

    return summary


def main():
    ap = argparse.ArgumentParser(description="Export engineered feature formulas to standalone images")
    ap.add_argument("--input-tex", default="tables/engineered_feature_formulas.tex", help="Path to LaTeX source with description items")
    ap.add_argument("--out-dir", default="figures/formulas", help="Output directory for per-formula images")
    ap.add_argument("--no-build", action="store_true", help="Only write .tex files; do not compile to images")
    ap.add_argument("--no-svg", action="store_true", help="Do not create SVGs")
    ap.add_argument("--no-png", action="store_true", help="Do not create PNGs")
    args = ap.parse_args()

    tex_input = Path(args.input_tex)
    out_dir = Path(args.out_dir)
    summary = export_formulas(
        tex_input=tex_input,
        out_dir=out_dir,
        build=not args.no_build,
        svg=not args.no_svg,
        png=not args.no_png,
    )

    print(f"Exported {len(summary)} formulas to {out_dir}")


if __name__ == "__main__":
    main()


