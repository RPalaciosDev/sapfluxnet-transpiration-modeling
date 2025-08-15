import os
import argparse
from pathlib import Path
from typing import List


def humanize(name: str) -> str:
    base = Path(name).stem
    return base.replace('_', ' ').strip().capitalize()


def build_table_tex(png_paths: List[Path], image_height_mm: int = 18) -> str:
    header = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{array}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{caption}
\captionsetup{labelformat=empty}
\begin{document}
\begin{center}\Large Engineered Feature Formulas\end{center}
\vspace{0.5em}
\setlength{\extrarowheight}{2pt}
\begin{longtable}{@{}p{0.28\linewidth} p{0.68\linewidth}@{}}
\toprule
\textbf{Feature} & \textbf{Formula} \\
\midrule
\endfirsthead
\toprule
\textbf{Feature} & \textbf{Formula} \\
\midrule
\endhead
\bottomrule
\endfoot
"""

    rows: List[str] = []
    for p in png_paths:
        feature = humanize(p.name)
        # Use forward slashes in LaTeX paths
        rel_path = p.as_posix()
        row = f"{feature} & \\includegraphics[height={image_height_mm}mm]{{{rel_path}}} \\\\"
        rows.append(row)

    footer = r"""\end{longtable}
\end{document}
"""
    return header + "\n".join(rows) + "\n" + footer


def main():
    ap = argparse.ArgumentParser(description="Build a LaTeX table including all formula PNGs")
    ap.add_argument("--images-dir", default="figures/formulas", help="Directory containing formula PNGs")
    ap.add_argument("--out-tex", default="tables/engineered_feature_formulas_table.tex", help="Output LaTeX file path")
    ap.add_argument("--height-mm", type=int, default=18, help="Image height in mm (consistent across rows)")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pngs = sorted(images_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNGs found in {images_dir}")

    tex = build_table_tex(pngs, image_height_mm=args.height_mm)
    out_path.write_text(tex, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


