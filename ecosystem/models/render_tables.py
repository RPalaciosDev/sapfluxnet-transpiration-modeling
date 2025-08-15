import os
import argparse
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def render_table_image(df: pd.DataFrame, out_base: str, title: str = "", max_rows_per_page: int = 30, fontsize: int = 8) -> List[str]:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    paths: List[str] = []

    # Split into pages
    num_pages = max(1, (len(df) + max_rows_per_page - 1) // max_rows_per_page)
    for page in range(num_pages):
        start = page * max_rows_per_page
        end = min(len(df), start + max_rows_per_page)
        chunk = df.iloc[start:end]

        # Estimate figure size: width grows with columns, height with rows
        ncols = len(chunk.columns)
        nrows = len(chunk) + (1 if title else 0)
        fig_w = max(6, min(18, 1.4 * ncols))
        fig_h = max(2, min(24, 0.4 * (len(chunk) + 2)))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')

        # Title
        if title:
            ax.set_title(title + (f" (part {page+1})" if num_pages > 1 else ""), fontsize=fontsize+2, pad=12)

        # Matplotlib table
        tbl = ax.table(cellText=chunk.values,
                       colLabels=chunk.columns,
                       loc='center',
                       cellLoc='left',
                       colLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)

        # Column widths heuristic
        for i, col in enumerate(chunk.columns):
            width = 0.12 if len(str(col)) < 16 else 0.18
            try:
                tbl.auto_set_column_width([i])
            except Exception:
                pass
            tbl._cells[(0, i)].set_fontsize(fontsize + 1)
        for key, cell in tbl._cells.items():
            r, c = key
            if r == 0:
                cell.set_facecolor('#f0f0f0')
                cell.set_edgecolor('#cccccc')
                cell.set_linewidth(1.0)
            else:
                cell.set_edgecolor('#e0e0e0')
                cell.set_linewidth(0.6)

        fig.tight_layout()
        png_path = f"{out_base}_part{page+1}.png" if num_pages > 1 else f"{out_base}.png"
        pdf_path = f"{out_base}_part{page+1}.pdf" if num_pages > 1 else f"{out_base}.pdf"
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        paths.append(png_path)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Render CSV tables to PNG/PDF images for slides/papers")
    ap.add_argument("--input", required=True, help="Path to CSV file")
    ap.add_argument("--out", required=True, help="Output path base without extension (e.g., figures/tables/engineered_features)")
    ap.add_argument("--title", default="", help="Optional table title")
    ap.add_argument("--max-rows", type=int, default=30, help="Max rows per page before splitting")
    ap.add_argument("--fontsize", type=int, default=8, help="Base font size")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    render_table_image(df, args.out, title=args.title, max_rows_per_page=args.max_rows, fontsize=args.fontsize)
    print(f"Rendered table images for {args.input} to base {args.out}")


if __name__ == "__main__":
    main()


