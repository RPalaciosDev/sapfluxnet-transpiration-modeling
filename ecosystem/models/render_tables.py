import os
import argparse
from typing import List
import textwrap

import pandas as pd
import matplotlib.pyplot as plt


def render_table_image(df: pd.DataFrame, out_base: str, title: str = "", max_rows_per_page: int = 30, fontsize: int = 8, transparent: bool = False) -> List[str]:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    paths: List[str] = []

    # Split into pages
    num_pages = max(1, (len(df) + max_rows_per_page - 1) // max_rows_per_page)
    for page in range(num_pages):
        start = page * max_rows_per_page
        end = min(len(df), start + max_rows_per_page)
        chunk = df.iloc[start:end]

        # Wrap long cell contents to reduce clipping in narrow columns
        def wrap_cell(value: object) -> str:
            s = "" if value is None else str(value)
            # Heuristic wrap width; adjust if many columns
            wrap_width = 40 if len(df.columns) <= 4 else 30 if len(df.columns) <= 6 else 24
            return "\n".join(textwrap.wrap(s, width=wrap_width)) if len(s) > wrap_width else s

        chunk_wrapped = chunk.astype(str).applymap(wrap_cell)

        # Estimate figure size: width grows with columns, height with rows
        ncols = len(chunk.columns)
        # Estimate width from content character lengths per column
        col_char_widths = []
        for col in chunk.columns:
            header_len = len(str(col))
            max_cell_len = chunk[col].astype(str).map(len).max() if len(chunk) > 0 else 0
            col_char_widths.append(max(header_len, max_cell_len))
        # inches per character heuristic and margins
        inches_per_char = 0.08 if ncols >= 6 else 0.10
        fig_w = sum(w * inches_per_char for w in col_char_widths) + 2.0
        fig_w = max(6, min(24, fig_w))
        # Height based on rows and presence of title
        base_rows = len(chunk) + (1 if title else 0)
        fig_h = 0.42 * (base_rows + 2)
        fig_h = max(2.5, min(30, fig_h))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')

        # Title
        if title:
            ax.set_title(title + (f" (part {page+1})" if num_pages > 1 else ""), fontsize=fontsize+2, pad=3)

        # Matplotlib table
        tbl = ax.table(cellText=chunk_wrapped.values,
                       colLabels=chunk.columns,
                       loc='upper center',
                       cellLoc='left',
                       colLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        # Slightly increase row heights to accommodate wrapped lines
        tbl.scale(1.0, 1.2)

        # Column widths heuristic
        for i, col in enumerate(chunk.columns):
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

        # Use tight bbox to avoid clipping
        png_path = f"{out_base}_part{page+1}.png" if num_pages > 1 else f"{out_base}.png"
        pdf_path = f"{out_base}_part{page+1}.pdf" if num_pages > 1 else f"{out_base}.pdf"
        fig.savefig(png_path, dpi=220, bbox_inches='tight', pad_inches=0.05, transparent=transparent)
        fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05, transparent=transparent)
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
    ap.add_argument("--transparent", action='store_true', help='Render with transparent background')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    render_table_image(df, args.out, title=args.title, max_rows_per_page=args.max_rows, fontsize=args.fontsize, transparent=args.transparent)
    print(f"Rendered table images for {args.input} to base {args.out}")


if __name__ == "__main__":
    main()


