import os
import re
import sys
import argparse
import urllib.parse
import urllib.request
from pathlib import Path


DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


def extract_expr(tex_path: Path) -> str:
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    m = DISPLAY_RE.search(text)
    if not m:
        return ""
    expr = m.group(1)
    # Flatten whitespace
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def download_codecogs_png(expr: str, out_path: Path, dpi: int = 300, bg: str = "white") -> bool:
    if not expr:
        return False
    prefix = f"\\dpi{{{dpi}}}\\bg_{bg} "
    query = urllib.parse.quote(prefix + expr, safe="")
    url = f"https://latex.codecogs.com/png.image?{query}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"Failed to fetch PNG for {out_path.name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Fetch PNGs for per-formula .tex files via online renderer")
    ap.add_argument("--tex-dir", default="figures/formulas", help="Directory containing per-formula .tex files")
    ap.add_argument("--bg", choices=["white", "transparent"], default="white", help="Background color")
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI")
    args = ap.parse_args()

    tex_dir = Path(args.tex_dir)
    if not tex_dir.is_dir():
        print(f"No such directory: {tex_dir}")
        sys.exit(1)

    tex_files = sorted(tex_dir.glob("*.tex"))
    if not tex_files:
        print(f"No .tex files in {tex_dir}")
        sys.exit(1)

    ok, fail = 0, 0
    for tex_path in tex_files:
        expr = extract_expr(tex_path)
        png_path = tex_path.with_suffix(".png")
        if not expr:
            print(f"No display math found in {tex_path.name}; skipping")
            fail += 1
            continue
        if download_codecogs_png(expr, png_path, dpi=args.dpi, bg=args.bg):
            print(f"Saved: {png_path}")
            ok += 1
        else:
            fail += 1

    print(f"Done. Saved {ok}, failed {fail}")


if __name__ == "__main__":
    main()


