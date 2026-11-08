#!/usr/bin/env python3
"""Render results/restore_sweep.json as one SVG chart.

No chart library: the output is deterministic and the file is committed, so a
reviewer can diff it. `--check` fails if the committed SVG is out of date with
the committed results.

Usage:
  python3 bench/chart.py                    # write docs/restore-vs-prefill.svg
  python3 bench/chart.py --check            # verify the committed chart
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "results" / "restore_sweep.json"
DEFAULT_OUT = ROOT / "docs" / "restore-vs-prefill.svg"

W, H = 920, 570
ML, MR, MT, MB = 92, 40, 96, 110
PW, PH = W - ML - MR, H - MT - MB

INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
PREFILL = "#94a3b8"
RESTORE = "#0284c7"
WIN = "#dcfce7"
BREAK = "#16a34a"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def x_pos(token, lo, hi):
    a, b = math.log2(lo), math.log2(hi)
    return ML + (math.log2(token) - a) / (b - a) * PW


def y_pos(ms, top):
    return MT + PH - (ms / top) * PH


def line(points, color, width, dash=None, opacity=1.0):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} opacity="{opacity}"/>')


def marker(x, y, color):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#ffffff" '
            f'stroke="{color}" stroke-width="2.5"/>')


def label(x, y, text, size=13, color=INK, anchor="middle", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="ui-sans-serif,system-ui,'
            f'-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{esc(text)}</text>')


def render(rows):
    tokens = [r[0] for r in rows]
    prefill = [r[1] for r in rows]
    restore = [r[2] for r in rows]
    lo, hi = min(tokens), max(tokens)
    top = max(max(prefill), max(restore)) * 1.12
    top = math.ceil(top / 100.0) * 100.0

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Restore from Valkey versus recomputing prefill, by prefix length">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]

    # Title and subtitle.
    out.append(label(ML, 42, "KV offload only pays above ~4k prefix tokens",
                     size=21, anchor="start", weight="600"))
    out.append(label(ML, 66, "Restore from Valkey versus recomputing the prefix. "
                             "Qwen3-4B, f16 KV (144 KiB/token), Valkey on localhost.",
                     size=13, color=MUTED, anchor="start"))

    # The region where a restore beats a prefill.
    cross = next((r[0] for r in rows if r[3] is not None and r[3] >= 1.0), None)
    if cross is not None:
        x0 = x_pos(cross, lo, hi)
        out.append(f'<rect x="{x0:.1f}" y="{MT}" width="{ML + PW - x0:.1f}" '
                   f'height="{PH}" fill="{WIN}" opacity="0.7"/>')

    # Y grid and ticks.
    step = 250
    for ms in range(0, int(top) + 1, step):
        y = y_pos(ms, top)
        out.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(label(ML - 12, y + 4, f"{ms} ms", size=12, color=MUTED, anchor="end"))

    # X ticks: powers of two inside the range, plus the largest data point.
    ticks = [t for t in (512, 1024, 2048, 4096, 8192) if lo <= t <= hi]
    if hi not in ticks:
        ticks.append(hi)
    for t in ticks:
        x = x_pos(t, lo, hi)
        out.append(f'<line x1="{x:.1f}" y1="{MT + PH}" x2="{x:.1f}" y2="{MT + PH + 6}" '
                   f'stroke="{MUTED}" stroke-width="1"/>')
        out.append(label(x, MT + PH + 26, f"{t:,}", size=12, color=MUTED))
    out.append(label(ML + PW / 2, MT + PH + 50, "prefix tokens (log scale)",
                     size=13, color=INK))

    # Axes.
    out.append(f'<line x1="{ML}" y1="{MT + PH}" x2="{ML + PW}" y2="{MT + PH}" '
               f'stroke="{MUTED}" stroke-width="1.2"/>')
    out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + PH}" '
               f'stroke="{MUTED}" stroke-width="1.2"/>')

    # Break-even marker.
    if cross is not None:
        x = x_pos(cross, lo, hi)
        out.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT + PH}" '
                   f'stroke="{BREAK}" stroke-width="1.6" stroke-dasharray="5 5"/>')
        out.append(label(x + 10, MT + 20, f"break-even ≈ {cross:,} tokens",
                         size=12, color=BREAK, anchor="start", weight="600"))
        out.append(label(ML + PW - 12, MT + PH - 14, "restore wins", size=12,
                         color=BREAK, anchor="end", weight="600"))

    # Data.
    pf_points = [(x_pos(t, lo, hi), y_pos(v, top)) for t, v in zip(tokens, prefill)]
    rs_points = [(x_pos(t, lo, hi), y_pos(v, top)) for t, v in zip(tokens, restore)]
    out.append(line(pf_points, PREFILL, 2.2, dash="7 5"))
    out.append(line(rs_points, RESTORE, 2.8))
    for p in pf_points:
        out.append(marker(*p, PREFILL))
    for p in rs_points:
        out.append(marker(*p, RESTORE))

    # Per-point ratio, the number the write-up quotes.
    for i, (row, p) in enumerate(zip(rows, rs_points)):
        if row[3] is None:
            continue
        # Keep the end labels inside the plot; keep the break-even label off
        # the dashed marker line.
        if i == 0:
            lx, anchor = p[0] + 9, "start"
        elif i == len(rows) - 1 or row[0] == cross:
            lx, anchor = p[0] - 9, "end"
        else:
            lx, anchor = p[0], "middle"
        out.append(label(lx, p[1] - 14, f"{row[3]:.2f}x", size=11,
                         color=RESTORE, anchor=anchor, weight="600"))

    # Legend.
    lx, ly = ML, H - 34
    out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="{PREFILL}" '
               f'stroke-width="2.2" stroke-dasharray="7 5"/>')
    out.append(label(lx + 34, ly + 4, "prefill (recompute)", size=12, color=INK, anchor="start"))
    out.append(f'<line x1="{lx + 190}" y1="{ly}" x2="{lx + 216}" y2="{ly}" '
               f'stroke="{RESTORE}" stroke-width="2.8"/>')
    out.append(label(lx + 224, ly + 4, "restore from Valkey", size=12, color=INK, anchor="start"))
    out.append(label(W - MR, ly + 4, "source: results/restore_sweep.json",
                     size=11, color=MUTED, anchor="end"))

    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed chart differs from a fresh render")
    args = ap.parse_args()

    rows = json.loads(Path(args.data).read_text())
    svg = render(rows)

    if args.check:
        current = Path(args.out).read_text() if Path(args.out).exists() else ""
        if current != svg:
            print(f"{args.out} is stale: run `make chart`", file=sys.stderr)
            return 1
        print(f"{args.out} matches {args.data}")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
