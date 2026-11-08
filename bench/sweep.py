#!/usr/bin/env python3
"""Sweep restore against prefill across prefix lengths.

`restore_vs_prefill.py` answers the question at one token count. This runs the
same measurement at several counts and records the crossover, which is the
result: below ~4k tokens a restore costs more than recomputing; above it the
restore competes with prefill because prefill grows faster than the bytes.

Two things make the numbers usable:

  - one warm-up prefill, discarded, because the first request on a Vulkan
    build pays for shader and graph setup (about 3x the steady-state time);
  - repeats with the median reported, because a single 900 MB read on a busy
    machine is not a measurement.

Writes results/restore_sweep.json as a list of
  [tokens, prefill_ms, restore_ms, prefill_over_restore]

Usage:
  python3 bench/sweep.py --url http://127.0.0.1:8097
  python3 bench/sweep.py --tokens 512,2048,4096,6144 --repeats 3
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from restore_vs_prefill import prefill, restore  # noqa: E402

DEFAULT_TOKENS = [512, 2048, 4096, 6144]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--port", type=int, default=6390)
    ap.add_argument("--tokens", default=",".join(str(t) for t in DEFAULT_TOKENS),
                    help="comma-separated prefix lengths")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/restore_sweep.json")
    args = ap.parse_args()

    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]

    warm = prefill(args.url, 128)
    print(f"warm-up: {warm.get('prompt_ms', 0):.0f} ms (discarded)\n", flush=True)

    rows = []
    for n in tokens:
        pfs, rss = [], []
        for _ in range(args.repeats):
            pf = prefill(args.url, n)
            rs = restore(args.port, n)
            pfs.append(pf["prompt_ms"])
            rss.append(rs["read_seconds"] * 1000)
        prefill_ms = round(statistics.median(pfs), 1)
        restore_ms = round(statistics.median(rss), 1)
        ratio = round(prefill_ms / restore_ms, 2)
        rows.append([n, prefill_ms, restore_ms, ratio])
        print(f"{n:>6} tokens  prefill {prefill_ms:8.1f} ms  "
              f"restore {restore_ms:8.1f} ms  ratio {ratio:.2f}  "
              f"(prefill {min(pfs):.0f}-{max(pfs):.0f}, "
              f"restore {min(rss):.0f}-{max(rss):.0f})", flush=True)
        time.sleep(0.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
