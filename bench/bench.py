"""Capacity and cost of putting Valkey behind a fixed KV pool.

Baseline: a fixed block pool that refuses new sequences when it is full.
Tiered:   the same pool, but the least recently used sequence spills to Valkey
          and is restored on its next read.

Run: python3 bench/bench.py [--out results/valkey_l2.json]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "paged-attention-lab" / "src"))

from l2 import L2BlockStore  # noqa: E402
from paged_kv import OutOfBlocks, PagedKVCache  # noqa: E402
from resp import Valkey  # noqa: E402
from tiered import TieredKVCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/valkey_l2.json")
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--sequences", type=int, default=200)
    ap.add_argument("--port", type=int, default=6390)
    args = ap.parse_args()

    rng = np.random.default_rng(21)
    lengths = rng.integers(32, 400, size=args.sequences).tolist()
    data = [(rng.normal(size=(n, args.dim)), rng.normal(size=(n, args.dim))) for n in lengths]

    # baseline: no L2, refuse when full
    base = PagedKVCache(args.blocks, args.block_size, args.dim)
    served_base = 0
    for i, (k, v) in enumerate(data):
        base.add_sequence(i)
        try:
            base.append(i, k, v)
            served_base += 1
        except OutOfBlocks:
            base.drop_sequence(i)
            break

    # tiered: same pool, Valkey behind it
    with Valkey(port=args.port) as vk:
        vk.cmd("FLUSHDB")
        store = L2BlockStore(vk, dim=args.dim, namespace="bench")
        tiered = TieredKVCache(args.blocks, args.block_size, args.dim, l2=store)
        served_tiered = 0
        t0 = time.perf_counter()
        for i, (k, v) in enumerate(data):
            tiered.add_sequence(i)
            tiered.append(i, k, v)
            served_tiered += 1
        elapsed = time.perf_counter() - t0

        # re-read the first 50 to force restores from L2
        re_read = min(50, served_tiered)
        t1 = time.perf_counter()
        for i in range(re_read):
            tiered.gather(i)
        restore_wall = time.perf_counter() - t1

        info = vk.info("memory")
        used_bytes = next((int(l.split(":")[1]) for l in info.splitlines()
                           if l.startswith("used_memory:")), None)

    result = {
        "l1_blocks": args.blocks,
        "block_size": args.block_size,
        "dim": args.dim,
        "sequences_offered": args.sequences,
        "tokens_total": int(sum(lengths)),
        "baseline_sequences_served": served_base,
        "tiered_sequences_served": served_tiered,
        "baseline_served_pct": round(100 * served_base / args.sequences, 1),
        "tiered_served_pct": round(100 * served_tiered / args.sequences, 1),
        "spills": tiered.stats.spills,
        "restores": tiered.stats.restores,
        "l2_hit_rate": round(tiered.stats.hits / (tiered.stats.hits + tiered.stats.misses), 4)
        if (tiered.stats.hits + tiered.stats.misses) else 0.0,
        "l2_bytes_written": store.stats.bytes_written,
        "l2_bytes_read": store.stats.bytes_read,
        "valkey_used_memory_bytes": used_bytes,
        "write_wall_seconds": round(elapsed, 2),
        "restore_wall_seconds": round(restore_wall, 2),
        "mean_restore_ms": round(1000 * restore_wall / re_read, 1) if re_read else None,
        "re_read_sequences": re_read,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
