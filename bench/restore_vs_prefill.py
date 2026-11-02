#!/usr/bin/env python3
"""Restore versus recompute, at real KV sizes.

The question the tiering benchmark did not answer: when a prefix is missing from
L1, is it cheaper to fetch its KV from Valkey or to prefill it again?

Measures both sides at the same token count on the same machine:

  recompute : llama-server prefill of an N-token prompt, prompt_ms from timings
  restore   : Valkey GET of the KV bytes those N tokens would occupy, plus decode

KV geometry is read from the model, not assumed: layers x 2 x kv_heads x head_dim
x bytes_per_element.

Usage:
  python3 bench/restore_vs_prefill.py --url http://127.0.0.1:8097 --tokens 4096
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from resp import Valkey  # noqa: E402

# Qwen3-4B-Instruct-2507
LAYERS = 36
KV_HEADS = 8
HEAD_DIM = 128
F16 = 2


def prefill(url, tokens):
    prompt = "The quick brown fox jumps over the lazy dog. " * (tokens // 9 + 1)
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "cache_prompt": False,      # force a real prefill, not a prefix hit
    }).encode()
    req = urllib.request.Request(f"{url}/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read())
    wall = time.perf_counter() - t0
    timings = body.get("timings") or {}
    return {
        "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
        "prompt_ms": timings.get("prompt_ms"),
        "wall_ms": round(wall * 1000, 1),
        "prompt_per_second": timings.get("prompt_per_second"),
    }


def kv_blocks(tokens, block_tokens=16):
    """Split N tokens of KV into 16-token blocks, one blob per block per layer."""
    per_token = LAYERS * 2 * KV_HEADS * HEAD_DIM * F16
    n_blocks = -(-tokens // block_tokens)
    return n_blocks, per_token * block_tokens


def restore(port, tokens, block_tokens=16):
    n_blocks, block_bytes = kv_blocks(tokens, block_tokens)
    rng = np.random.default_rng(9)
    blob = rng.bytes(block_bytes)

    with Valkey(port=port) as vk:
        vk.cmd("FLUSHDB")
        t0 = time.perf_counter()
        for i in range(n_blocks):
            vk.set(f"kv:{i:06d}", blob)
        write_s = time.perf_counter() - t0

        # measure GET + deserialize only; reading is what a miss costs
        t1 = time.perf_counter()
        total = 0
        for i in range(n_blocks):
            got = vk.get(f"kv:{i:06d}")
            total += len(got)
        read_s = time.perf_counter() - t1

    return {
        "blocks": n_blocks,
        "bytes_per_block": block_bytes,
        "total_mib": round(total / (1024 * 1024), 1),
        "write_seconds": round(write_s, 3),
        "read_seconds": round(read_s, 3),
        "read_mib_per_s": round(total / read_s / (1024 * 1024), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--port", type=int, default=6390)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--out", default="results/restore_vs_prefill.json")
    args = ap.parse_args()

    pf = prefill(args.url, args.tokens)
    rs = restore(args.port, args.tokens)

    prefill_ms = pf.get("prompt_ms")
    restore_ms = rs["read_seconds"] * 1000
    verdict = None
    if prefill_ms and restore_ms:
        verdict = round(prefill_ms / restore_ms, 2)

    result = {
        "tokens": args.tokens,
        "kv_bytes_per_token": LAYERS * 2 * KV_HEADS * HEAD_DIM * F16,
        "prefill": pf,
        "restore": rs,
        "restore_ms": round(restore_ms, 1),
        "prefill_over_restore": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
