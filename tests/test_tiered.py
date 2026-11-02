"""Tests: spilled blocks must come back byte-identical, and tiering must admit
more work than a fixed pool on its own.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resp import Valkey  # noqa: E402
from l2 import L2BlockStore, block_key, decode, encode  # noqa: E402
from tiered import TieredKVCache  # noqa: E402

FAILS = 0


def ok(cond, label):
    global FAILS
    if cond:
        print(f"ok   {label}")
    else:
        FAILS += 1
        print(f"FAIL {label}")


def test_codec():
    rng = np.random.default_rng(1)
    k, v = rng.normal(size=(16, 8)), rng.normal(size=(16, 8))
    blob = encode(k, v)
    k2, v2 = decode(blob, 8)
    ok(np.array_equal(k, k2) and np.array_equal(v, v2), "encode/decode is lossless")
    # random floats are incompressible; low-entropy KV blocks are not
    low_k = np.repeat(np.arange(16, dtype=np.float64).reshape(16, 1), 8, axis=1)
    low_blob = encode(low_k, low_k)
    ok(len(low_blob) < low_k.nbytes * 2,
       f"low-entropy blocks compress ({len(low_blob)} < {low_k.nbytes * 2})")
    ok(len(blob) < k.nbytes + v.nbytes * 1.2,
       "random blocks stay within framing overhead")
    ok(block_key(k, v) == block_key(k.copy(), v.copy()), "keys are content-addressed")


def test_valkey_roundtrip():
    with Valkey(port=6390) as vk:
        ok(vk.ping(), "valkey responds to PING")
        store = L2BlockStore(vk, dim=4, namespace="test")
        rng = np.random.default_rng(2)
        k, v = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
        key = store.put(k, v)
        ok(vk.exists(key), "block written to valkey")
        got = store.get(key)
        ok(got is not None and np.array_equal(got[0], k) and np.array_equal(got[1], v),
           "block restored byte-identical")
        store.drop(key)
        ok(not vk.exists(key), "block removed")


def test_spill_and_restore():
    with Valkey(port=6390) as vk:
        store = L2BlockStore(vk, dim=8, namespace="test")
        cache = TieredKVCache(num_blocks=4, block_size=4, dim=8, l2=store)
        rng = np.random.default_rng(3)

        for s in range(3):
            cache.add_sequence(s)
            k, v = rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
            cache.append(s, k, v)

        before = cache.gather(0)
        # force a spill by adding a fourth sequence that needs blocks
        cache.add_sequence(3)
        k, v = rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
        cache.append(3, k, v)
        ok(cache.stats.spills >= 1, "a sequence spilled under block pressure")

        after = cache.gather(0)
        same = np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
        ok(same, "restored sequence is identical to the pre-spill one")
        ok(cache.stats.hits > 0, "restore produced L2 hits")


def test_capacity_gain():
    with Valkey(port=6390) as vk:
        store = L2BlockStore(vk, dim=8, namespace="capacity")
        cache = TieredKVCache(num_blocks=8, block_size=4, dim=8, l2=store)
        rng = np.random.default_rng(4)
        served = 0
        for s in range(20):
            cache.add_sequence(s)
            k, v = rng.normal(size=(8, 8)), rng.normal(size=(8, 8))
            try:
                cache.append(s, k, v)
                served += 1
            except Exception:
                break
        ok(served > cache.cache.alloc.num_blocks // 2,
           f"tiering served {served} sequences with only {cache.cache.alloc.num_blocks} blocks")


if __name__ == "__main__":
    test_codec()
    test_valkey_roundtrip()
    test_spill_and_restore()
    test_capacity_gain()
    print(f"\n{'FAILED' if FAILS else 'all passed'} ({FAILS} failure(s))")
    raise SystemExit(1 if FAILS else 0)
