"""Tiered KV cache: L1 is a fixed block pool, L2 is Valkey.

When L1 runs out of blocks, the least recently used sequence is spilled to Valkey
and its blocks are freed, so the pool keeps serving instead of refusing. Reading a
spilled sequence restores it from L2. Content-addressed keys mean a restored
sequence can hit an entry written by a different sequence or a previous process.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIBLING = Path(__file__).resolve().parents[2] / "paged-attention-lab" / "src"
if SIBLING.exists():
    sys.path.insert(0, str(SIBLING))

from paged_kv import OutOfBlocks, PagedKVCache  # noqa: E402

from l2 import L2BlockStore  # noqa: E402


@dataclass
class TierStats:
    spills: int = 0
    restores: int = 0
    hits: int = 0
    misses: int = 0
    restore_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "spills": self.spills,
            "restores": self.restores,
            "hit_rate": round(self.hits / (self.hits + self.misses), 4)
            if (self.hits + self.misses) else 0.0,
            "restore_seconds": round(self.restore_seconds, 4),
        }


class TieredKVCache:
    def __init__(self, num_blocks: int, block_size: int, dim: int, l2: L2BlockStore):
        self.cache = PagedKVCache(num_blocks, block_size, dim)
        self.l2 = l2
        self.dim = dim
        self.block_size = block_size
        self.keys: dict[int, list[str]] = {}       # seq -> one key per logical block
        self.spilled: dict[int, tuple[list[str], int]] = {}
        self.last_used: dict[int, float] = {}
        self.stats = TierStats()

    # ---- lifecycle ----------------------------------------------------------
    def add_sequence(self, seq_id: int) -> None:
        self.cache.add_sequence(seq_id)
        self.keys[seq_id] = []
        self.last_used[seq_id] = time.time()

    def append(self, seq_id: int, k: np.ndarray, v: np.ndarray) -> None:
        if seq_id in self.spilled:
            self._restore(seq_id)
        self.last_used[seq_id] = time.time()
        before = self.cache.lengths[seq_id]
        rows = np.atleast_2d(k).shape[0]
        # reserve the whole append up front; a mid-append failure leaves the
        # sequence half written and cannot be retried cleanly
        end = before + rows
        needed_total = math.ceil(end / self.block_size)
        have = len(self.cache.tables[seq_id])
        if needed_total > have:
            self._ensure_free(needed_total - have, keep=seq_id)
        self.cache.append(seq_id, k, v)
        self._index_blocks(seq_id, before)

    def gather(self, seq_id: int) -> tuple[np.ndarray, np.ndarray]:
        if seq_id in self.spilled:
            self._restore(seq_id)
        self.last_used[seq_id] = time.time()
        return self.cache.gather(seq_id)

    # ---- L1 -> L2 -----------------------------------------------------------
    def _index_blocks(self, seq_id: int, from_pos: int) -> None:
        """Record a content key for every block that became complete or changed."""
        length = self.cache.lengths[seq_id]
        first = from_pos // self.block_size
        last = (length - 1) // self.block_size if length else -1
        for bi in range(first, last + 1):
            block = self.cache.tables[seq_id][bi]
            key = self.l2.put(self.cache.k[block], self.cache.v[block])
            while len(self.keys[seq_id]) <= bi:
                self.keys[seq_id].append("")
            self.keys[seq_id][bi] = key

    def _spill_lru(self, keep: int) -> bool:
        candidates = [s for s in self.cache.tables if s != keep and s not in self.spilled]
        if not candidates:
            return False
        victim = min(candidates, key=lambda s: self.last_used.get(s, 0.0))
        self._spill(victim)
        return True

    def _spill(self, seq_id: int) -> None:
        keys = list(self.keys.get(seq_id, []))
        length = self.cache.lengths[seq_id]
        self.cache.drop_sequence(seq_id)
        self.spilled[seq_id] = (keys, length)
        self.stats.spills += 1

    def _ensure_free(self, needed: int, keep: int) -> None:
        """Spill least-recently-used sequences until `needed` blocks are free."""
        guard = 0
        while self.cache.alloc.free_blocks < needed:
            if not self._spill_lru(keep=keep):
                raise OutOfBlocks(f"need {needed} blocks to restore {keep}, none spillable")
            guard += 1
            if guard > len(self.spilled) + 1024:
                raise OutOfBlocks("spill loop did not converge")

    def _restore(self, seq_id: int) -> None:
        keys, length = self.spilled.pop(seq_id)
        t0 = time.perf_counter()
        self._ensure_free(math.ceil(length / self.block_size), keep=seq_id)
        self.cache.add_sequence(seq_id)
        restored = 0
        for key in keys:
            pair = self.l2.get(key)
            if pair is None:
                self.stats.misses += 1
                continue
            self.stats.hits += 1
            k_blk, v_blk = pair
            n = min(self.block_size, length - restored)
            self.cache.append(seq_id, k_blk[:n], v_blk[:n])
            restored += n
        self.stats.restore_seconds += time.perf_counter() - t0
        self.stats.restores += 1
        self.keys[seq_id] = keys
        self.last_used[seq_id] = time.time()

    @property
    def resident_sequences(self) -> int:
        return len(self.cache.tables)
