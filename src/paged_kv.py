"""Paged attention: KV cache in fixed blocks with per-sequence block tables.

The point of paging is that a sequence does not reserve a contiguous slab for
its maximum possible length. KV is allocated a block at a time and the block
table records where each logical position lives. Attention gathers K and V
through the table, so a sequence's blocks can be scattered across the cache and
shared with other sequences.

This module is the reference implementation plus the baseline it is compared
against. It is deliberately small: one head, one layer, float64, so that a
parity failure is a real bug rather than float noise.
"""
from __future__ import annotations

import math

import numpy as np


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    """Fixed pool of physical blocks, handed out in block_size slots."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))
        self._free.reverse()

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocks(f"requested {n} blocks, {len(self._free)} free")
        return [self._free.pop() for _ in range(n)]

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if b in self._free:
                raise ValueError(f"block {b} freed twice")
            self._free.append(b)


class PagedKVCache:
    """KV cache addressed by (block table, block offset) instead of raw index."""

    def __init__(self, num_blocks: int, block_size: int, dim: int):
        self.block_size = block_size
        self.dim = dim
        self.alloc = BlockAllocator(num_blocks, block_size)
        self.k = np.zeros((num_blocks, block_size, dim), dtype=np.float64)
        self.v = np.zeros((num_blocks, block_size, dim), dtype=np.float64)
        self.tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}
        # refcount so a shared prefix block is not freed until every owner drops it
        self.refs: dict[int, int] = {}

    # ---- sequence lifecycle -------------------------------------------------
    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} exists")
        self.tables[seq_id] = []
        self.lengths[seq_id] = 0

    def drop_sequence(self, seq_id: int) -> None:
        for b in self.tables.pop(seq_id, []):
            self._release(b)
        self.lengths.pop(seq_id, None)

    def fork(self, src: int, dst: int) -> None:
        """Share every block of src with dst (copy-on-write on the next append)."""
        self.add_sequence(dst)
        for b in self.tables[src]:
            self.refs[b] += 1
        self.tables[dst] = list(self.tables[src])
        self.lengths[dst] = self.lengths[src]

    # ---- blocks -------------------------------------------------------------
    def _acquire(self, block: int) -> None:
        self.refs[block] = self.refs.get(block, 0) + 1

    def _release(self, block: int) -> None:
        self.refs[block] -= 1
        if self.refs[block] == 0:
            self.alloc.free([block])

    def _private_block(self, seq_id: int, block: int) -> int:
        """Copy a shared block before writing to it."""
        if self.refs[block] == 1:
            return block
        clone = self.alloc.allocate(1)[0]
        self.k[clone] = self.k[block]
        self.v[clone] = self.v[block]
        self._acquire(clone)
        self._release(block)
        return clone

    # ---- writes -------------------------------------------------------------
    def append(self, seq_id: int, k_new: np.ndarray, v_new: np.ndarray) -> None:
        k_new = np.atleast_2d(np.asarray(k_new, dtype=np.float64))
        v_new = np.atleast_2d(np.asarray(v_new, dtype=np.float64))
        if k_new.shape[0] != v_new.shape[0]:
            raise ValueError("k and v must have the same number of tokens")
        if k_new.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {k_new.shape[1]}")

        for i in range(k_new.shape[0]):
            pos = self.lengths[seq_id]
            offset = pos % self.block_size
            if offset == 0:
                self.tables[seq_id].append(self.alloc.allocate(1)[0])
                self._acquire(self.tables[seq_id][-1])
            idx = len(self.tables[seq_id]) - 1
            block = self._private_block(seq_id, self.tables[seq_id][idx])
            self.tables[seq_id][idx] = block
            self.k[block, offset] = k_new[i]
            self.v[block, offset] = v_new[i]
            self.lengths[seq_id] += 1

    # ---- reads --------------------------------------------------------------
    def gather(self, seq_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Materialise the sequence's K and V in logical order."""
        n = self.lengths[seq_id]
        out_k = np.empty((n, self.dim), dtype=np.float64)
        out_v = np.empty((n, self.dim), dtype=np.float64)
        for pos in range(n):
            block = self.tables[seq_id][pos // self.block_size]
            offset = pos % self.block_size
            out_k[pos] = self.k[block, offset]
            out_v[pos] = self.v[block, offset]
        return out_k, out_v

    def reserved_slots_contiguous(self, max_len: int) -> int:
        """What a contiguous cache would have reserved for these sequences."""
        return len(self.tables) * max_len


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True) -> np.ndarray:
    """Single-head scaled dot-product attention over already-gathered K and V."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.T) * scale
    if causal:
        n_q, n_k = scores.shape
        mask = np.triu(np.ones((n_q, n_k), dtype=bool), k=n_k - n_q + 1)
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def paged_attention(cache: PagedKVCache, seq_id: int, q: np.ndarray) -> np.ndarray:
    """Attention that reads K/V through the block table, never materialising a slab."""
    n = cache.lengths[seq_id]
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = np.zeros((q.shape[0], cache.dim), dtype=np.float64)
    for i in range(q.shape[0]):
        scores = np.empty(n, dtype=np.float64)
        for pos in range(n):
            block = cache.tables[seq_id][pos // cache.block_size]
            offset = pos % cache.block_size
            scores[pos] = float(q[i] @ cache.k[block, offset]) * scale
        # causal: query i may attend up to position n - q_count + i
        limit = n - q.shape[0] + i + 1
        scores[limit:] = -np.inf
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        acc = np.zeros(cache.dim, dtype=np.float64)
        for pos in range(n):
            if w[pos] == 0.0:
                continue
            block = cache.tables[seq_id][pos // cache.block_size]
            offset = pos % cache.block_size
            acc += w[pos] * cache.v[block, offset]
        out[i] = acc
    return out
