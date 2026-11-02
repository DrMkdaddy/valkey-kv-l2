"""L2 block store for LLM KV cache, backed by Valkey.

L1 is the resident KV pool (see paged-attention-lab). When L1 is full, a block is
serialised, compressed and written to Valkey under a content hash, then freed.
A later read of the same content fetches it back instead of recomputing.

Keys are content-addressed, so two sequences that share a prefix share the L2
entry, and a restarted process still hits it.
"""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field

import numpy as np

from resp import Valkey


def block_key(k: np.ndarray, v: np.ndarray, namespace: str = "kv") -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(k, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return f"{namespace}:{h.hexdigest()[:32]}"


def encode(k: np.ndarray, v: np.ndarray) -> bytes:
    """Pack two float64 arrays into one compressed blob."""
    k = np.ascontiguousarray(k, dtype=np.float64)
    v = np.ascontiguousarray(v, dtype=np.float64)
    header = np.array([k.shape[0], k.shape[1]], dtype=np.int32).tobytes()
    return zlib.compress(header + k.tobytes() + v.tobytes(), level=1)


def decode(blob: bytes, dim: int) -> tuple[np.ndarray, np.ndarray]:
    raw = zlib.decompress(blob)
    rows = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    k = np.frombuffer(raw[8:8 + rows * dim * 8], dtype=np.float64).reshape(rows, dim)
    v = np.frombuffer(raw[8 + rows * dim * 8:], dtype=np.float64).reshape(rows, dim)
    return k.copy(), v.copy()


@dataclass
class L2Stats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    bytes_written: int = 0
    bytes_read: int = 0
    restore_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "writes": self.writes,
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
            "restore_seconds": round(self.restore_seconds, 4),
        }


class L2BlockStore:
    """Content-addressed KV blocks in Valkey, with optional TTL."""

    def __init__(self, client: Valkey, dim: int, ttl_seconds: int | None = 3600,
                 namespace: str = "kv"):
        self.client = client
        self.dim = dim
        self.ttl = ttl_seconds
        self.namespace = namespace
        self.stats = L2Stats()

    def put(self, k: np.ndarray, v: np.ndarray) -> str:
        key = block_key(k, v, self.namespace)
        blob = encode(k, v)
        self.client.set(key, blob, ex=self.ttl)
        self.stats.writes += 1
        self.stats.bytes_written += len(blob)
        return key

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        blob = self.client.get(key)
        if blob is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        self.stats.bytes_read += len(blob)
        return decode(blob, self.dim)

    def has(self, key: str) -> bool:
        return self.client.exists(key)

    def drop(self, key: str) -> None:
        self.client.delete(key)
