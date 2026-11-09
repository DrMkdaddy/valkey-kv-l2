# valkey-kv-l2

A Valkey-backed L2 for an LLM KV cache. With a fixed 64-block L1, 4 of 200
sequences completed; with the L2 behind it, all 200 did. Restoring a prefix
beats recomputing it above roughly 4k tokens.

L1 is a fixed block pool. When it fills, the least recently used sequence
spills to Valkey and is restored on its next read. Blocks are content-addressed
(sha256 over the K and V bytes), so a restored sequence can hit an entry written
by another sequence or by a previous process.

## Results

64-block L1, block size 16, dim 64; 200 sequences of 32–400 tokens, 44,166
tokens total, 43x more KV than L1 can hold.

| Metric | Value |
|---|---|
| Sequences served, fixed pool | **4 of 200 (2%)** |
| Sequences served, tiered | **200 of 200 (100%)** |
| L2 hit rate on restore | **1.00** |
| Mean restore time | **1.7 ms** (localhost) |
| Valkey memory holding 44k tokens | **60 MB** |
| Bytes written to Valkey | 49.7 MB |

Restore versus recompute, Qwen3-4B f16 KV (144 KiB/token), same machine:

| Prefix tokens | Prefill | Restore from Valkey | Prefill / restore |
|---:|---:|---:|---:|
| 512 | 100 ms | 104 ms | 0.97 |
| 2,048 | 373 ms | 416 ms | 0.90 |
| 4,096 | 810 ms | 783 ms | 1.03 |
| 6,144 | 1,310 ms | 1,025 ms | **1.28** |

Break-even is around 4k tokens. Below it, the round trip and transfer cost more
than recomputing. Above it, restore wins because prefill cost rises with context
while restore scales with the bytes transferred.

An external KV store pays when the model is large, the prefix is long, or the
KV is quantized. On this 4B model with 512-token prompts, it is slower than
recomputing.

## How it works

- `src/resp.py`: RESP client using stdlib sockets. PING, SET with EX, GET, DEL,
  EXISTS, DBSIZE, INFO.
- `src/l2.py`: content-addressed block store. sha256 over K and V bytes, zlib
  level 1, TTL, hit/miss and byte counters.
- `src/tiered.py`: tiering policy. Reserves blocks for a whole append before
  writing, spills LRU sequences, restores on read, and spills again if a restore
  needs space.
- `bench/`: capacity benchmark, restore-vs-prefill sweep, and a deterministic
  SVG chart.

## Quick start

Nix provides Valkey, Python with numpy, and make. The benchmark starts and stops
its own Valkey if none is listening.

```bash
nix develop
make bench
```

That runs the round-trip and tiering tests, the capacity benchmark, and
re-renders the chart. To check the committed chart matches the results:

```bash
make check
```

The crossover numbers need a `llama-server` on `:8097`:

```bash
make bench-restore LLAMA_URL=http://127.0.0.1:8097
```

## Correctness

The tests cover lossless encode/decode, content-addressed keys, spill and
restore, and tiering under block pressure. A restored sequence must match the
spilled sequence; otherwise the cache corrupts inference.

```bash
make test
```

## Scope

Block-level caching for a reference KV stack, not a drop-in for a production
serving engine. No TTL tuning, no compression benchmark beyond lossless
round-trip, no cluster mode. The 1.7 ms restore is localhost.

Method and sensitivity: [`docs/kv-offload-crossover.md`](docs/kv-offload-crossover.md).
Chart: [`docs/restore-vs-prefill.svg`](docs/restore-vs-prefill.svg).

## License

MIT
