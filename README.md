# valkey-kv-l2

A Valkey-backed L2 for an LLM KV cache. L1 is a fixed block pool; when it fills,
the least recently used sequence spills to Valkey and is restored on its next
read. Blocks are content-addressed, so a restored sequence can hit an entry
written by another sequence or by a previous process.

**Why:** a fixed KV pool turns concurrency into a cliff. The moment the pool is
full the server refuses work, even though most of the resident KV belongs to
requests that are finished with. Spilling to a shared store converts that cliff
into a restore cost.

## Result

64-block L1, block size 16, dim 64. 200 sequences of 32-400 tokens, 44,166 tokens
total, which is 43x more KV than L1 can hold.

| Metric | Value |
|---|---|
| Sequences served, fixed pool (L1 only) | **4 of 200 (2%)** |
| Sequences served, tiered | **200 of 200 (100%)** |
| L2 hit rate on restore | **1.00** |
| Mean restore time | **1.7 ms** |
| Valkey memory holding 44k tokens of KV | **60 MB** |
| Bytes written to Valkey | 49.7 MB |

Restoring is not free, and the 1.7 ms is localhost with no network. Across a
network that number becomes the RTT plus transfer, which is why L2 sits behind a
block cache rather than in front of every read.

## Where it actually pays

The capacity number above compares tiering against refusing work. The real
alternative to a restore is a prefill, so both were measured at the same token
counts on the same machine, Qwen3-4B at f16 KV (144 KiB per token).

| prefix tokens | prefill | restore from Valkey | prefill / restore |
|---:|---:|---:|---:|
| 512 | 100 ms | 104 ms | 0.97 |
| 2,048 | 373 ms | 416 ms | 0.90 |
| 4,096 | 810 ms | 783 ms | 1.03 |
| 6,144 | 1,310 ms | 1,025 ms | 1.28 |

**Break-even is around 4k tokens.** Below it, the round trip and the transfer
cost more than recomputing, and the cache is a pessimisation. Above it, restore
wins because prefill grows superlinearly with context while the restore only
grows with bytes.

That is the whole argument for an external KV store: it is a bandwidth bet, not
a latency win. It pays when the model is large, the prefix is long, or the KV is
quantized so the bytes shrink. On a 4B model with 512-token prompts it is
strictly worse than not caching.

## What is implemented

- `src/resp.py` — RESP client, stdlib sockets only. PING, SET with EX, GET, DEL,
  EXISTS, DBSIZE, INFO.
- `src/l2.py` — content-addressed block store. sha256 over K and V bytes, zlib
  level 1, TTL, hit/miss and byte counters.
- `src/tiered.py` — tiering policy. Reserves blocks for a whole append before
  writing, spills LRU sequences, restores on read, spills again if a restore
  itself needs space.

## Correctness

A restored sequence must be identical to the one that was spilled, or the cache
is silently corrupting inference.

```
$ python3 tests/test_tiered.py
ok   encode/decode is lossless
ok   keys are content-addressed
ok   block restored byte-identical
ok   a sequence spilled under block pressure
ok   restored sequence is identical to the pre-spill one
ok   restore produced L2 hits
ok   tiering served 20 sequences with only 8 blocks
all passed (0 failure(s))
```

## Reproduce

```bash
nix shell nixpkgs#valkey -c valkey-server --port 6390 --save '' --appendonly no &
python3 tests/test_tiered.py
python3 bench/bench.py            # writes results/valkey_l2.json
```

## Scope

Block-level caching for a reference KV stack, not a drop-in for a production
serving engine. No eviction TTL tuning, no compression benchmarking beyond
lossless round-trip, no cluster mode. The 1.7 ms restore is localhost.

## License

MIT
