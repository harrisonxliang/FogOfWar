# sample() latency: int8_arena30k_lookahead

Python 3.14.10, NumPy 2.3.5, ONNX Runtime 1.24.4, cgroup CPUs 2, intra-op threads 2, J = 8, 200 calls per cell. Init (load + warm-up): 0.14 s.

| batch | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| 16 | 15 / 21 ms | 25 / 37 ms | 46 / 59 ms | 86 / 103 ms |
| 32 | 27 / 37 ms | 48 / 85 ms | 142 / 199 ms | 288 / 363 ms |
| 64 | 83 / 104 ms | 158 / 192 ms | 296 / 341 ms | 578 / 677 ms |
| 128 | 159 / 180 ms | 297 / 339 ms | 563 / 619 ms | 1105 / 1157 ms |

Cells are p50 / p99 of one full sample() call.
Peak RSS: 331 MB.
Target (batch 64, T = 4, p99 < 0.4 s): MET (341 ms)
