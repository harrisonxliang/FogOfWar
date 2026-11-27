# sample() latency: fp32

Python 3.14.10, NumPy 2.3.5, ONNX Runtime 1.24.4, cgroup CPUs 2, intra-op threads 2, J = 8, 200 calls per cell. Init (load + warm-up): 0.26 s.

| batch | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| 16 | 38 / 60 ms | 67 / 99 ms | 124 / 170 ms | 239 / 297 ms |
| 32 | 70 / 109 ms | 129 / 174 ms | 245 / 284 ms | 473 / 548 ms |
| 64 | 138 / 192 ms | 262 / 328 ms | 508 / 576 ms | 1005 / 1117 ms |
| 128 | 278 / 332 ms | 538 / 608 ms | 1065 / 1184 ms | 2069 / 2195 ms |

Cells are p50 / p99 of one full sample() call.
Peak RSS: 214 MB.
Target (batch 64, T = 4, p99 < 0.4 s): MISSED (576 ms)
