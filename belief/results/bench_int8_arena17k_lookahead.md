# sample() latency: int8

Python 3.14.10, NumPy 2.3.5, ONNX Runtime 1.24.4, cgroup CPUs 2, intra-op threads 2, J = 8, 200 calls per cell. Init (load + warm-up): 0.23 s.

| batch | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| 16 | 29 / 40 ms | 47 / 66 ms | 88 / 121 ms | 164 / 212 ms |
| 32 | 50 / 73 ms | 90 / 126 ms | 165 / 218 ms | 314 / 398 ms |
| 64 | 97 / 131 ms | 172 / 226 ms | 332 / 422 ms | 643 / 713 ms |
| 128 | 189 / 229 ms | 357 / 413 ms | 694 / 756 ms | 1364 / 1471 ms |

Cells are p50 / p99 of one full sample() call.
Peak RSS: 330 MB.
Target (batch 64, T = 4, p99 < 0.4 s): MISSED (422 ms)
