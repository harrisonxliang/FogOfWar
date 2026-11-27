# sample() latency: int8_py3.14.10_ort1.24.4_cpus2

Python 3.14.10, NumPy 2.3.5, ONNX Runtime 1.24.4, cgroup CPUs 2, intra-op threads 2, J = 8, 200 calls per cell. Init (load + warm-up): 0.52 s.

| batch | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| 16 | 25 / 45 ms | 43 / 57 ms | 80 / 100 ms | 154 / 202 ms |
| 32 | 42 / 89 ms | 82 / 130 ms | 149 / 196 ms | 287 / 345 ms |
| 64 | 81 / 130 ms | 157 / 203 ms | 301 / 372 ms | 587 / 649 ms |
| 128 | 166 / 209 ms | 326 / 384 ms | 640 / 712 ms | 1255 / 1442 ms |

Cells are p50 / p99 of one full sample() call.
Peak RSS: 331 MB.
Target (batch 64, T = 4, p99 < 0.4 s): MET (372 ms)
