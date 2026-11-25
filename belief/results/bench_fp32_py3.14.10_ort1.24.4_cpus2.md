# sample() latency: py3.14.10_ort1.24.4_cpus2

Python 3.14.10, NumPy 2.3.5, ONNX Runtime 1.24.4, cgroup CPUs 2, intra-op threads 2, J = 8, 200 calls per cell. Init (load + warm-up): 0.21 s.

| batch | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| 16 | 33 / 45 ms | 60 / 90 ms | 119 / 163 ms | 224 / 281 ms |
| 32 | 61 / 87 ms | 115 / 165 ms | 224 / 270 ms | 447 / 518 ms |
| 64 | 124 / 155 ms | 238 / 296 ms | 472 / 569 ms | 935 / 1095 ms |
| 128 | 252 / 322 ms | 499 / 576 ms | 962 / 1117 ms | 1907 / 2083 ms |

Cells are p50 / p99 of one full sample() call.
Peak RSS: 182 MB.
Target (batch 64, T = 4, p99 < 0.4 s): MISSED (569 ms)
