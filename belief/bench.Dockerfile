# Approximates the arena play-time runtime for belief/bench.py:
# Python 3.14, NumPy 2.3, ONNX Runtime 1.24, CPU only. Run with --cpus=2 --memory=4g.
FROM python:3.14-slim
RUN pip install --no-cache-dir "numpy==2.3.*" "onnxruntime==1.24.*"
ENV PYTHONDONTWRITEBYTECODE=1
