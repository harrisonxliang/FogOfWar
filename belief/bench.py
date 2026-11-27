"""CPU latency benchmark for DiffusionBelief.sample() (NumPy + onnxruntime only).

Grid: batch (total sequences = views x J, J = 8) in {16, 32, 64, 128} x
T in {1, 2, 4, 8}. Reports p50/p99 latency of one full sample() call and peak
RSS, plus init (session load + one warm-up batch) and forward-only time.
Views are real validation positions read from .npz shards.
Target: batch 64 at T = 4 under 0.4 s at p99.

Run inside the arena-like container:
    docker build -t fow-bench -f belief/bench.Dockerfile belief
    docker run --rm --cpus=2 --memory=4g -v "$PWD":/work:ro -v "$PWD/belief/results":/out \
        -w /work fow-bench python -m belief.bench --shards data/shards/arena3k --out /out
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from belief.features import view_from_arrays
from belief.sampler import DiffusionBelief, SamplerConfig

TARGET = {'batch': 64, 'T': 4, 'p99_s': 0.4}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cgroup_cpus() -> str:
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        return 'unlimited' if quota == 'max' else f'{int(quota) / int(period):g}'
    except (OSError, ValueError):
        return 'unknown'


def load_views(shards: Path, n: int, seed: int):
    files = sorted((shards / 'val').glob('shard_*.npz')) or sorted((shards / 'test').glob('shard_*.npz'))
    with np.load(files[0]) as z:
        z = {k: z[k] for k in ('cur', 'last', 'age', 'inv', 'ply', 'to_move', 'viewer', 't')}
    idx = np.random.default_rng(seed).choice(len(z['t']), n, replace=False)
    return [view_from_arrays(z['cur'][i], z['last'][i], z['age'][i], z['inv'][i], z['ply'][i],
                             z['to_move'][i], z['viewer'][i] == 0, z['t'][i]) for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--onnx', default='belief/B.onnx')
    ap.add_argument('--shards', type=Path, default=Path('data/shards/arena3k'))
    ap.add_argument('--out', type=Path, default=Path('belief/results'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--J', type=int, default=8)
    ap.add_argument('--batches', type=int, nargs='+', default=[16, 32, 64, 128])
    ap.add_argument('--steps', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--reps', type=int, default=200)
    args = ap.parse_args()
    import onnxruntime as ort

    t0 = time.perf_counter()
    belief = DiffusionBelief(args.onnx, threads=args.threads, config=SamplerConfig())
    t_load = time.perf_counter() - t0
    belief.warmup(batch=64, T=1)
    t_init = time.perf_counter() - t0

    pool = load_views(args.shards, 4096, seed=0)
    rng = np.random.default_rng(0)
    env = {'python': platform.python_version(), 'numpy': np.__version__, 'onnxruntime': ort.__version__,
           'machine': platform.machine(), 'cpu': platform.processor() or platform.uname().machine,
           'os_cpu_count': os.cpu_count(), 'cgroup_cpus': cgroup_cpus(), 'threads': args.threads,
           'onnx': args.onnx, 'onnx_bytes': os.path.getsize(args.onnx)}
    print(json.dumps(env), file=sys.stderr)
    print(f'init: session load {t_load:.3f}s, load + warm-up {t_init:.3f}s', file=sys.stderr)

    rows = []
    for batch in args.batches:
        n_views = max(1, batch // args.J)
        for T in args.steps:
            times, fwd = [], []
            for r in range(args.reps + 3):
                views = [pool[i] for i in rng.choice(len(pool), n_views, replace=False)]
                s = time.perf_counter()
                belief.sample(views, args.J, T, rng)
                if r >= 3:                                         # first calls of a new shape are warm-up
                    times.append(time.perf_counter() - s)
            # forward-only cost of one denoising step at this batch size
            x = {k: np.zeros((batch, 64), np.int64) for k in ('cur', 'last', 'age', 'xt')}
            x.update(inv=np.ones((batch, 7), np.int64), ply=np.zeros(batch, np.int64), to_move=np.ones(batch, np.int64))
            for _ in range(20):
                s = time.perf_counter()
                belief._forward(x)
                fwd.append(time.perf_counter() - s)
            t = np.array(times)
            row = {'batch': batch, 'views': n_views, 'J': args.J, 'T': T, 'reps': len(t),
                   'p50_s': float(np.percentile(t, 50)), 'p99_s': float(np.percentile(t, 99)),
                   'mean_s': float(t.mean()), 'forward_p50_s': float(np.median(fwd)),
                   'peak_rss_mb': peak_rss_mb()}
            rows.append(row)
            print(f'batch {batch:4d} T {T}: p50 {row["p50_s"] * 1e3:7.1f} ms  p99 {row["p99_s"] * 1e3:7.1f} ms  '
                  f'(one forward {row["forward_p50_s"] * 1e3:5.1f} ms)  peak RSS {row["peak_rss_mb"]:.0f} MB',
                  file=sys.stderr, flush=True)

    tgt = next((r for r in rows if r['batch'] == TARGET['batch'] and r['T'] == TARGET['T']), None)
    result = {'env': env, 'init': {'session_load_s': t_load, 'load_plus_warmup_s': t_init},
              'target': TARGET, 'target_met': bool(tgt and tgt['p99_s'] < TARGET['p99_s']), 'grid': rows}
    tag = args.tag or f'py{platform.python_version()}_ort{ort.__version__}_cpus{env["cgroup_cpus"]}'
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'bench_{tag}.json').write_text(json.dumps(result, indent=1))
    lines = [f'# sample() latency: {tag}', '',
             f'Python {env["python"]}, NumPy {env["numpy"]}, ONNX Runtime {env["onnxruntime"]}, '
             f'cgroup CPUs {env["cgroup_cpus"]}, intra-op threads {args.threads}, J = {args.J}, '
             f'{args.reps} calls per cell. Init (load + warm-up): {t_init:.2f} s.', '',
             '| batch | T=' + ' | T='.join(map(str, args.steps)) + ' |', '|---|' + '---|' * len(args.steps)]
    for batch in args.batches:
        cells = [r for r in rows if r['batch'] == batch]
        lines.append(f'| {batch} | ' + ' | '.join(f'{r["p50_s"] * 1e3:.0f} / {r["p99_s"] * 1e3:.0f} ms' for r in cells) + ' |')
    lines += ['', 'Cells are p50 / p99 of one full sample() call.',
              f'Peak RSS: {max(r["peak_rss_mb"] for r in rows):.0f} MB.',
              f'Target (batch 64, T = 4, p99 < 0.4 s): {"MET" if result["target_met"] else "MISSED"}'
              + (f' ({tgt["p99_s"] * 1e3:.0f} ms)' if tgt else '')]
    (args.out / f'bench_{tag}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
