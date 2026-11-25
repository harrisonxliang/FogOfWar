"""Optional Weights & Biases mirror of a run's local logs (training-side only).

The local files in runs/<name>/ (config.json, metrics.jsonl, *.png) stay the
source of truth; wandb gets the same metrics and config, never game data.
Enable during training with --set wandb=true [wandb_project=...] [wandb_entity=...].
Requires a one-time `.venv/bin/wandb login`.

Backfill finished runs from their local files:
    .venv/bin/python -m belief.wandb_log runs/smoke runs/arena3k_s0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_GLOBS = ('loss_curves.png', 'eval/*_calibration.png')


class WandbRun:
    def __init__(self, meta: dict, run_dir: Path, project: str, entity: str = '', group: str = ''):
        try:
            import wandb
        except ImportError as err:
            raise SystemExit('wandb=true but wandb is not installed: .venv/bin/pip install wandb') from err
        self.wandb = wandb
        cfg = meta.get('config', {})
        self.run = wandb.init(
            project=project, entity=entity or None, name=run_dir.name, dir=str(run_dir),
            group=group or None, config=meta, job_type='train',
            tags=[Path(cfg.get('data', '')).name, cfg.get('loss', '')], reinit='finish_previous')
        # step-indexed metrics; evals and train rows share the global step
        self.run.define_metric('step')
        self.run.define_metric('*', step_metric='step')
        self.run.summary['n_params'] = meta.get('n_params')
        for k, v in (meta.get('baselines') or {}).items():
            self.run.summary[f'baseline/{k}'] = v

    def log(self, row: dict) -> None:
        out = {'step': row['step']}
        for k, v in row.items():
            if k == 'step':
                continue
            prefix = 'val/' if k.startswith('val_') else 'train/'
            out[prefix + k.removeprefix('val_')] = v
        self.run.log(out)

    def finish(self, run_dir: Path, summary: dict | None = None) -> None:
        for k, v in (summary or {}).items():
            self.run.summary[k] = v
        for pattern in IMAGE_GLOBS:
            for img in sorted(run_dir.glob(pattern)):
                self.run.log({f'figures/{img.stem}': self.wandb.Image(str(img))})
        for table in sorted(run_dir.glob('eval/*_metrics.json')):
            res = json.loads(table.read_text())
            split = res.get('split', 'test')
            for stratum, m in res.get('metrics', {}).items():
                for method, v in m.items():
                    self.run.summary[f'{split}/{stratum}/{method}_bits'] = v['bits']
        self.run.finish()


def backfill(run_dir: Path, project: str, entity: str = '') -> None:
    meta = json.loads((run_dir / 'config.json').read_text())
    run = WandbRun(meta, run_dir, project, entity)
    for line in (run_dir / 'metrics.jsonl').read_text().splitlines():
        if line.strip():
            run.log(json.loads(line))
    run.finish(run_dir, {k: meta[k] for k in ('best_val_elbo_bits', 'best_step', 'final_step') if k in meta})
    print(f'uploaded {run_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--project', default='fow-belief')
    ap.add_argument('--entity', default='')
    args = ap.parse_args()
    for run_dir in args.runs:
        backfill(run_dir, args.project, args.entity)


if __name__ == '__main__':
    main()
