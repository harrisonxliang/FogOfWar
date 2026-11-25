"""Train model B (masked discrete diffusion, MDLM-style, linear schedule alpha_t = 1 - t).

Config: a dataclass with defaults from the handoff; override with a YAML/JSON
file (--config) and/or --set key=value (model fields as model.d_model=96).
Every run writes runs/<name>/{config.json, metrics.jsonl, best.pt, last.pt,
loss_curves.png}; config.json records the full config, seed, git commit and
dirty flag, parameter count, data summary and baseline scores.

    # sanity 1: overfit 1k examples
    .venv/bin/python -m belief.train --data data/shards/selfplay --name overfit1k \
        --set subset=1000 max_steps=3000 model.dropout=0 weight_decay=0 warmup_steps=100
    # sanity 2/3: smoke set, then arena
    .venv/bin/python -m belief.train --data data/shards/selfplay --name smoke
    .venv/bin/python -m belief.train --data data/shards/arena --name arena_s0 --set seed=0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from belief.dataset import load_shards
from belief.eval import (FrequencyBaseline, LastSeenBaseline, LN2, baseline_nll, diffusion_loss,
                         model_scores, plot_run, stratified_t, to_long)
from belief.model import ModelConfig, build, n_params

ROOT = Path(__file__).resolve().parent.parent
TENSOR_KEYS = ('cur', 'last', 'age', 'tgt', 'inv', 'ply', 'to_move')


@dataclass
class TrainConfig:
    data: str = 'data/shards/selfplay'
    out: str = 'runs'
    name: str = ''
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: str = 'elbo'                 # 'elbo' (MDLM, 1/t weights) or 'maskgit' (unweighted masked CE)
    eps: float = 1e-3                  # t ~ U(eps, 1), stratified across the batch
    lr: float = 3e-4
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 1024
    max_epochs: float = 30.0           # upper bound; early stopping on val ELBO usually ends sooner
    max_steps: int = 0                 # 0 = derive from max_epochs
    eval_every: int = 1000             # steps (also at the end of training)
    patience: int = 8                  # evals without a relative val-ELBO improvement > min_delta
    min_delta: float = 1e-3
    val_examples: int = 200_000        # cap for periodic validation (full split in eval.py)
    val_elbo_draws: int = 2
    bf16: bool = True
    compile: bool = False
    grad_clip: float = 1.0
    subset: int = 0                    # >0: train on the first N train examples and validate on them (overfit test)
    log_every: int = 100
    device: str = 'cuda'


def apply_overrides(cfg: TrainConfig, overrides: dict) -> TrainConfig:
    for key, value in overrides.items():
        target, attr = cfg, key
        if '.' in key:
            head, attr = key.split('.', 1)
            target = getattr(cfg, head)
        if not hasattr(target, attr):
            raise KeyError(f'unknown config key {key}')
        if isinstance(value, dict) and dataclasses.is_dataclass(getattr(target, attr)):
            apply_overrides(getattr(target, attr), value)
            continue
        old = getattr(target, attr)
        if isinstance(value, str):
            if isinstance(old, bool):
                value = value.lower() in ('1', 'true', 'yes', 'on')
            elif isinstance(old, (int, float)) and not isinstance(old, bool):
                value = type(old)(float(value)) if isinstance(old, int) else float(value)
            elif isinstance(old, tuple):
                value = tuple(float(x) for x in value.strip('()').split(','))
        setattr(target, attr, tuple(value) if isinstance(old, tuple) else value)
    return cfg


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    return {'commit': run('git', 'rev-parse', 'HEAD'), 'branch': run('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty': bool(run('git', 'status', '--porcelain'))}


def lr_at(step: int, cfg: TrainConfig, total: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


def to_device(arrays: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(arrays[k])).to(device) for k in TENSOR_KEYS}


def evaluate(model, val, device, cfg) -> dict:
    sc = model_scores(model, val, device, elbo_draws=cfg.val_elbo_draws, eps=cfg.eps)
    nh = max(sc['n_hidden'].sum(), 1)
    return {'val_elbo_bits': float(sc['elbo'].sum() / nh / LN2),
            'val_onestep_bits': float(sc['onestep'].sum() / nh / LN2)}


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_np = load_shards(cfg.data, 'train')
    if cfg.subset:
        train_np = {k: v[:cfg.subset] for k, v in train_np.items()}
        val_np = train_np
    else:
        val_np = load_shards(cfg.data, 'val', cfg.val_examples)
    n_train = len(train_np['t'])
    steps_per_epoch = max(1, n_train // cfg.batch_size) if n_train >= cfg.batch_size else 1
    total = cfg.max_steps or int(cfg.max_epochs * steps_per_epoch)

    ls, fq = LastSeenBaseline(train_np), FrequencyBaseline(train_np)
    vh = max((val_np['cur'] == 0).sum(), 1)
    baselines = {'last_seen_eps': ls.eps,
                 'last_seen_val_bits': float(baseline_nll(ls, val_np).sum() / vh / LN2),
                 'frequency_val_bits': float(baseline_nll(fq, val_np).sum() / vh / LN2)}

    model = build(cfg.model).to(device)
    # weight decay on Linear weights only; no decay on embeddings, biases, LayerNorm
    decay_ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Linear)}
    decay = [p for p in model.parameters() if id(p) in decay_ids]
    no_decay = [p for p in model.parameters() if id(p) not in decay_ids]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=cfg.lr, betas=tuple(cfg.betas), fused=device.type == 'cuda')
    step_model = torch.compile(model) if cfg.compile else model

    name = cfg.name or time.strftime('run_%Y%m%d_%H%M%S')
    run_dir = ROOT / cfg.out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {'config': asdict(cfg), 'seed': cfg.seed, 'git': git_info(), 'n_params': n_params(model),
            'n_train': n_train, 'n_val': len(val_np['t']), 'steps_per_epoch': steps_per_epoch,
            'total_steps': total, 'baselines': baselines, 'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}
    (run_dir / 'config.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ('n_params', 'n_train', 'n_val', 'total_steps', 'baselines')}))
    metrics_f = open(run_dir / 'metrics.jsonl', 'w')

    def log(row):
        metrics_f.write(json.dumps(row) + '\n')
        metrics_f.flush()

    data = to_device(train_np, device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    best, best_step, bad_evals = float('inf'), -1, 0
    t0 = time.time()
    perm, pos, epoch = torch.randperm(n_train, device=device, generator=gen), 0, 0
    zero = lambda: torch.zeros((), device=device)
    run_loss, run_nll, run_hidden = zero(), zero(), zero()
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == 'cuda')

    def save(path, step, extra):
        torch.save({'model': model.state_dict(), 'model_config': asdict(cfg.model), 'train_config': asdict(cfg),
                    'step': step, 'git': meta['git'], **extra}, path)

    for step in range(total):
        if pos + cfg.batch_size > n_train and n_train >= cfg.batch_size:
            perm, pos, epoch = torch.randperm(n_train, device=device, generator=gen), 0, epoch + 1
        idx = perm[pos:pos + cfg.batch_size] if n_train >= cfg.batch_size else \
            torch.randint(0, n_train, (cfg.batch_size,), device=device, generator=gen)
        pos += cfg.batch_size
        b = to_long({k: v[idx] for k, v in data.items()})
        for g in opt.param_groups:
            g['lr'] = lr_at(step, cfg, total)
        model.train()
        with autocast:
            loss, elbo_ex, n_hidden = diffusion_loss(step_model, b, stratified_t(len(idx), cfg.eps, device, gen),
                                                     cfg.loss, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        run_loss += loss.detach()
        run_nll += elbo_ex.sum()
        run_hidden += n_hidden.sum()
        if (step + 1) % cfg.log_every == 0:
            row = {'step': step + 1, 'epoch': step / steps_per_epoch, 'lr': opt.param_groups[0]['lr'],
                   'train_loss': run_loss.item() / cfg.log_every,
                   'train_bits': (run_nll / run_hidden.clamp(min=1)).item() / LN2, 'time': time.time() - t0}
            log(row)
            run_loss, run_nll, run_hidden = zero(), zero(), zero()
        if (step + 1) % cfg.eval_every == 0 or step + 1 == total:
            ev = evaluate(model, val_np, device, cfg)
            row = {'step': step + 1, 'epoch': step / steps_per_epoch, **ev, 'time': time.time() - t0}
            log(row)
            improved = ev['val_elbo_bits'] < best * (1 - cfg.min_delta)
            if ev['val_elbo_bits'] < best:
                best, best_step = ev['val_elbo_bits'], step + 1
                save(run_dir / 'best.pt', step + 1, {'val': ev})
            bad_evals = 0 if improved else bad_evals + 1
            print(f'step {step + 1}/{total} ep {step / steps_per_epoch:.2f} '
                  f'val elbo {ev["val_elbo_bits"]:.4f} one-step {ev["val_onestep_bits"]:.4f} '
                  f'(best {best:.4f} @ {best_step}) {time.time() - t0:.0f}s', flush=True)
            if bad_evals >= cfg.patience:
                print(f'early stop: no {cfg.min_delta:.1%} improvement in {cfg.patience} evals', flush=True)
                break
    save(run_dir / 'last.pt', step + 1, {})
    metrics_f.close()
    meta.update(best_val_elbo_bits=best, best_step=best_step, final_step=step + 1,
                finished=time.strftime('%Y-%m-%dT%H:%M:%S'), train_seconds=time.time() - t0)
    (run_dir / 'config.json').write_text(json.dumps(meta, indent=1))
    plot_run(run_dir)
    print(f'done: best val ELBO {best:.4f} bits/hidden square at step {best_step}; '
          f'baselines last-seen {baselines["last_seen_val_bits"]:.4f}, '
          f'frequency {baselines["frequency_val_bits"]:.4f}; {run_dir}')
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, help='YAML or JSON file of TrainConfig overrides')
    ap.add_argument('--data')
    ap.add_argument('--name')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    args = ap.parse_args()
    cfg = TrainConfig()
    if args.config:
        text = args.config.read_text()
        if args.config.suffix in ('.yaml', '.yml'):
            import yaml
            apply_overrides(cfg, yaml.safe_load(text) or {})
        else:
            apply_overrides(cfg, json.loads(text))
    if args.data:
        cfg.data = args.data
    if args.name:
        cfg.name = args.name
    apply_overrides(cfg, dict(kv.split('=', 1) for kv in args.set))
    train(cfg)


if __name__ == '__main__':
    main()
