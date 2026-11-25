"""Metrics, baselines and figures for the belief model (training-side; uses torch).

Model metrics (per hidden square, i.e. squares with cur == unseen):
  elbo bits    MDLM ELBO estimate: E_t[(1/t) * sum_masked CE] / |U|, in bits
  onestep bits cross-entropy with every unseen square masked (t = 1)
  accuracy     argmax at t = 1 vs truth, overall and per true class
  ECE          per-class calibration of the t = 1 marginals
Baselines, scored the same way:
  last-seen    p = 1 - eps on the last-seen value (empty if never seen),
               eps / 6 elsewhere; eps fitted on train (= its error rate)
  frequency    per-(ply bucket, square) class frequencies of hidden squares in train

Reported overall and stratified by ply (0-20, 21-60, 61+) and by number of
unseen squares, with 95% bootstrap CIs over games.

Sample validity, diversity and log_prob need the ONNX sampler (task 7) and
are not computed here yet.

    .venv/bin/python -m belief.eval --run runs/<name> --split test
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from belief.model import MASK, VIS, build

LN2 = math.log(2)
CLASSES = ('empty', 'P', 'N', 'B', 'R', 'Q', 'K')
PLY_BUCKETS = ((0, 20), (21, 60), (61, 10**9))
UNSEEN_BUCKETS = ((0, 24), (25, 31), (32, 39), (40, 64))


# --- diffusion corruption, shared by train and eval ------------------------------------

def to_long(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: batch[k].long() for k in ('cur', 'last', 'age', 'inv', 'ply', 'to_move', 'tgt')}


def stratified_t(n: int, eps: float, device, gen: torch.Generator | None = None) -> torch.Tensor:
    u = (torch.arange(n, device=device) + torch.rand(1, device=device, generator=gen)) / n
    t = eps + (1 - eps) * u
    return t[torch.randperm(n, device=device, generator=gen)]


def corrupt(b: dict[str, torch.Tensor], t: torch.Tensor, gen: torch.Generator | None = None):
    """Absorbing-state forward process on unseen squares only.
    Returns xt and the boolean mask of masked squares."""
    unseen = b['cur'] == 0
    masked = unseen & (torch.rand(unseen.shape, device=unseen.device, generator=gen) < t[:, None])
    xt = torch.where(unseen, b['tgt'].clamp(min=0), torch.full_like(b['tgt'], VIS))
    xt = torch.where(masked, torch.full_like(xt, MASK), xt)
    return xt, masked, unseen


def forward(model, b, xt):
    return model(b['cur'], b['last'], b['age'], xt, b['inv'], b['ply'], b['to_move'])


def diffusion_loss(model, b, t, loss_type: str = 'elbo', gen=None):
    """Returns (loss for backprop, per-example weighted NLL sum [B], per-example hidden count [B])."""
    xt, masked, unseen = corrupt(b, t, gen)
    logits = forward(model, b, xt).float()
    ce = F.cross_entropy(logits.transpose(1, 2), b['tgt'].clamp(min=0), reduction='none')   # [B, 64]
    ce = ce * masked
    elbo_ex = ce.sum(1) / t                                   # MDLM weight -alpha'/(1-alpha) = 1/t
    n_hidden = unseen.sum(1)
    if loss_type == 'elbo':
        loss = elbo_ex.mean()
    elif loss_type == 'maskgit':
        loss = ce.sum() / masked.sum().clamp(min=1)
    else:
        raise ValueError(loss_type)
    return loss, elbo_ex.detach(), n_hidden


# --- per-example model scores -----------------------------------------------------------

@torch.no_grad()
def model_scores(model, data: dict[str, np.ndarray], device, batch_size: int = 4096,
                 elbo_draws: int = 2, eps: float = 1e-3, seed: int = 1234, with_probs: bool = False):
    """Per-example sums over hidden squares. Deterministic given seed."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    n = len(data['t'])
    out = {'elbo': np.zeros(n), 'onestep': np.zeros(n), 'n_hidden': np.zeros(n, dtype=np.int64)}
    probs_list, tgt_list = [], []
    for s in range(0, n, batch_size):
        b = to_long({k: torch.from_numpy(data[k][s:s + batch_size]).to(device) for k in
                     ('cur', 'last', 'age', 'inv', 'ply', 'to_move', 'tgt')})
        m = len(b['ply'])
        elbo = torch.zeros(m, device=device)
        for _ in range(elbo_draws):
            _, e, nh = diffusion_loss(model, b, stratified_t(m, eps, device, gen), gen=gen)
            elbo += e / elbo_draws
        ones = torch.ones(m, device=device)
        xt, masked, unseen = corrupt(b, ones)
        logp = F.log_softmax(forward(model, b, xt).float(), -1)           # [m, 64, 7]
        nll = -logp.gather(-1, b['tgt'].clamp(min=0)[..., None])[..., 0] * unseen
        out['elbo'][s:s + m] = elbo.cpu().numpy()
        out['onestep'][s:s + m] = nll.sum(1).cpu().numpy()
        out['n_hidden'][s:s + m] = nh.cpu().numpy()
        if with_probs:
            probs_list.append(logp.exp()[unseen].cpu().numpy())
            tgt_list.append(b['tgt'][unseen].cpu().numpy())
    if with_probs:
        out['probs'] = np.concatenate(probs_list)
        out['probs_tgt'] = np.concatenate(tgt_list)
    return out


# --- baselines -----------------------------------------------------------------------------

def last_seen_guess(last: np.ndarray) -> np.ndarray:
    """last code -> class guess: never seen / no enemy -> empty, else the enemy type."""
    return np.where(last >= 2, last.astype(np.int64) - 1, 0)


class LastSeenBaseline:
    def __init__(self, train: dict[str, np.ndarray]):
        hidden = train['cur'] == 0
        guess = last_seen_guess(train['last'])
        err = float((guess[hidden] != train['tgt'][hidden]).mean())
        self.eps = min(max(err, 1e-4), 0.999)

    def log_probs(self, data) -> np.ndarray:
        """[N, 64, 7] natural-log probabilities."""
        guess = last_seen_guess(data['last'])
        lp = np.full(guess.shape + (7,), math.log(self.eps / 6), dtype=np.float32)
        np.put_along_axis(lp, guess[..., None], math.log(1 - self.eps), axis=-1)
        return lp


class FrequencyBaseline:
    def __init__(self, train: dict[str, np.ndarray], alpha: float = 0.5):
        counts = np.full(32 * 64 * 7, alpha)
        for s in range(0, len(train['t']), 500_000):
            cur, tgt = train['cur'][s:s + 500_000], train['tgt'][s:s + 500_000]
            ply = train['ply'][s:s + 500_000].astype(np.int64)
            ex, sq = np.nonzero(cur == 0)
            counts += np.bincount((ply[ex] * 64 + sq) * 7 + tgt[ex, sq], minlength=counts.size)
        counts = counts.reshape(32, 64, 7)
        self.logp = np.log(counts / counts.sum(-1, keepdims=True)).astype(np.float32)

    def log_probs(self, data) -> np.ndarray:
        return self.logp[data['ply'].astype(np.int64)]


def baseline_nll(baseline, data, chunk: int = 200_000) -> np.ndarray:
    """Per-example NLL sum over hidden squares (nats)."""
    out = np.zeros(len(data['t']))
    for s in range(0, len(out), chunk):
        sl = {k: data[k][s:s + chunk] for k in ('cur', 'last', 'ply', 'tgt')}
        lp = baseline.log_probs(sl)
        hidden = sl['cur'] == 0
        tgt = np.where(hidden, sl['tgt'], 0).astype(np.int64)
        nll = -np.take_along_axis(lp, tgt[..., None], -1)[..., 0] * hidden
        out[s:s + chunk] = nll.sum(1)
    return out


# --- aggregation -----------------------------------------------------------------------------

def bits_with_ci(nll: np.ndarray, n_hidden: np.ndarray, games: np.ndarray, n_boot: int = 1000,
                 seed: int = 0) -> dict:
    """Bits per hidden square = sum nll / sum hidden / ln 2, with a bootstrap CI over games."""
    total = nll.sum() / max(n_hidden.sum(), 1) / LN2
    ug, inv = np.unique(games, return_inverse=True)
    g_nll = np.bincount(inv, weights=nll, minlength=len(ug))
    g_n = np.bincount(inv, weights=n_hidden, minlength=len(ug))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), size=(n_boot, len(ug)))
    boots = g_nll[idx].sum(1) / np.maximum(g_n[idx].sum(1), 1) / LN2
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'bits': float(total), 'ci95': [float(lo), float(hi)], 'n_examples': int(len(nll)),
            'n_games': int(len(ug)), 'n_hidden': int(n_hidden.sum())}


def strata(data) -> dict[str, np.ndarray]:
    n_unseen = (data['cur'] == 0).sum(1)
    t = data['t'].astype(np.int64)
    out = {'all': np.ones(len(t), dtype=bool)}
    for lo, hi in PLY_BUCKETS:
        out[f'ply_{lo}-{hi if hi < 10**9 else "inf"}'] = (t >= lo) & (t <= hi)
    for lo, hi in UNSEEN_BUCKETS:
        out[f'unseen_{lo}-{hi}'] = (n_unseen >= lo) & (n_unseen <= hi)
    return out


def ece(probs: np.ndarray, tgt: np.ndarray, n_bins: int = 15) -> dict:
    """Per-class expected calibration error of per-square marginals, plus reliability curves."""
    out = {}
    edges = np.linspace(0, 1, n_bins + 1)
    for k, name in enumerate(CLASSES):
        p, y = probs[:, k], (tgt == k).astype(np.float64)
        b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins)
        conf = np.bincount(b, weights=p, minlength=n_bins) / np.maximum(cnt, 1)
        acc = np.bincount(b, weights=y, minlength=n_bins) / np.maximum(cnt, 1)
        out[name] = {'ece': float((cnt * np.abs(conf - acc)).sum() / max(cnt.sum(), 1)),
                     'conf': conf.tolist(), 'acc': acc.tolist(), 'count': cnt.tolist()}
    return out


def accuracy(probs: np.ndarray, tgt: np.ndarray) -> dict:
    pred = probs.argmax(1)
    per = {name: float((pred[tgt == k] == k).mean()) if (tgt == k).any() else None
           for k, name in enumerate(CLASSES)}
    vals = [v for v in per.values() if v is not None]
    return {'overall': float((pred == tgt).mean()), 'balanced': float(np.mean(vals)), 'per_class': per,
            'class_freq': {name: float((tgt == k).mean()) for k, name in enumerate(CLASSES)}}


def plot_calibration(cal: dict, path: Path, title: str = ''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(3 * len(CLASSES), 3.2), sharey=True)
    for ax, name in zip(axes, CLASSES):
        c = cal[name]
        cnt = np.array(c['count'])
        keep = cnt > 0
        ax.plot([0, 1], [0, 1], color='0.75', lw=1)
        ax.plot(np.array(c['conf'])[keep], np.array(c['acc'])[keep], marker='o', ms=3, color='#3366aa')
        ax.set_title(f'{name}  ECE {c["ece"]:.3f}', fontsize=10)
        ax.set_xlabel('predicted p')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('observed frequency')
    fig.suptitle(title or 'Reliability of one-step marginals (t = 1)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_run(run_dir: Path):
    """Loss curves from a run's metrics.jsonl -> loss_curves.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in (run_dir / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    cfg = json.loads((run_dir / 'config.json').read_text())
    tr = [r for r in rows if 'train_bits' in r]
    va = [r for r in rows if 'val_elbo_bits' in r]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([r['step'] for r in tr], [r['train_bits'] for r in tr], color='0.6', lw=1, label='train ELBO')
    ax.plot([r['step'] for r in va], [r['val_elbo_bits'] for r in va], color='#3366aa', lw=2, label='val ELBO')
    ax.plot([r['step'] for r in va], [r['val_onestep_bits'] for r in va], color='#cc6633', lw=2,
            label='val one-step (t=1)')
    base = cfg.get('baselines', {})
    for key, color, label in (('last_seen_val_bits', '#888833', 'last-seen baseline'),
                              ('frequency_val_bits', '#883388', 'frequency baseline')):
        if key in base:
            ax.axhline(base[key], color=color, ls='--', lw=1, label=label)
    ax.set_xlabel('step')
    ax.set_ylabel('bits per hidden square')
    ax.set_title(run_dir.name)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / 'loss_curves.png', dpi=130)
    plt.close(fig)


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build(ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def main():
    from belief.dataset import load_shards
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True, help='run directory with best.pt and config.json')
    ap.add_argument('--data', type=Path, default=None, help='shard dir (default: the run\'s)')
    ap.add_argument('--split', default='test')
    ap.add_argument('--max-examples', type=int, default=None)
    ap.add_argument('--elbo-draws', type=int, default=4)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_checkpoint(args.run / 'best.pt', device)
    data_dir = args.data or Path(ckpt['train_config']['data'])
    train = load_shards(data_dir, 'train')
    data = load_shards(data_dir, args.split, args.max_examples)
    sc = model_scores(model, data, device, elbo_draws=args.elbo_draws, with_probs=True)
    base = {'last_seen': baseline_nll(LastSeenBaseline(train), data),
            'frequency': baseline_nll(FrequencyBaseline(train), data)}
    del train
    results = {'run': str(args.run), 'split': args.split, 'data': str(data_dir), 'step': ckpt['step'],
               'git': ckpt.get('git'), 'metrics': {}}
    for name, mask in strata(data).items():
        if not mask.any():
            continue
        g, nh = data['game'][mask], sc['n_hidden'][mask]
        results['metrics'][name] = {
            'model_elbo': bits_with_ci(sc['elbo'][mask], nh, g),
            'model_onestep': bits_with_ci(sc['onestep'][mask], nh, g),
            'last_seen': bits_with_ci(base['last_seen'][mask], nh, g),
            'frequency': bits_with_ci(base['frequency'][mask], nh, g),
        }
    results['accuracy_onestep'] = accuracy(sc['probs'], sc['probs_tgt'])
    cal = ece(sc['probs'], sc['probs_tgt'])
    results['ece_onestep'] = {k: v['ece'] for k, v in cal.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f'{args.run.name}_{args.split}'
    (args.out / f'{stem}_metrics.json').write_text(json.dumps({**results, 'calibration': cal}, indent=1))
    plot_calibration(cal, args.out / f'{stem}_calibration.png', f'{args.run.name} ({args.split}): one-step marginals')
    lines = [f'| stratum | model ELBO | model one-step | last-seen | frequency | games |', '|---|---|---|---|---|---|']
    fmt = lambda m: f'{m["bits"]:.3f} [{m["ci95"][0]:.3f}, {m["ci95"][1]:.3f}]'
    for name, m in results['metrics'].items():
        lines.append(f'| {name} | {fmt(m["model_elbo"])} | {fmt(m["model_onestep"])} | {fmt(m["last_seen"])} '
                     f'| {fmt(m["frequency"])} | {m["model_elbo"]["n_games"]} |')
    acc = results['accuracy_onestep']
    lines += ['', f'one-step accuracy: overall {acc["overall"]:.3f}, balanced {acc["balanced"]:.3f}; per class: ' +
              ', '.join(f'{k} {v:.3f}' for k, v in acc['per_class'].items() if v is not None),
              'ECE: ' + ', '.join(f'{k} {v:.4f}' for k, v in results['ece_onestep'].items())]
    (args.out / f'{stem}_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
