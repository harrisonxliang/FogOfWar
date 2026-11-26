"""Belief-weighted world sampling + metrics. Settled design (NOTES.md §3):
randomized truncation of S + renormalized weighted mean.

  (a) score all candidates with the belief model -> normalized w_hat;
  (b) epsilon-mix:  w~ = (1 - eps) * w_hat + eps / |S|   (floor: no consistent
      world is ever starved — the true board is always in S);
  (c) SELECT K without replacement proportional to w~ (default mode), or
      deterministic top-k, or uniform (the ablation baseline);
  (d) AGGREGATE with w~ renormalized WITHIN the sample (returned here as
      agg_weights; decide.aggregate consumes them).

Not double-counting: without replacement at K≈8 each board appears once, so
selection frequency cannot carry plausibility — selection only picks WHICH
truncation of S we can afford; the weights act exactly once, inside it.

Rules from the task sheet: never weight selection when |S| <= K (take all),
and epsilon > 0 always. The veto uses its own, more conservative epsilon
(veto_weights): mass(danger) = (1-eps)*w_hat(danger) + eps*count_share, so a
miscalibrated model can suppress danger mass at most down to eps*count_share.

Metrics (need ground truth: self-play or downloaded replays):
- truth_rank / truth_in_sample: where the true board lands;
- effective_sample_size: (sum w)^2 / sum w^2 over the FULL w~;
- captured_mass: sum of w~ over the K picked worlds ("these 8 worlds hold 74%
  of the belief") — the per-turn truncation-quality diagnostic.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boardgen.sample import score

SELECTION_MODES = ('weighted-sample', 'top-k', 'uniform')


def mixed_probabilities(log_weights, epsilon=0.25, temperature=1.0):
    logw = np.asarray(log_weights, dtype=np.float64) / max(temperature, 1e-6)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    return (1.0 - epsilon) * w + epsilon / len(w)


def veto_weights(log_weights, epsilon_veto=0.5):
    """Epsilon-mixed weights for the FULL-S veto mass (NOTES.md §3).

    Separate, more conservative epsilon than sampling: the veto is the last
    line of defense, and its errors are asymmetric — a hidden real king-loss
    world loses the game; an over-veto costs a little strength.
    """
    return mixed_probabilities(log_weights, epsilon=epsilon_veto)


def sample_worlds(fens, k, rng, predictor=None, state=None, epsilon=0.25,
                  temperature=1.0, mode='weighted-sample', danger_index=None):
    """Select worlds and their aggregation weights.

    Returns (indices, agg_weights, diag):
      indices      — chosen world indices into `fens`;
      agg_weights  — w~ renormalized within the sample, summing to 1; feed to
                     decide.aggregate. Uniform when no model is available.
      diag         — {'captured_mass', 'ess_full', 'probabilities'} (None-valued
                     when no model).

    mode: 'weighted-sample' (default, WOR ∝ w~) | 'top-k' (deterministic
    truncation by w~) | 'uniform' (ablation baseline). Falls back to uniform
    when predictor is None. |S| <= k always takes everything (weights still
    returned for aggregation). danger_index, when given, is force-included by
    replacing the lowest-w~ pick; it enters with its own w~ like any other.
    """
    n = len(fens)
    if mode not in SELECTION_MODES:
        raise ValueError(f'unknown selection mode: {mode!r} (see NOTES.md §0)')
    p = None
    if predictor is not None:
        log_weights = score(predictor, state, fens)
        p = mixed_probabilities(log_weights, epsilon, temperature)

    if n <= k:
        indices = list(range(n))
    elif p is None or mode == 'uniform':
        indices = [int(i) for i in rng.choice(n, size=k, replace=False)]
    elif mode == 'top-k':
        indices = [int(i) for i in np.argsort(-p)[:k]]
    else:  # weighted-sample
        indices = [int(i) for i in rng.choice(n, size=k, replace=False, p=p)]

    if danger_index is not None and danger_index not in indices and n > k:
        if p is not None:
            drop = min(indices, key=lambda i: p[i])
        else:
            drop = indices[-1]
        indices[indices.index(drop)] = int(danger_index)

    if p is not None:
        agg = p[indices].astype(np.float64)
        agg = agg / agg.sum()
        diag = {'captured_mass': float(p[indices].sum()),
                'ess_full': effective_sample_size(p),
                'probabilities': p}
    else:
        agg = np.full(len(indices), 1.0 / max(len(indices), 1))
        diag = {'captured_mass': None, 'ess_full': None, 'probabilities': None}
    return indices, agg, diag


def truth_rank(log_weights, true_index):
    w = np.asarray(log_weights)
    return int((w > w[true_index]).sum()) + 1


def effective_sample_size(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    return float((p.sum() ** 2) / (p ** 2).sum())


def weighting_report(predictor, state, fens, true_index, k, rng, epsilon=0.25,
                     temperature=1.0):
    """One position's worth of metrics, as a dict ready for aggregation."""
    log_weights = score(predictor, state, fens)
    p = mixed_probabilities(log_weights, epsilon, temperature)
    size = min(k, len(fens))
    chosen = rng.choice(len(fens), size=size, replace=False, p=p)
    uniform_chosen = set(rng.choice(len(fens), size=size, replace=False))
    return {
        'set_size': len(fens),
        'truth_rank': truth_rank(log_weights, true_index),
        'truth_in_sample_weighted': true_index in set(chosen),
        'truth_in_sample_uniform': true_index in uniform_chosen,
        'ess': effective_sample_size(p),
        'captured_mass': float(p[chosen].sum()),
    }
