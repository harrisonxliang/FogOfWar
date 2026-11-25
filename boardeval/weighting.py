"""Belief-weighted world sampling (the weighted-vs-uniform ablation) + metrics.

sample_worlds: pick K worlds from an enumerated set, either uniformly or
proportional to the belief model's score() with an epsilon-uniform mix:
    p(world) = (1 - eps) * softmax(log_weights / temperature) + eps / N
Rules from the task sheet: never weight when |S| <= K (take everything), and
eps > 0 always, so no consistent world is ever starved to zero probability.

Metrics (need ground truth, so: self-play games or downloaded replays):
- truth_rank: where the true board ranks under the weights (1 = best)
- truth_in_sample: whether the true board made it into the K sampled worlds
- effective_sample_size: (sum w)^2 / sum w^2 — how many "real" worlds the
  weight distribution is worth; ESS ~ 1 means collapse onto one board.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boardgen.sample import score


def mixed_probabilities(log_weights, epsilon=0.1, temperature=1.0):
    logw = np.asarray(log_weights, dtype=np.float64) / max(temperature, 1e-6)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    return (1.0 - epsilon) * w + epsilon / len(w)


def sample_worlds(fens, k, rng, predictor=None, state=None, epsilon=0.1,
                  temperature=1.0):
    """Returns (chosen_indices, probabilities_used_or_None).

    Uniform when predictor is None (the ablation baseline) or |S| <= k.
    """
    n = len(fens)
    if n <= k:
        return list(range(n)), None
    if predictor is None:
        return list(rng.choice(n, size=k, replace=False)), None
    log_weights = score(predictor, state, fens)
    p = mixed_probabilities(log_weights, epsilon, temperature)
    return list(rng.choice(n, size=k, replace=False, p=p)), p


def truth_rank(log_weights, true_index):
    w = np.asarray(log_weights)
    return int((w > w[true_index]).sum()) + 1


def effective_sample_size(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    return float((p.sum() ** 2) / (p ** 2).sum())


def weighting_report(predictor, state, fens, true_index, k, rng, epsilon=0.1,
                     temperature=1.0):
    """One position's worth of metrics, as a dict ready for aggregation."""
    log_weights = score(predictor, state, fens)
    p = mixed_probabilities(log_weights, epsilon, temperature)
    chosen = set(rng.choice(len(fens), size=min(k, len(fens)), replace=False, p=p))
    uniform_chosen = set(rng.choice(len(fens), size=min(k, len(fens)), replace=False))
    return {
        'set_size': len(fens),
        'truth_rank': truth_rank(log_weights, true_index),
        'truth_in_sample_weighted': true_index in chosen,
        'truth_in_sample_uniform': true_index in uniform_chosen,
        'ess': effective_sample_size(p),
    }
