"""Aggregation across worlds + the king-safety veto -> final move choice.

Input: V[move][world] — the evaluated outcome of each candidate move in each
sampled world — plus world weights. Output: one move.

- aggregate: per-move score = (1 - lam) * weighted_mean + lam * worst_case.
  lam = 0 is expected value; lam -> 1 is paranoid (the proposal's danger tilt).
- veto: a move is vetoed when the probability mass of worlds where it loses
  the king exceeds `threshold`. threshold = 0 reproduces the proposal's binary
  "loses in ANY world" rule. If every move is vetoed (the large-S failure mode),
  fall back to the move with the LEAST losing mass instead of having no answer.
"""
from __future__ import annotations

import numpy as np

LOSS_SCORE = -50_000.0  # anything at or below this counts as "loses the king"


def aggregate(values, weights=None, lam=0.0):
    """values: (n_moves, n_worlds) array -> (n_moves,) scores."""
    v = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.full(v.shape[1], 1.0 / v.shape[1])
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    mean = v @ weights
    worst = v.min(axis=1)
    return (1.0 - lam) * mean + lam * worst


def losing_mass(values, weights=None):
    v = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.full(v.shape[1], 1.0 / v.shape[1])
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return (v <= LOSS_SCORE) @ weights


def choose(moves, values, weights=None, lam=0.0, veto_threshold=0.0):
    """Returns (move, diagnostics). Never returns None if `moves` is nonempty.

    veto_threshold: max tolerated probability mass of king-losing worlds.
    """
    scores = aggregate(values, weights, lam)
    mass = losing_mass(values, weights)
    allowed = mass <= veto_threshold
    diagnostics = {'scores': scores, 'losing_mass': mass,
                   'vetoed': int((~allowed).sum()), 'all_vetoed': not allowed.any()}
    if allowed.any():
        candidates = np.where(allowed)[0]
        best = candidates[np.argmax(scores[candidates])]
    else:
        # Every move loses somewhere: least-bad fallback by losing mass,
        # breaking ties by aggregate score.
        least = mass.min()
        candidates = np.where(mass <= least + 1e-12)[0]
        best = candidates[np.argmax(scores[candidates])]
    return moves[best], diagnostics
