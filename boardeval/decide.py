"""Aggregation across worlds + the king-safety veto -> final move choice.

Input: V[move][world] — the evaluated outcome of each candidate move in each
sampled world — plus world weights (pass sample_worlds' agg_weights: that is
the renormalized-within-sample weighted mean of NOTES.md §3). Output: one move.

- aggregate: per-move score = (1 - lam) * weighted_mean + lam * worst_case.
  lam = 0 is expected value; lam -> 1 is paranoid (the proposal's danger tilt).
- veto (sample-level): a move is vetoed when the probability mass of sampled
  worlds where it loses the king exceeds `veto_threshold`. threshold = 0
  reproduces the proposal's binary "loses in ANY world" rule. If every move is
  vetoed (the large-S failure mode), fall back to the LEAST losing mass.
- veto_mass_full_s (set-level seam): the real delta veto runs over ALL of S,
  not the K sample — danger flags come from the tracker/rules, weights from
  the belief model with the conservative epsilon_veto mix. Plugs in when
  Christine's tracker lands.

Scale note: LOSS_SCORE matches the centipawn evaluators (+-WIN = 100k). The
normalized tanh pipeline passes loss_threshold=-0.999 instead — thresholds are
parameters, not constants baked into the logic.
"""
from __future__ import annotations

import numpy as np

LOSS_SCORE = -50_000.0  # cp scale: anything at or below counts as "loses the king"
LOSS_SCORE_TANH = -0.999  # same rule on the tanh_eval [-1, 1] scale


def _normalized_weights(values, weights):
    if weights is None:
        return np.full(values.shape[1], 1.0 / values.shape[1])
    weights = np.asarray(weights, dtype=np.float64)
    return weights / weights.sum()


def aggregate(values, weights=None, lam=0.0):
    """values: (n_moves, n_worlds) array -> (n_moves,) scores."""
    v = np.asarray(values, dtype=np.float64)
    weights = _normalized_weights(v, weights)
    mean = v @ weights
    worst = v.min(axis=1)
    return (1.0 - lam) * mean + lam * worst


def losing_mass(values, weights=None, loss_threshold=LOSS_SCORE):
    v = np.asarray(values, dtype=np.float64)
    weights = _normalized_weights(v, weights)
    return (v <= loss_threshold) @ weights


def veto_mass_full_s(danger_flags, log_weights=None, epsilon_veto=0.5):
    """Epsilon-mixed king-danger mass over ALL of S for one candidate move.

    danger_flags: per-world booleans (world lets them capture our king after
    this move) — exact, from the tracker/rules. log_weights: belief-model
    scores for the same worlds; None -> pure count veto (uniform mass).

    Blend identity (NOTES.md §3): with model weights w_hat,
        mass = (1 - eps) * w_hat(danger) + eps * count(danger) / |S|
    so the model can never suppress the danger mass below eps * count-share.
    Compare against delta (default 1% of S) at the call site.
    """
    danger = np.asarray(danger_flags, dtype=bool)
    if log_weights is None:
        return float(danger.mean())
    from boardeval.weighting import veto_weights  # lazy: keeps decide import-light
    return float(danger @ veto_weights(log_weights, epsilon_veto))


def choose(moves, values, weights=None, lam=0.0, veto_threshold=0.0,
           loss_threshold=LOSS_SCORE):
    """Returns (move, diagnostics). Never returns None if `moves` is nonempty.

    veto_threshold: max tolerated probability mass of king-losing worlds.
    loss_threshold: the "loses the king" score floor (LOSS_SCORE for cp
    evaluators, LOSS_SCORE_TANH for tanh_eval).
    """
    scores = aggregate(values, weights, lam)
    mass = losing_mass(values, weights, loss_threshold)
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
