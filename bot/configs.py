"""Method configurations A/N/C/B/D — the mode registry (boardeval/NOTES.md §0)
made executable. Every switch is a named config field, never a hidden branch.

The five methods share one planner and differ only in two seams:
  - belief:     which boards the reply predictor scores the opponent's replies on
                ('true' = A, 'naive' = N, 'diffusion' = C)
  - own_worlds: where OUR candidate worlds come from ('tracker' = A/N/C exact
                enumeration; 'diffusion' = B/D — not yet wired, see belief_sources)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class PlannerConfig:
    method: str = 'A'
    # Seams (the A/N/C/B/D axes)
    belief: str = 'true'            # 'true' | 'naive' | 'diffusion'
    own_worlds: str = 'tracker'     # 'tracker' | 'diffusion' (B/D, later)
    # Dials (proposal defaults)
    K: int = 8                      # worlds analyzed deeply
    M: int = 5                      # candidate moves analyzed deeply (None in a round = all)
    J: int = 2                      # believed boards per opponent view
    T: int = 4                      # diffusion denoising steps
    tau: float = 150.0              # reply softmax temperature (cp scale)
    beta: float = 0.0               # clairvoyant-reply mix; 1.0 = method A's opponent
    floor: float = 0.02             # reply-distribution smoothing floor
    lam: float = 0.0                # danger tilt in aggregation
    veto_delta: float = 0.0         # tolerated king-losing mass over the sample
    epsilon: float = 0.25           # sampling epsilon-mix (weighting)
    epsilon_veto: float = 0.5       # conservative veto epsilon (full-S seam)
    selection_mode: str = 'weighted-sample'  # | 'top-k' | 'uniform'
    eval_mode: str = 'handcrafted'  # | 'sf-hybrid' | 'sf-full' (gates in NOTES §1)
    # Anytime widening schedule: (k, m) per round; m=None means every legal move.
    widening: tuple = ((8, 5), (8, None))
    fallback_worlds: int = 2        # uniform worlds for the always-ready PIMC floor


METHODS = {
    # A: exact own worlds, clairvoyant opponent — the PIMC baseline (M2 target).
    # beta=1.0 IS the clairvoyance: the reply distribution collapses onto their
    # true-board best reply, so the belief source is never consulted.
    'A': PlannerConfig(method='A', belief='true', beta=1.0),
    # N: opponent believes our pieces stand where THEY last saw them.
    # Needs the per-world opponent view from P2's tracker (Nov 30+).
    'N': PlannerConfig(method='N', belief='naive', beta=0.0),
    # C: the thesis — diffusion samples of the opponent's belief (P1's sampler,
    # Dec 1 handoff).
    'C': PlannerConfig(method='C', belief='diffusion', beta=0.0),
    # B/D: own worlds from diffusion — declared so the registry is complete;
    # belief_sources raises NotImplementedError until own_worlds exists.
    'B': PlannerConfig(method='B', belief='true', own_worlds='diffusion', beta=1.0),
    'D': PlannerConfig(method='D', belief='diffusion', own_worlds='diffusion'),
}


def for_method(name: str, **overrides) -> PlannerConfig:
    """METHODS lookup with dial overrides, e.g. for_method('A', K=4)."""
    return replace(METHODS[name.upper()], **overrides)
