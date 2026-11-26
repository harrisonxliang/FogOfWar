"""Belief sources — the boards_for_reply seam from the team interface contract.

boards_for_reply(x2, mover_side, j, rng) -> list of full boards the OPPONENT
(the player to move on x2) might believe they are playing on. Their own pieces
are always correct (players know their own material); the sources differ in
where they place OUR pieces. Swapping this object is exactly what separates
methods A, N and C — the planner never knows which one it holds.

Status (matches the team milestones):
  TrueBoardSource   — live. Method A; with beta=1 the planner's fast path means
                      it is rarely even consulted.
  NaiveBoardSource  — interface ready; needs the per-world opponent view
                      (last-seen map) from P2's tracker (Nov 30–Dec 2).
  DiffusionSource   — interface ready; needs P1's sampler + per-world opponent
                      BeliefState (Dec 1 handoff).
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TrueBoardSource:
    """The opponent believes the truth of this world (clairvoyant / method A)."""

    def boards_for_reply(self, x2, mover_side, j, rng):
        return [x2]


class NaiveBoardSource:
    """The opponent believes our pieces stand where they last saw them (method N).

    Requires each world to carry the opponent's view (observation + last-seen
    history) — that is P2's opp_view_after deliverable. Until it lands, using
    this source is an explicit error, not a silent approximation.
    """

    def __init__(self, opp_view_provider=None):
        self.opp_view_provider = opp_view_provider

    def boards_for_reply(self, x2, mover_side, j, rng):
        if self.opp_view_provider is None:
            raise NotImplementedError(
                "NaiveBoardSource needs P2's per-world opponent views "
                '(tracker opp_view_after, Nov 30-Dec 2). Method N runs in the '
                'X1 harness today (replays carry both views); live play waits '
                'for the tracker.')
        view, history = self.opp_view_provider(x2)
        from boardeval.reply import last_seen_board
        board = last_seen_board(view, history, not mover_side)
        return [board] if board is not None else [x2]


class DiffusionSource:
    """J diffusion-sampled boards of what the opponent believes (method C).

    Requires P1's predictor plus the opponent's BeliefState for the world —
    the Dec 1 sampler handoff. Explicit error until wired.
    """

    def __init__(self, predictor=None, belief_state_provider=None, steps=4):
        self.predictor = predictor
        self.belief_state_provider = belief_state_provider
        self.steps = steps

    def boards_for_reply(self, x2, mover_side, j, rng):
        if self.predictor is None or self.belief_state_provider is None:
            raise NotImplementedError(
                "DiffusionSource needs P1's sampler and per-world opponent "
                'BeliefState (Dec 1 integration gate). Method C runs in the '
                'X1 harness today; live play waits for the handoff.')
        from boardgen.sample import generate
        from boardeval.reply import board_from_squares  # noqa: F401 (fen path below)
        from referee import FowBoard
        state = self.belief_state_provider(x2)
        fens, _ = generate(self.predictor, state, n=j, steps=self.steps, rng=rng)
        boards = []
        for fen in fens:
            board = FowBoard(fen + (' b ' if mover_side else ' w ') + '- - 0 1')
            board.turn = not mover_side
            if board.king(True) is not None and board.king(False) is not None:
                boards.append(board)
        return boards or [x2]


def get_belief_source(cfg, predictor=None, opp_view_provider=None,
                      belief_state_provider=None):
    """Config -> source object (the registry rule: named modes, no branches)."""
    if cfg.belief == 'true':
        return TrueBoardSource()
    if cfg.belief == 'naive':
        return NaiveBoardSource(opp_view_provider)
    if cfg.belief == 'diffusion':
        return DiffusionSource(predictor, belief_state_provider, steps=cfg.T)
    raise ValueError(f'unknown belief source: {cfg.belief!r}')
