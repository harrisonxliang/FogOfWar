"""Opponent-move prediction ("reply rule"): the X1 contestants.

Every model implements predict(ctx) -> list of (uci, score) ranked best-first,
predicting the move the player to move (the "mover") will choose. They differ
only in which board(s) they evaluate on:

- clairvoyant: the true board (assumes the mover sees everything)
- naive: pieces stand where the mover last saw them
- like_me: J boards sampled from the belief model given the mover's view

All share one decision rule — 1-ply lookahead scored by the evaluator — so X1
differences isolate the *beliefs*, not the decision rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from referee import FowBoard, fow_moves
from boardgen.encoding import UNKNOWN, PIECES, HistoryTracker
from boardgen.sample import BeliefState, generate


@dataclass
class MoverContext:
    """Everything a model may condition on, for one ply."""
    side: bool                       # mover color: True = white
    true_board: FowBoard             # full truth (clairvoyant only!)
    observation: str                 # mover's current 64-char view
    tracker: HistoryTracker          # mover's last-seen/since memory (raw squares)
    belief_state: object = None      # boardgen BeliefState for the mover
    ply: int = 0


def board_from_squares(squares, side):
    """64-length iterable of piece symbols/None -> FowBoard with `side` to move.

    Castling rights and en passant are cleared (unknown to a reconstruction).
    """
    board = FowBoard(None)
    board.turn = side
    for square, symbol in enumerate(squares):
        if symbol and symbol not in '-?':
            board.set_piece_at(square, chess.Piece.from_symbol(symbol))
    board.castling_rights = 0
    return board


def rank_moves(board, evaluate, side):
    """1-ply lookahead on one board: rank every FoW-legal move by eval-after."""
    ranked = []
    for move in fow_moves(board):
        board.push(move)
        ranked.append((move.uci(), evaluate(board, side)))
        board.pop()
    ranked.sort(key=lambda pair: -pair[1])
    return ranked


def clairvoyant(ctx, evaluate):
    return rank_moves(ctx.true_board, evaluate, ctx.side)


STARTING = 'RNBQKBNRPPPPPPPP' + 32 * '-' + 'pppppppp' + 'rnbqkbnr'  # a1..h8


def last_seen_board(observation, tracker, side):
    """Reconstruct a viewer's last-seen world: pieces stand where last seen,
    never-seen squares keep their starting piece (everyone knows the initial
    setup) if it belongs to the opponent. Stale duplicates are possible.

    Returns a FowBoard with `side` to move, or None when the reconstruction is
    degenerate (a king missing). Shared by the naive X1 contestant and the
    bot's stub single-world tracker (bot/stub_tracker.py).
    """
    squares = []
    for square in range(64):
        symbol = observation[square]
        if symbol == '?':
            last = tracker.last_seen[square]
            if last < 12:
                symbol = PIECES[last]
            elif last == UNKNOWN:
                start = STARTING[square]
                opponent_piece = start != '-' and start.islower() == bool(side)
                symbol = start if opponent_piece else '-'
            else:
                symbol = '-'  # seen empty at some point, nothing since
        squares.append(symbol)
    board = board_from_squares(squares, side)
    if board.king(side) is None or board.king(not side) is None:
        return None
    return board


def naive_last_seen(ctx, evaluate):
    """Pieces stay where last seen — the naivety being measured."""
    board = last_seen_board(ctx.observation, ctx.tracker, ctx.side)
    if board is None:
        return []  # degenerate reconstruction; let the harness skip it
    return rank_moves(board, evaluate, ctx.side)


def like_me(ctx, evaluate, predictor, j=4, steps=4, rng=None):
    """Sample J believed boards from the belief model; rank moves by average
    eval over the boards where they are legal."""
    fens, _ = generate(predictor, ctx.belief_state, n=j, steps=steps, rng=rng)
    totals, counts = {}, {}
    for fen in fens:
        board = FowBoard(fen + (' w ' if ctx.side else ' b ') + '- - 0 1')
        board.turn = ctx.side
        if board.king(ctx.side) is None or board.king(not ctx.side) is None:
            continue
        for uci, score in rank_moves(board, evaluate, ctx.side):
            totals[uci] = totals.get(uci, 0.0) + score
            counts[uci] = counts.get(uci, 0) + 1
    ranked = [(uci, totals[uci] / counts[uci]) for uci in totals]
    ranked.sort(key=lambda pair: -pair[1])
    return ranked


def uniform_random(ctx, evaluate):
    """Level-0 floor: every legal move on the true board, no preference.

    Scored as chance: the harness treats equal scores as a uniform pick.
    """
    return [(move.uci(), 0.0) for move in fow_moves(ctx.true_board)]


# ---------------------------------------------------------------------------
# Ranked scores -> probability distribution (NOTES.md §4).
# The invariant: predict their reply on what they BELIEVE; the softmax/floor
# below is how ranked believed-board scores become p(r) for X1 log-likelihood
# and for the planner's expected value.
# ---------------------------------------------------------------------------

def to_distribution(ranked, legal_ucis, tau=150.0, floor=0.02):
    """Ranked (uci, score) pairs -> p(r) over the LEGAL move set.

    tau: softmax temperature on the evaluator's cp scale (fit per team offline
    for X1; play time needs a pooled default — protocol may hide identity).
    floor: uniform smoothing mass mixed in so every legal move keeps at least
    floor/len(legal) — log-likelihood can never hit -inf, and moves missing
    from `ranked` (e.g. illegal on every believed board) stay possible.
    """
    legal = list(legal_ucis)
    if not legal:
        return {}
    uniform = 1.0 / len(legal)
    legal_set = set(legal)
    scores = {uci: s for uci, s in ranked if uci in legal_set}
    soft = {}
    if scores:
        tau = max(float(tau), 1e-6)
        top = max(scores.values())
        expo = {uci: np.exp((s - top) / tau) for uci, s in scores.items()}
        total = sum(expo.values())
        soft = {uci: e / total for uci, e in expo.items()}
    else:
        soft = {uci: uniform for uci in legal}
    return {uci: (1.0 - floor) * soft.get(uci, 0.0) + floor * uniform
            for uci in legal}


def mix_with_clairvoyant(p, clairvoyant_ranked, beta=0.0):
    """Proposal step 5d: with weight beta, the opponent plays their clairvoyant
    best reply instead of the belief-based distribution.

    beta = 0 -> pure belief model; beta = 1 -> method A's opponent. Accepts the
    clairvoyant contestant's ranked list; its top entry is the best reply.
    """
    if beta <= 0.0 or not clairvoyant_ranked:
        return dict(p)
    best = clairvoyant_ranked[0][0]
    mixed = {uci: (1.0 - beta) * prob for uci, prob in p.items()}
    mixed[best] = mixed.get(best, 0.0) + beta
    return mixed
