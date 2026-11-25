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


def naive_last_seen(ctx, evaluate):
    """Pieces stay where last seen. Never-seen squares keep their starting
    piece (everyone knows the initial setup) if it belongs to the opponent.
    Stale duplicates are possible; that's the naivety being measured."""
    squares = []
    for square in range(64):
        symbol = ctx.observation[square]
        if symbol == '?':
            last = ctx.tracker.last_seen[square]
            if last < 12:
                symbol = PIECES[last]
            elif last == UNKNOWN:
                start = STARTING[square]
                opponent_piece = start != '-' and start.islower() == bool(ctx.side)
                symbol = start if opponent_piece else '-'
            else:
                symbol = '-'  # seen empty at some point, nothing since
        squares.append(symbol)
    board = board_from_squares(squares, ctx.side)
    if board.king(ctx.side) is None or board.king(not ctx.side) is None:
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
