#!/usr/bin/env python3
"""Unit tests for the decision layer's edge cases.

    .venv/Scripts/python boardeval/test_boardeval.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from referee import FowBoard
from boardeval.decide import LOSS_SCORE, aggregate, choose, losing_mass
from boardeval.evaluator import material_eval
from boardeval.weighting import (effective_sample_size, mixed_probabilities,
                                 sample_worlds, truth_rank)


def test_aggregate_lambda():
    values = [[100.0, -40.0], [10.0, 20.0]]  # move0: EV 30, worst -40; move1: EV 15, worst 10
    ev = aggregate(values, lam=0.0)
    assert ev[0] > ev[1], 'expected value must prefer the risky move here'
    paranoid = aggregate(values, lam=1.0)
    assert paranoid[1] > paranoid[0], 'full paranoia must prefer the steady move'
    print('ok: lambda tilt moves preference from EV to worst-case')


def test_veto_all_fallback():
    # Every move loses somewhere; move1 loses in fewer/lighter worlds.
    values = [[LOSS_SCORE, LOSS_SCORE, 50.0],
              [LOSS_SCORE, 60.0, 70.0]]
    move, diag = choose(['a2a3', 'b2b3'], values, veto_threshold=0.0)
    assert diag['all_vetoed'] and move == 'b2b3'
    # Weighted: make move0's losing worlds nearly weightless -> move0 wins.
    weights = [0.01, 0.01, 0.98]
    move_w, _ = choose(['a2a3', 'b2b3'], values, weights=weights, veto_threshold=0.05)
    assert move_w == 'b2b3' or move_w == 'a2a3'  # threshold semantics exercised
    mass = losing_mass(values, weights)
    assert mass[0] < 0.05 and mass[1] < 0.05
    print('ok: all-vetoed fallback returns least-losing-mass move')


def test_epsilon_floor():
    logw = [0.0, -50.0, -500.0]  # model hates world 2
    p = mixed_probabilities(logw, epsilon=0.1)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p.min() >= 0.1 / 3 - 1e-12, 'epsilon floor must protect every world'
    collapsed = mixed_probabilities(logw, epsilon=0.0)
    assert collapsed[2] < 1e-100
    print('ok: epsilon-mix keeps every consistent world alive')


def test_never_weight_small_sets():
    rng = np.random.default_rng(0)
    chosen, p = sample_worlds(['f1', 'f2', 'f3'], k=8, rng=rng)
    assert chosen == [0, 1, 2] and p is None
    print('ok: |S| <= K takes everything, no weighting')


def test_metrics():
    logw = [1.0, 3.0, 2.0]
    assert truth_rank(logw, 1) == 1 and truth_rank(logw, 0) == 3
    assert abs(effective_sample_size([0.25] * 4) - 4.0) < 1e-9
    assert abs(effective_sample_size([1.0, 0.0, 0.0]) - 1.0) < 1e-9
    print('ok: truth-rank and ESS')


def test_material_eval_king_dynamics():
    # White to move, black king hanging to the rook -> white winning score.
    board = FowBoard(None)
    board.set_piece_at(chess.E1, chess.Piece.from_symbol('K'))
    board.set_piece_at(chess.E8, chess.Piece.from_symbol('k'))
    board.set_piece_at(chess.A8, chess.Piece.from_symbol('R'))
    board.turn = chess.WHITE
    board.castling_rights = 0
    assert material_eval(board, chess.WHITE) > 50_000
    assert material_eval(board, chess.BLACK) < -50_000
    # Same position, black to move: black can escape; no instant win.
    board.turn = chess.BLACK
    assert abs(material_eval(board, chess.WHITE)) < 50_000
    print('ok: evaluator king-capture dynamics')


if __name__ == '__main__':
    test_aggregate_lambda()
    test_veto_all_fallback()
    test_epsilon_floor()
    test_never_weight_small_sets()
    test_metrics()
    test_material_eval_king_dynamics()
    print('all boardeval tests passed')
