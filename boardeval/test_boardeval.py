#!/usr/bin/env python3
"""Unit tests for the decision layer's edge cases (NOTES.md §6 test plan).

    .venv/Scripts/python boardeval/test_boardeval.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from referee import FowBoard
import boardeval.weighting as weighting
from boardeval.decide import (LOSS_SCORE, LOSS_SCORE_TANH, aggregate, choose,
                              losing_mass, veto_mass_full_s)
from boardeval.evaluator import (WIN, fog_terms, material_eval, tanh_eval,
                                 terminal_score)
from boardeval.reply import mix_with_clairvoyant, to_distribution
from boardeval.weighting import (effective_sample_size, mixed_probabilities,
                                 sample_worlds, truth_rank, veto_weights)


# ---------------------------------------------------------------------------
# decide.py
# ---------------------------------------------------------------------------

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
    weights = [0.01, 0.01, 0.98]
    mass = losing_mass(values, weights)
    assert mass[0] < 0.05 and mass[1] < 0.05
    print('ok: all-vetoed fallback returns least-losing-mass move')


def test_loss_threshold_tanh_scale():
    # Same veto logic on the tanh [-1, 1] scale via loss_threshold.
    values = [[-1.0, 0.5], [0.1, 0.2]]
    move, diag = choose(['risky', 'steady'], values, veto_threshold=0.0,
                        loss_threshold=LOSS_SCORE_TANH)
    assert move == 'steady' and diag['vetoed'] == 1
    print('ok: loss_threshold parameterizes the veto for normalized evals')


def test_veto_mass_full_s_blend():
    logw = [0.0, -1.0, -2.0, -3.0]
    danger = [True, False, False, True]
    eps = 0.5
    w_hat = np.exp(logw - np.max(logw))
    w_hat /= w_hat.sum()
    expected = (1 - eps) * (w_hat[0] + w_hat[3]) + eps * (2 / 4)
    got = veto_mass_full_s(danger, logw, epsilon_veto=eps)
    assert abs(got - expected) < 1e-12, 'blend identity: (1-e)*model + e*count-share'
    assert abs(veto_mass_full_s(danger) - 0.5) < 1e-12, 'no weights -> count veto'
    # Floor property: even if the model hates the danger worlds, mass >= e * share.
    hostile = veto_mass_full_s(danger, [-100.0, 0.0, 0.0, -100.0], epsilon_veto=eps)
    assert hostile >= eps * 0.5 - 1e-12
    print('ok: full-S veto mass blend + floor')


# ---------------------------------------------------------------------------
# weighting.py
# ---------------------------------------------------------------------------

def test_epsilon_floor():
    logw = [0.0, -50.0, -500.0]  # model hates world 2
    p = mixed_probabilities(logw, epsilon=0.1)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p.min() >= 0.1 / 3 - 1e-12, 'epsilon floor must protect every world'
    collapsed = mixed_probabilities(logw, epsilon=0.0)
    assert collapsed[2] < 1e-100
    # veto_weights is the same mix with its own (conservative) epsilon.
    assert np.allclose(veto_weights(logw, 0.5), mixed_probabilities(logw, 0.5))
    print('ok: epsilon-mix keeps every consistent world alive')


def test_never_weight_small_sets():
    rng = np.random.default_rng(0)
    chosen, agg, diag = sample_worlds(['f1', 'f2', 'f3'], k=8, rng=rng)
    assert chosen == [0, 1, 2]
    assert np.allclose(agg, [1 / 3] * 3), 'no model -> uniform aggregation weights'
    assert diag['captured_mass'] is None
    print('ok: |S| <= K takes everything')


def test_selection_modes_and_renormalized_weights():
    # Patch the belief-model scorer so no model/torch state is needed.
    original = weighting.score
    weighting.score = lambda predictor, state, fens: [0.0, -1.0, -2.0, -3.0, -4.0]
    try:
        rng = np.random.default_rng(1)
        fens = ['f0', 'f1', 'f2', 'f3', 'f4']
        # top-k: deterministic truncation by w~, renormalized unequal weights.
        idx, agg, diag = sample_worlds(fens, k=3, rng=rng, predictor=object(),
                                       epsilon=0.25, mode='top-k')
        assert idx == [0, 1, 2]
        assert abs(agg.sum() - 1.0) < 1e-9
        assert agg[0] > agg[2], 'renormalized-within-sample weights, not uniform'
        assert abs(diag['captured_mass'] - diag['probabilities'][idx].sum()) < 1e-12
        assert 0.0 < diag['captured_mass'] <= 1.0
        # epsilon = 1 -> uniform w~ -> equal aggregation weights (plain-mean edge).
        _, agg_eps1, _ = sample_worlds(fens, k=3, rng=rng, predictor=object(),
                                       epsilon=1.0, mode='top-k')
        assert np.allclose(agg_eps1, 1 / 3)
        # weighted-sample: valid draw, diagnostics consistent.
        idx_w, agg_w, diag_w = sample_worlds(fens, k=3, rng=rng, predictor=object(),
                                             epsilon=0.25, mode='weighted-sample')
        assert len(idx_w) == len(set(idx_w)) == 3
        assert abs(agg_w.sum() - 1.0) < 1e-9
        assert 0.0 < diag_w['captured_mass'] <= 1.0
        # danger world forced in, entering with its own w~.
        idx_d, agg_d, _ = sample_worlds(fens, k=3, rng=rng, predictor=object(),
                                        epsilon=0.25, mode='top-k', danger_index=4)
        assert 4 in idx_d and len(idx_d) == 3
        assert abs(agg_d.sum() - 1.0) < 1e-9
    finally:
        weighting.score = original
    print('ok: selection modes, renormalized weights, captured mass, danger include')


def test_metrics():
    logw = [1.0, 3.0, 2.0]
    assert truth_rank(logw, 1) == 1 and truth_rank(logw, 0) == 3
    assert abs(effective_sample_size([0.25] * 4) - 4.0) < 1e-9
    assert abs(effective_sample_size([1.0, 0.0, 0.0]) - 1.0) < 1e-9
    print('ok: truth-rank and ESS')


# ---------------------------------------------------------------------------
# reply.py distribution layer
# ---------------------------------------------------------------------------

def test_reply_distribution():
    legal = ['a1a2', 'b1b2', 'c1c2']
    ranked = [('a1a2', 100.0), ('b1b2', 0.0)]
    p = to_distribution(ranked, legal, tau=150.0, floor=0.03)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert abs(p['c1c2'] - 0.01) < 1e-12, 'unranked legal move gets exactly floor/len'
    assert p['a1a2'] > p['b1b2'] > p['c1c2']
    # tau -> 0: argmax takes all non-floor mass.
    sharp = to_distribution(ranked, legal, tau=1e-9, floor=0.03)
    assert abs(sharp['a1a2'] - (0.97 + 0.01)) < 1e-9
    # tau large: ranked moves approach equality.
    flat = to_distribution(ranked, legal, tau=1e9, floor=0.0)
    assert abs(flat['a1a2'] - flat['b1b2']) < 1e-6
    # Empty ranking -> uniform over legal; LL can never be -inf.
    empty = to_distribution([], legal, floor=0.02)
    assert np.allclose(list(empty.values()), 1 / 3)
    print('ok: reply distribution (softmax, floor, tau limits)')


def test_beta_mix():
    p = {'a1a2': 0.6, 'b1b2': 0.4}
    clair = [('b1b2', 5.0), ('a1a2', 1.0)]
    assert mix_with_clairvoyant(p, clair, beta=0.0) == p
    full = mix_with_clairvoyant(p, clair, beta=1.0)
    assert abs(full['b1b2'] - 1.0) < 1e-12 and abs(full['a1a2']) < 1e-12
    half = mix_with_clairvoyant(p, clair, beta=0.5)
    assert abs(half['b1b2'] - (0.2 + 0.5)) < 1e-12
    assert abs(sum(half.values()) - 1.0) < 1e-12
    print('ok: beta mix (beta=1 reproduces clairvoyant)')


# ---------------------------------------------------------------------------
# evaluator.py
# ---------------------------------------------------------------------------

def test_material_eval_king_dynamics():
    board = FowBoard(None)
    board.set_piece_at(chess.E1, chess.Piece.from_symbol('K'))
    board.set_piece_at(chess.E8, chess.Piece.from_symbol('k'))
    board.set_piece_at(chess.A8, chess.Piece.from_symbol('R'))
    board.turn = chess.WHITE
    board.castling_rights = 0
    assert material_eval(board, chess.WHITE) > 50_000
    assert material_eval(board, chess.BLACK) < -50_000
    board.turn = chess.BLACK
    assert abs(material_eval(board, chess.WHITE)) < 50_000
    print('ok: evaluator king-capture dynamics')


def test_terminal_precedence_and_tanh():
    # White is up a queen AND white's king hangs with black to move:
    # the terminal check must win over material, and tanh_eval must say -1.
    board = FowBoard(None)
    board.set_piece_at(chess.E1, chess.Piece.from_symbol('K'))
    board.set_piece_at(chess.D8, chess.Piece.from_symbol('Q'))
    board.set_piece_at(chess.H8, chess.Piece.from_symbol('k'))
    board.set_piece_at(chess.A1, chess.Piece.from_symbol('r'))
    board.turn = chess.BLACK
    board.castling_rights = 0
    assert terminal_score(board, chess.WHITE) == -WIN
    assert material_eval(board, chess.WHITE) == -WIN
    assert tanh_eval(board, chess.WHITE) == -1.0
    assert tanh_eval(board, chess.BLACK) == 1.0
    # Quiet position: tanh output strictly inside (-1, 1), scale monotone.
    quiet = FowBoard(None)
    quiet.set_piece_at(chess.G1, chess.Piece.from_symbol('K'))
    quiet.set_piece_at(chess.G8, chess.Piece.from_symbol('k'))
    quiet.set_piece_at(chess.A2, chess.Piece.from_symbol('P'))
    quiet.turn = chess.WHITE
    quiet.castling_rights = 0
    value = tanh_eval(quiet, chess.WHITE, fog=False)
    assert -1.0 < value < 1.0 and value > 0
    assert tanh_eval(quiet, chess.WHITE, scale=300, fog=False) > value, \
        'smaller scale saturates faster'
    print('ok: terminal precedence + tanh bounds')


def test_eval_mirror_symmetry():
    # Color-mirror (flip vertical, swap colors, flip turn): eval must negate —
    # equivalently eval(mirror, BLACK) == eval(original, WHITE). Fog terms ON,
    # since they are the newest and most likely accidentally-asymmetric code.
    board = FowBoard(None)
    for square, symbol in [(chess.G1, 'K'), (chess.F2, 'P'), (chess.G2, 'P'),
                           (chess.H2, 'P'), (chess.C3, 'N'), (chess.D1, 'Q'),
                           (chess.E8, 'k'), (chess.B8, 'r'), (chess.A7, 'p')]:
        board.set_piece_at(square, chess.Piece.from_symbol(symbol))
    board.turn = chess.WHITE
    board.castling_rights = 0
    mirrored = board.mirror()
    assert abs(material_eval(board, chess.WHITE, fog=True)
               - material_eval(mirrored, chess.BLACK, fog=True)) < 1e-9
    assert abs(material_eval(board, chess.BLACK, fog=True)
               - material_eval(mirrored, chess.WHITE, fog=True)) < 1e-9
    # Same-board symmetry: one side's gain is the other's loss.
    assert abs(material_eval(board, chess.WHITE, fog=True)
               + material_eval(board, chess.BLACK, fog=True)) < 1e-9
    print('ok: eval mirror + same-board symmetry (fog terms active)')


def test_fog_terms_hidden_attacker():
    # Black rook on a8 attacks white king a1 from OUTSIDE white's vision
    # (white sees only a2/b1/b2 + pawn squares): hidden attacker -40. Black's
    # rook also sees the white king, so white has 1 seen key piece: -8.
    board = FowBoard(None)
    board.set_piece_at(chess.A1, chess.Piece.from_symbol('K'))
    board.set_piece_at(chess.H2, chess.Piece.from_symbol('P'))
    board.set_piece_at(chess.A8, chess.Piece.from_symbol('r'))
    board.set_piece_at(chess.H8, chess.Piece.from_symbol('k'))
    board.turn = chess.WHITE
    board.castling_rights = 0
    assert terminal_score(board, chess.WHITE) is None
    assert abs(fog_terms(board, chess.WHITE) - (-48.0)) < 1e-9
    assert material_eval(board, chess.WHITE, fog=True) \
        < material_eval(board, chess.WHITE, fog=False)
    # Make the rook visible (adjacent to the king): no HIDDEN attacker left.
    board.remove_piece_at(chess.A8)
    board.set_piece_at(chess.A2, chess.Piece.from_symbol('r'))
    visible_fog = fog_terms(board, chess.WHITE)
    assert visible_fog > -48.0, 'visible attacker must not pay the hidden penalty'
    print('ok: fog terms (hidden-attacker penalty toggles with visibility)')


if __name__ == '__main__':
    test_aggregate_lambda()
    test_veto_all_fallback()
    test_loss_threshold_tanh_scale()
    test_veto_mass_full_s_blend()
    test_epsilon_floor()
    test_never_weight_small_sets()
    test_selection_modes_and_renormalized_weights()
    test_metrics()
    test_reply_distribution()
    test_beta_mix()
    test_material_eval_king_dynamics()
    test_terminal_precedence_and_tanh()
    test_eval_mirror_symmetry()
    test_fog_terms_hidden_attacker()
    print('all boardeval tests passed')
