#!/usr/bin/env python3
"""Planner unit tests (boardeval/NOTES.md §6, planner section).

    .venv/Scripts/python bot/test_planner.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from referee import FowBoard, fow_moves
from boardeval.evaluator import WIN, material_eval
from bot.belief_sources import (DiffusionSource, NaiveBoardSource,
                                TrueBoardSource, get_belief_source)
from bot.configs import METHODS, for_method
from bot.planner import plan_move, pimc_fallback, score_pair
from bot.shortlist import build_shortlist, eval_after_best_reply


def start_world():
    return FowBoard()  # standard starting position, white to move


def test_configs_registry():
    assert set(METHODS) == {'A', 'N', 'C', 'B', 'D'}
    assert METHODS['A'].beta == 1.0 and METHODS['A'].belief == 'true'
    assert for_method('a', K=4).K == 4
    # Unwired seams fail loudly, never silently approximate.
    for unwired in (NaiveBoardSource(), DiffusionSource()):
        try:
            unwired.boards_for_reply(start_world(), True, 2, None)
            raise AssertionError('expected NotImplementedError')
        except NotImplementedError:
            pass
    for method in ('B', 'D'):
        try:
            plan_move([start_world()], list(fow_moves(start_world())), True,
                      METHODS[method], 1.0, np.random.default_rng(0))
            raise AssertionError('expected NotImplementedError')
        except NotImplementedError:
            pass
    print('ok: config registry + unwired seams raise')


def test_shortlist_structure():
    world = start_world()
    legal = list(fow_moves(world))
    moves, diag = build_shortlist(world, chess.WHITE, legal, material_eval, m=4)
    assert len(moves) == len(set(moves)), 'deduplicated'
    assert all(mv in legal for mv in moves)
    best_cheap = max(diag['cheap'], key=lambda mv: diag['cheap'][mv])
    assert best_cheap in moves, 'top cheap-scored move always survives'
    assert diag['slots']['scouting'] is not None, 'opening moves reveal squares'
    assert len(moves) <= 4 + 3 + 4  # m + three single slots + capture slots
    # m >= all legal moves -> everything passes through.
    all_moves, _ = build_shortlist(world, chess.WHITE, legal, material_eval, m=99)
    assert set(all_moves) == set(legal)
    print('ok: shortlist structure (dedup, subset, top-cheap, scouting slot)')


def test_shortlist_capture_slot():
    # White pawn e4 can take the visible black pawn d5; captures are
    # force-included even with m=1.
    world = FowBoard(None)
    world.set_piece_at(chess.E1, chess.Piece.from_symbol('K'))
    world.set_piece_at(chess.E4, chess.Piece.from_symbol('P'))
    world.set_piece_at(chess.H8, chess.Piece.from_symbol('k'))
    world.set_piece_at(chess.D5, chess.Piece.from_symbol('p'))
    world.turn = chess.WHITE
    world.castling_rights = 0
    legal = list(fow_moves(world))
    capture = chess.Move.from_uci('e4d5')
    assert capture in legal
    moves, diag = build_shortlist(world, chess.WHITE, legal, material_eval, m=1)
    assert capture in moves, 'capture slot must survive any m'
    assert capture in diag['slots']['captures']
    print('ok: capture guaranteed slot')


def test_score_pair_terminal():
    # Qd7xe8 captures the black king: terminal +WIN, no reply machinery.
    world = FowBoard(None)
    world.set_piece_at(chess.E1, chess.Piece.from_symbol('K'))
    world.set_piece_at(chess.D7, chess.Piece.from_symbol('Q'))
    world.set_piece_at(chess.E8, chess.Piece.from_symbol('k'))
    world.turn = chess.WHITE
    world.castling_rights = 0
    cfg = METHODS['A']
    value = score_pair(world, chess.Move.from_uci('d7e8'), chess.WHITE, cfg,
                       TrueBoardSource(), material_eval, material_eval,
                       np.random.default_rng(0))
    assert value == WIN
    print('ok: score_pair terminal shortcut')


def test_score_pair_beta_paths_agree_on_true_source():
    # With the TrueBoard source, the belief-based path (beta=0, tiny tau)
    # should pick out the same best replies as the clairvoyant fast path
    # (beta=1); scores need not be identical (distribution vs min) but the
    # sign and ordering of a clearly-better vs clearly-worse move must agree.
    world = start_world()
    rng = np.random.default_rng(0)
    cfg_fast = for_method('A')                       # beta = 1 fast path
    cfg_dist = for_method('A', beta=0.0, tau=1.0, floor=0.0)  # near-argmax dist
    hang = chess.Move.from_uci('f2f3')
    develop = chess.Move.from_uci('e2e4')
    for cfg in (cfg_fast, cfg_dist):
        v_dev = score_pair(world, develop, chess.WHITE, cfg, TrueBoardSource(),
                           material_eval, material_eval, rng)
        v_hang = score_pair(world, hang, chess.WHITE, cfg, TrueBoardSource(),
                            material_eval, material_eval, rng)
        assert np.isfinite(v_dev) and np.isfinite(v_hang)
    print('ok: both beta paths produce finite scores on the true source')


def test_fallback_first_and_zero_budget():
    world = start_world()
    legal = list(fow_moves(world))
    uci, diag = plan_move([world], legal, chess.WHITE, METHODS['A'], 0.0,
                          np.random.default_rng(0))
    assert diag['completed_rounds'] == 0
    assert uci == diag['fallback']
    assert uci in {mv.uci() for mv in legal}
    print('ok: zero budget -> legal fallback move, no rounds')


def test_full_rounds_and_determinism():
    world = start_world()
    legal = list(fow_moves(world))
    cfg = METHODS['A']
    uci1, diag1 = plan_move([world], legal, chess.WHITE, cfg, 30.0,
                            np.random.default_rng(7))
    uci2, diag2 = plan_move([world], legal, chess.WHITE, cfg, 30.0,
                            np.random.default_rng(7))
    assert diag1['completed_rounds'] == len(cfg.widening)
    assert uci1 == uci2, 'same seed -> same move'
    assert uci1 in {mv.uci() for mv in legal}
    print('ok: all rounds complete; deterministic under a fixed seed')


def test_partial_round_discarded():
    # Injectable clock: each call advances 10s. budget=15 means the deadline
    # passes during the first round's scoring loop -> the partial round must
    # be discarded and the fallback returned (NOTES §5 incumbent rule).
    ticks = {'t': 0.0}

    def fake_now():
        ticks['t'] += 10.0
        return ticks['t']

    world = start_world()
    legal = list(fow_moves(world))
    uci, diag = plan_move([world], legal, chess.WHITE, METHODS['A'], 15.0,
                          np.random.default_rng(0), now=fake_now)
    assert diag['completed_rounds'] == 0
    assert uci == diag['fallback']
    print('ok: deadline mid-round -> partial round discarded, fallback stands')


def test_stub_world_always_has_both_kings():
    # Regression: kings hide all game, so the enemy king is often absent from
    # a last-seen reconstruction. A world with no enemy king evaluates +WIN
    # for every move -> blind veto -> the bot walked onto a remembered
    # bishop's diagonal (local game 001, Nov 26). The stub must place it.
    from bot.stub_tracker import StubTracker
    tracker = StubTracker(True)
    # White king visible on e1, e8 seen EMPTY (king moved away unseen),
    # everything else fog: the reconstruction has no black king.
    obs = ['?'] * 64
    obs[chess.E1] = 'K'
    obs[chess.E8] = '-'
    tracker.update(''.join(obs))
    world = tracker.worlds()[0]
    assert world.king(chess.BLACK) is not None, 'enemy king must be placed'
    placed = world.king(chess.BLACK)
    assert tracker.current[placed] == '?', 'placed on an unseen square'
    assert not world.attackers(chess.WHITE, placed), 'not instantly capturable'
    assert world.king(chess.WHITE) == chess.E1
    print('ok: stub world always contains both kings')


def test_fallback_veto():
    # Two legal moves; one walks the king into a visible rook's file.
    world = FowBoard(None)
    world.set_piece_at(chess.A1, chess.Piece.from_symbol('K'))
    world.set_piece_at(chess.B8, chess.Piece.from_symbol('r'))
    world.set_piece_at(chess.H8, chess.Piece.from_symbol('k'))
    world.turn = chess.WHITE
    world.castling_rights = 0
    legal = list(fow_moves(world))
    move = pimc_fallback([world], legal, chess.WHITE, METHODS['A'],
                         material_eval, np.random.default_rng(0))
    # a1b1 and a1b2 step into the b-file rook -> losing; a1a2 survives.
    assert move == 'a1a2'
    print('ok: fallback respects the king veto')


if __name__ == '__main__':
    test_configs_registry()
    test_shortlist_structure()
    test_shortlist_capture_slot()
    test_score_pair_terminal()
    test_score_pair_beta_paths_agree_on_true_source()
    test_fallback_first_and_zero_budget()
    test_full_rounds_and_determinism()
    test_partial_round_discarded()
    test_stub_world_always_has_both_kings()
    test_fallback_veto()
    print('all planner tests passed')
