"""ViewTracker tests: orientation, last-seen/age, capture/inventory inference and
the true-board consistency checks, on hand-built positions and on replays.

    .venv/bin/python -m pytest tests/test_features.py -q
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from referee import FowBoard, fow_moves, make_frame, observation  # noqa: E402
from belief.dataset import FeatureCheckError, game_examples, load_replay  # noqa: E402
from belief.features import (INV_START, LIGHT, ViewTracker, board_from_canonical,  # noqa: E402
                             canonical_cur, canonical_target)

REPLAY_DIRS = [ROOT / 'data' / 'selfplay', ROOT / 'data' / 'selfplay_uu', ROOT / 'data' / 'raw' / 'games']


def replay_paths(limit_per_dir=400):
    out = []
    for d in REPLAY_DIRS:
        if d.exists():
            out += sorted(list(d.glob('*.json')) + list(d.glob('*.json.gz')))[:limit_per_dir]
    return out


def play(moves, fen=None):
    """Replay-format frames for a UCI move list from a start position."""
    board = FowBoard(fen) if fen else FowBoard()
    frames = [make_frame(board, [180, 180])]
    for uci in moves:
        move = chess.Move.from_uci(uci)
        assert move in fow_moves(board), uci
        board.push(move)
        frames.append(make_frame(board, [180, 180], uci))
    return {'id': 'handmade', 'frames': frames}


# --- orientation ---------------------------------------------------------------

def test_initial_position_is_symmetric():
    board = FowBoard()
    w = canonical_cur(observation(board, chess.WHITE), True)
    b = canonical_cur(observation(board, chess.BLACK), False)
    assert np.array_equal(w, b)
    assert (w[:8] == [5, 3, 4, 6, 7, 4, 3, 5]).all()      # own R N B Q K B N R on rank 1
    assert (w[8:16] == 2).all() and (w[32:] == 0).all()


@pytest.mark.parametrize('seed', range(20))
def test_black_view_equals_white_view_of_mirrored_board(seed):
    rng = np.random.default_rng(seed)
    board = FowBoard()
    for _ in range(int(rng.integers(1, 40))):
        moves = fow_moves(board)
        if not moves or board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            break
        board.push(moves[int(rng.integers(len(moves)))])
    mirrored = board.mirror()
    b = canonical_cur(observation(board, chess.BLACK), False)
    w = canonical_cur(observation(mirrored, chess.WHITE), True)
    assert np.array_equal(b, w)


def test_bishop_colours_in_canonical_coordinates():
    # Each side starts with one light and one dark bishop in its own canonical frame.
    assert LIGHT[chess.C1] != LIGHT[chess.F1]
    assert list(INV_START) == [8, 2, 1, 1, 2, 1, 1]
    t = ViewTracker(True)
    t.observe(observation(FowBoard(), chess.WHITE))
    assert list(t.features().inv) == [8, 2, 1, 1, 2, 1, 1]


# --- last-seen and age on hand-built observation sequences -----------------------

def obs_with(fill: dict[int, str], default='?'):
    s = [default] * 64
    s[chess.E1] = 'K'                       # the viewer always sees its own king
    for sq, ch in fill.items():
        s[sq] = ch
    return ''.join(s)


def test_last_seen_and_age_buckets():
    t = ViewTracker(True)
    # t=0: d5 holds an enemy knight, e5 is empty, f5 holds our pawn
    t.observe(obs_with({chess.D5: 'n', chess.E5: '-', chess.F5: 'P'}))
    f = t.features()
    assert f.cur[chess.D5] == 9 and f.last[chess.D5] == 3 and f.age[chess.D5] == 0
    assert f.last[chess.E5] == 1 and f.last[chess.F5] == 1
    assert f.last[chess.A8] == 0 and f.age[chess.A8] == 5
    assert f.ply == 0 and f.to_move == 1 and f.t == 0
    expected_age = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 9: 4}
    for step in range(1, 10):
        t.observe(obs_with({chess.F5: 'P'}))
        f = t.features()
        assert f.cur[chess.D5] == 0
        assert f.last[chess.D5] == 3, 'last keeps the enemy knight until the square is seen again'
        if step in expected_age:
            assert f.age[chess.D5] == expected_age[step], step
        assert f.to_move == int(step % 2 == 0)
    # seeing the square again overwrites last and resets age
    t.observe(obs_with({chess.D5: '-'}))
    f = t.features()
    assert f.last[chess.D5] == 1 and f.age[chess.D5] == 0 and f.ply == 10 // 8


def test_black_viewer_canonical_squares():
    t = ViewTracker(False)
    s = ['?'] * 64
    s[chess.E8] = 'k'
    s[chess.D4] = 'Q'                     # enemy queen on d4 -> canonical d5
    t.observe(''.join(s))
    f = t.features()
    assert f.cur[chess.E1] == 7           # own king on canonical e1
    assert f.cur[chess.D5] == 12          # enemy queen
    assert f.to_move == 0                 # White moves first


# --- capture and inventory inference ---------------------------------------------

def test_ordinary_capture_updates_inventory_by_bishop_colour():
    # 1. e4 e5 2. Bc4 Bc5 3. Bxf7 Kxf7: Black captures White's f1 bishop.
    rep = play(['e2e4', 'e7e5', 'f1c4', 'f8c5', 'c4f7', 'e8f7'])
    arrays, stats = game_examples(rep)          # strict: raises on any mismatch
    t = ViewTracker(False)                      # Black viewer captured White's Bc4 on f7
    for k, fr in enumerate(rep['frames']):
        if k > 0 and k % 2 == 0:
            t.own_move(fr['move'])
        t.observe(fr['black'])
    # White's f1 bishop: canonical for Black is f8, (5+7)%2 = 0 -> dark
    assert list(t.features().inv) == [8, 2, 1, 0, 2, 1, 1]
    assert stats['inv_mismatch'] == 0


def test_en_passant_capture():
    rep = play(['e2e4', 'a7a6', 'e4e5', 'd7d5', 'e5d6'])
    t = ViewTracker(True)
    for k, fr in enumerate(rep['frames']):
        if k > 0 and k % 2 == 1:
            t.own_move(fr['move'])
        t.observe(fr['white'])
    assert t.captures and t.captures[-1][1] == 1
    assert t.features().inv[0] == 7
    _, stats = game_examples(rep)
    assert stats['inv_mismatch'] == 0


def test_capturing_a_promoted_piece_removes_a_pawn_unit():
    t = ViewTracker(True)
    t.inv[:] = [7, 2, 1, 1, 2, 0, 1]           # queen already gone
    s = ['?'] * 64
    s[chess.E1], s[chess.D1], s[chess.D8] = 'K', 'R', 'q'
    for sq in range(chess.D2, chess.D8, 8):
        s[sq] = '-'
    t.observe(''.join(s))
    t.own_move('d1d8')
    assert list(t.features().inv) == [6, 2, 1, 1, 2, 0, 1]


def test_own_move_requires_own_piece():
    t = ViewTracker(True)
    t.observe(observation(FowBoard(), chess.WHITE))
    with pytest.raises(ValueError):
        t.own_move('e4e5')


# --- consistency with the true board on replays -------------------------------

@pytest.mark.parametrize('path', replay_paths(), ids=lambda p: p.name)
def test_replay_features_match_true_board(path):
    rep = load_replay(path)
    arrays, stats = game_examples(rep, strict=True)
    assert stats['inv_mismatch'] == 0
    n = len(rep['frames'])
    assert len(arrays['t']) == 2 * n
    # targets: every hidden square labelled, no visible square labelled
    hidden = arrays['cur'] == 0
    assert ((arrays['tgt'] >= 0) == hidden).all()


def test_assertion_catches_inconsistent_view():
    rep = play(['e2e4', 'e7e5'])
    bad = copy.deepcopy(rep)
    w = list(bad['frames'][1]['white'])
    w[chess.E4] = '?'                        # hide our own pawn
    bad['frames'][1]['white'] = ''.join(w)
    with pytest.raises(FeatureCheckError):
        game_examples(bad)
    bad = copy.deepcopy(rep)
    w = list(bad['frames'][2]['white'])
    w[chess.E7] = 'p'                        # claim to see a pawn that moved away
    bad['frames'][2]['white'] = ''.join(w)
    with pytest.raises(FeatureCheckError):
        game_examples(bad)


@pytest.mark.parametrize('path', replay_paths(40), ids=lambda p: p.name)
def test_board_round_trip(path):
    rep = load_replay(path)
    for viewer in (True, False):
        t = ViewTracker(viewer)
        for k, fr in enumerate(rep['frames']):
            if k > 0 and (k % 2 == 1) == viewer:
                t.own_move(fr['move'])
            t.observe(fr['white' if viewer else 'black'])
            f = t.features()
            truth = canonical_target(fr['board'], f)
            fill = np.where(truth <= 6, truth, 0)
            assert board_from_canonical(fill, f) == fr['board']
