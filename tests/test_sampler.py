"""DiffusionBelief tests: constraint logic on hand-built views and sample validity,
observation consistency, marginals and log_prob with the exported model.

    .venv/bin/python -m pytest tests/test_sampler.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from belief.features import ViewFeatures, view_from_arrays  # noqa: E402
from belief.sampler import DiffusionBelief, SamplerConfig, budget, check_board  # noqa: E402

ONNX = ROOT / 'belief' / 'B.onnx'
SHARDS = [ROOT / 'data' / 'shards' / d for d in ('arena3k', 'arena', 'selfplay')]


def uniform_forward(inputs):
    """A model with no opinion: equal logits everywhere."""
    return np.zeros((len(inputs['ply']), 64, 7), dtype=np.float32)


def view(cur_fill: dict[int, int], inv, white=True) -> ViewFeatures:
    cur = np.zeros(64, np.uint8)
    cur[4] = 7                                   # own king on e1
    for sq, code in cur_fill.items():
        cur[sq] = code
    v = view_from_arrays(cur, np.zeros(64), np.full(64, 5), np.array(inv), 0, 1, white)
    return v


def test_budget_subtracts_visible_pieces_by_bishop_colour():
    # visible enemy bishops on c8 (light, canonical) and an enemy king on e8
    v = view({58: 10, 60: 13}, [8, 2, 1, 1, 2, 1, 1])
    need, deficit = budget(v.cur[None], v.inv[None])
    assert need[0].tolist() == [8, 2, 0, 1, 2, 1, 0] and deficit[0] == 0
    v = view({58: 10, 56: 10}, [8, 2, 1, 1, 2, 1, 1])  # bishops on c8 and a8, both light: unseen promotion
    need, deficit = budget(v.cur[None], v.inv[None])
    assert need[0, 2] == 0 and deficit[0] == 1


def test_constraints_place_exact_inventory_with_uninformed_model():
    # only the king and one rook remain: exactly one K and at most one R on unseen squares
    v = view({}, [0, 0, 0, 0, 1, 0, 1])
    s = DiffusionBelief(forward=uniform_forward)
    boards = s.sample([v], 32, 4, np.random.default_rng(0))[0]
    for b in boards:
        assert b.count('k') == 1 and b.count('r') == 1 and b.count('p') == 0
        assert all(check_board(v, b).values())
    assert not s.last_info['violated'].any()


def test_forbid_empty_when_squares_must_be_filled():
    # 3 unseen squares, 3 pieces left: every unseen square must hold a piece
    cur = np.ones(64, np.uint8)
    cur[4] = 7
    cur[[40, 41, 42]] = 0
    v = view_from_arrays(cur, np.zeros(64), np.zeros(64), np.array([1, 1, 0, 0, 0, 0, 1]), 0, 1, True)
    s = DiffusionBelief(forward=uniform_forward)
    for b in s.sample([v], 16, 2, np.random.default_rng(1))[0]:
        assert sorted(b[40:43]) == ['k', 'n', 'p']


def test_lookahead_never_strands_a_pawn_on_back_rank_squares():
    # unseen: 3 back-rank squares (a8 b8 c8) + 1 non-back square (a6); budget: king + one pawn.
    # The pawn can only go on a6, so a6 must never be left empty or given the king.
    cur = np.ones(64, np.uint8)
    cur[4] = 7
    cur[[56, 57, 58, 40]] = 0
    v = view_from_arrays(cur, np.zeros(64), np.zeros(64), np.array([1, 0, 0, 0, 0, 0, 1]), 0, 1, True)
    s = DiffusionBelief(forward=uniform_forward)
    for T in (1, 2, 4):
        boards, info = s.sample_with_info([v], 64, T, np.random.default_rng(T))
        assert not info['violated'].any()
        for b in boards[0]:
            assert b[40] == 'p' and b.count('k') == 1 and all(check_board(v, b).values())


def test_lookahead_keeps_a_square_for_each_bishop_colour():
    # unseen: one light (b1 is light) and several dark squares; budget: king + light bishop.
    cur = np.ones(64, np.uint8)
    cur[4] = 7
    unseen = [1, 41, 43, 45]          # b1 light; b6, d6, f6 dark
    cur[unseen] = 0
    v = view_from_arrays(cur, np.zeros(64), np.zeros(64), np.array([0, 0, 1, 0, 0, 0, 1]), 0, 1, True)
    s = DiffusionBelief(forward=uniform_forward)
    boards, info = s.sample_with_info([v], 64, 2, np.random.default_rng(0))
    assert not info['violated'].any()
    assert all(b[1] == 'b' and b.count('k') == 1 for b in boards[0])


def test_rank_rules_and_black_orientation():
    v = view({}, [8, 0, 0, 0, 0, 0, 1], white=False)
    s = DiffusionBelief(forward=uniform_forward)
    for b in s.sample([v], 16, 4, np.random.default_rng(2))[0]:
        assert b[4 ^ 56] == 'k'                    # own king: canonical e1 -> original e8
        assert 'P' not in b[:8] + b[56:]           # enemy (White) pawns never on ranks 1/8
        assert b.count('K') == 1
        assert all(check_board(v, b).values())


def test_rank_rules_off_allows_back_rank_pawns():
    v = view({}, [8, 0, 0, 0, 0, 0, 1])
    s = DiffusionBelief(forward=uniform_forward, config=SamplerConfig(rank_rules=False))
    boards = s.sample([v], 64, 2, np.random.default_rng(3))[0]
    assert any('p' in b[56:] or 'p' in b[:8] for b in boards)


def load_views(n=64, seed=0):
    for d in SHARDS:
        f = sorted((d / 'val').glob('shard_*.npz')) if d.exists() else []
        if f:
            z = dict(np.load(f[0]))
            idx = np.random.default_rng(seed).choice(len(z['t']), n, replace=False)
            return [view_from_arrays(z['cur'][i], z['last'][i], z['age'][i], z['inv'][i], z['ply'][i],
                                     z['to_move'][i], z['viewer'][i] == 0, z['t'][i]) for i in idx], z, idx
    pytest.skip('no validation shards')


needs_model = pytest.mark.skipif(not ONNX.exists(), reason='belief/B.onnx not exported')


@needs_model
def test_sample_validity_and_observation_consistency():
    views, z, idx = load_views(64)
    s = DiffusionBelief(str(ONNX))
    boards = s.sample(views, 8, 4, np.random.default_rng(0))
    checks = [check_board(v, b) for v, bs in zip(views, boards) for b in bs]
    assert all(c['obs_consistent'] and c['unseen_only_enemy'] for c in checks)
    # unseen promotions make inv too small for the truth; skip those views for the validity rate
    ok_views = [not z['promo'][i] for i in idx]
    valid = [all(c.values()) for c, keep in zip(checks, np.repeat(ok_views, 8)) if keep]
    assert np.mean(valid) >= 0.99, np.mean(valid)


@needs_model
def test_marginals_and_log_prob():
    views, z, idx = load_views(8, seed=1)
    s = DiffusionBelief(str(ONNX))
    for v, i in zip(views, idx):
        p = s.marginals(v)
        assert p.shape == (64, 7) and np.allclose(p.sum(1), 1)
        b = s.sample([v], 1, 4, np.random.default_rng(0))[0][0]
        lp = s.log_prob(v, b)
        assert np.isfinite(lp) and lp <= 0
        assert s.log_prob(v, b.replace('?', '-')) == lp
    # a board that contradicts a visible square has probability zero
    v = views[0]
    vis = next(k for k, ch in enumerate(v.obs) if ch == '-')
    bad = list(s.sample([v], 1, 1, np.random.default_rng(0))[0][0])
    bad[vis] = 'q' if v.viewer_is_white else 'Q'
    assert s.log_prob(v, ''.join(bad)) == float('-inf')
