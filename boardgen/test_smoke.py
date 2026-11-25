#!/usr/bin/env python3
"""End-to-end smoke test: encoding round-trips, score(), generate().

Run after training any checkpoint:
    .venv/Scripts/python boardgen/test_smoke.py --checkpoint checkpoints/smoke/belief.pt --replay data/smoke/game-000000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoding import EMPTY, UNKNOWN, board_string_to_classes, canonicalize
from sample import (BeliefState, TorchPredictor, classes_to_fen_board,
                    fen_board_to_classes, generate, score)


def test_canonicalize_involution():
    rng = np.random.default_rng(0)
    classes = rng.integers(0, 14, size=(5, 64)).astype(np.uint8)
    once = canonicalize(classes, False)
    assert np.array_equal(canonicalize(once, False), classes)
    assert np.array_equal(canonicalize(classes, True), classes)
    print('ok: canonicalize is an involution')


def test_fen_round_trip():
    start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    for is_white in (True, False):
        classes = fen_board_to_classes(start, is_white)
        assert classes_to_fen_board(classes, is_white) == start
    # White's back rank should be at squares 0..7 for a white observer...
    white_view = fen_board_to_classes(start, True)
    assert white_view[4] == 5  # K on e1
    # ...and for a canonicalized black observer, their king lands on d1 (mirror).
    black_view = fen_board_to_classes(start, False)
    assert black_view[4] == 5  # black k at e8 -> mirrored to e1, case-swapped to K
    print('ok: FEN round-trip and orientation')


def replay_to_state(replay, is_white, upto_ply):
    state = BeliefState(is_white)
    for frame in replay['frames'][:upto_ply + 1]:
        obs = frame['white' if is_white else 'black']
        to_move = (frame['ply'] % 2 == 0) == is_white
        state.update(obs, frame['ply'], to_move)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--ply', type=int, default=10)
    args = parser.parse_args()

    test_canonicalize_involution()
    test_fen_round_trip()

    replay = json.loads(args.replay.read_text())
    frame = replay['frames'][args.ply]
    predictor = TorchPredictor(args.checkpoint)
    state = replay_to_state(replay, True, args.ply)

    # The true board, as a FEN board field.
    true_classes_raw = board_string_to_classes(frame['board'])
    true_fen = classes_to_fen_board(canonicalize(true_classes_raw[None], True)[0], True)

    # score(): true board vs decoys (true board with hidden-square noise).
    rng = np.random.default_rng(1)
    hidden = np.where(state.obs == UNKNOWN)[0]
    decoys = []
    true_canonical = canonicalize(true_classes_raw[None], True)[0]
    for _ in range(63):
        noisy = true_canonical.copy()
        squares = rng.choice(hidden, size=min(6, len(hidden)), replace=False)
        noisy[squares] = rng.integers(0, 13, size=len(squares))
        decoys.append(classes_to_fen_board(noisy, True))
    weights = score(predictor, state, [true_fen] + decoys)
    rank = int((weights > weights[0]).sum()) + 1
    print(f'ok: score() ran; true board rank {rank}/64 among noisy decoys '
          f'(log-weight {weights[0]:.1f}, decoy mean {weights[1:].mean():.1f})')

    # generate(): visible squares must match the observation exactly.
    boards, acceptance = generate(predictor, state, n=16, steps=4, rng=rng)
    assert len(boards) == 16 and acceptance == 1.0
    visible = state.obs != UNKNOWN
    for fen in boards:
        classes = fen_board_to_classes(fen, True)
        assert np.array_equal(classes[visible], state.obs[visible]), 'visible square mismatch'
        assert (classes != UNKNOWN).all(), 'MASK token leaked into a generated board'
    unique = len(set(boards))
    print(f'ok: generate() produced 16 boards consistent with the observation '
          f'({unique} unique)')
    print('all smoke tests passed')


if __name__ == '__main__':
    main()
