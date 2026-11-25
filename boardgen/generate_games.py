#!/usr/bin/env python3
"""In-process fog-of-war self-play for dataset generation (Person C).

Runs games directly against the referee's rules code (FowBoard, observation,
fow_moves) without subprocess bots, so it is fast and works on Windows.
Output matches the arena replay schema (frames with true board + both
observations), so dataset.py treats downloaded arena replays and these
self-play games identically.

Policies choose among the true board's legal moves but score them using only
that side's observation, like a real bot would.

Usage (from the repo root):
    .venv/Scripts/python boardgen/generate_games.py --games 1500 --out data/selfplay --workers 8
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from multiprocessing import Pool
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from referee import FowBoard, fow_moves, make_frame, observation, outcome, position_key

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def uniform_policy(board, obs, rng):
    return rng.choice(fow_moves(board))


def greedy_policy(board, obs, rng):
    # baseline.py's capture-preferring heuristic, scored from the observation only.
    def score(move):
        target = obs[move.to_square]
        value = 0
        if target not in '?-':
            value = VALUES[chess.Piece.from_symbol(target).piece_type] * 10
        if move.promotion:
            value += VALUES[move.promotion] * 5
        return value + rng.random()
    return max(fow_moves(board), key=score)


def epsilon_greedy_policy(board, obs, rng, epsilon=0.3):
    if rng.random() < epsilon:
        return uniform_policy(board, obs, rng)
    return greedy_policy(board, obs, rng)


POLICIES = {'uniform': uniform_policy, 'greedy': greedy_policy, 'mixed': epsilon_greedy_policy}


def play_game(seed, white_policy, black_policy, max_plies=1000):
    rng = random.Random(seed)
    board = FowBoard()
    policies = {chess.WHITE: POLICIES[white_policy], chess.BLACK: POLICIES[black_policy]}
    repetitions = Counter({position_key(board): 1})
    clocks = [180.0, 180.0]  # schema compatibility; no clock pressure in-process
    replay = {'id': f'selfplay-{seed}', 'white': white_policy, 'black': black_policy,
              'result': '*', 'reason': 'In progress',
              'timeControl': {'baseSeconds': 180, 'incrementSeconds': 2},
              'frames': [make_frame(board, clocks)]}
    while True:
        terminal = outcome(board, repetitions)
        if terminal:
            replay.update(zip(('result', 'reason'), terminal))
            return replay
        if len(board.move_stack) >= max_plies:
            replay.update(result='1/2-1/2', reason='Move limit reached')
            return replay
        obs = observation(board, board.turn)
        move = policies[board.turn](board, obs, rng)
        board.push(move)
        repetitions[position_key(board)] += 1
        replay['frames'].append(make_frame(board, clocks, move.uci()))


def generate_one(task):
    index, seed, out_dir, pairing = task
    white_policy, black_policy = pairing
    replay = play_game(seed, white_policy, black_policy)
    path = Path(out_dir) / f'game-{index:06d}.json'
    path.write_text(json.dumps(replay))
    return len(replay['frames']) - 1, replay['result']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--out', type=Path, default=Path('data/selfplay'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairings = [('uniform', 'uniform'), ('greedy', 'greedy'), ('mixed', 'mixed'),
                ('mixed', 'uniform'), ('uniform', 'mixed'),
                ('greedy', 'mixed'), ('mixed', 'greedy')]
    tasks = [(i, args.seed * 1_000_000 + i, str(args.out), pairings[i % len(pairings)])
             for i in range(args.games)]

    started = time.monotonic()
    plies = 0
    results = Counter()
    with Pool(args.workers) as pool:
        for done, (game_plies, result) in enumerate(pool.imap_unordered(generate_one, tasks), 1):
            plies += game_plies
            results[result] += 1
            if done % 50 == 0 or done == args.games:
                elapsed = time.monotonic() - started
                print(f'{done}/{args.games} games, {plies} plies, '
                      f'{elapsed:.0f}s ({done / elapsed:.1f} games/s)', flush=True)
    print(f'Results: {dict(results)}')
    print(f'Total plies: {plies} -> ~{plies * 2} training examples (both perspectives)')


if __name__ == '__main__':
    main()
