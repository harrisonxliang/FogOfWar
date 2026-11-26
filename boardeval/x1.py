#!/usr/bin/env python3
"""X1 harness: which opponent model best predicts real moves?

Replays finished games; at every ply, each contestant predicts the mover's
move from what that contestant is allowed to see; we score predictions against
the move actually played. Works on self-play games today and on downloaded
arena replays unchanged (same JSON schema).

Usage:
    .venv/Scripts/python boardeval/x1.py --replays data/selfplay --checkpoint checkpoints/v1/belief.pt --games 50
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from referee import FowBoard, fow_moves
from boardgen.encoding import HistoryTracker
from boardgen.sample import BeliefState, TorchPredictor
from boardeval.evaluator import get_evaluator
from boardeval.reply import (MoverContext, clairvoyant, like_me, naive_last_seen,
                             to_distribution, uniform_random)


def board_from_frame(frame, side):
    board = FowBoard(None)
    board.turn = side
    import chess
    for square, symbol in enumerate(frame['board']):
        if symbol != '-':
            board.set_piece_at(square, chess.Piece.from_symbol(symbol))
    board.castling_rights = 0  # replay frames carry no rights; 1-ply eval only
    return board


class Tally:
    def __init__(self, tau=150.0, floor=0.02):
        self.tau = tau
        self.floor = floor
        self.top1 = 0
        self.top3 = 0
        self.reciprocal = 0.0
        self.count = 0
        self.logp = 0.0       # sum of log p(actual) under the smoothed distribution
        self.logp_count = 0

    def add(self, ranked, actual, legal):
        self.count += 1
        ucis = [uci for uci, _ in ranked]
        if actual in ucis:
            rank = ucis.index(actual) + 1
            self.top1 += rank == 1
            self.top3 += rank <= 3
            self.reciprocal += 1.0 / rank
        # absent from the ranking -> all rank-misses; LL still scored below.
        # Log-likelihood: the smoothing floor (NOTES.md §4) guarantees every
        # legal move keeps mass, so this never hits -inf.
        dist = to_distribution(ranked, legal, tau=self.tau, floor=self.floor)
        p_actual = dist.get(actual)
        if p_actual:  # actual not legal would mean a replay/rules mismatch; skip
            self.logp += float(np.log(p_actual))
            self.logp_count += 1

    def row(self):
        if not self.count:
            return '    (no predictions)'
        ll = self.logp / self.logp_count if self.logp_count else float('nan')
        return (f'top1 {self.top1 / self.count:6.3f}   '
                f'top3 {self.top3 / self.count:6.3f}   '
                f'MRR {self.reciprocal / self.count:6.3f}   '
                f'LL {ll:7.3f}   '
                f'n={self.count}')


def run(replay_paths, checkpoint, evaluator_kind, games_limit, j, steps, seed,
        tau=150.0, floor=0.02):
    evaluate, close = get_evaluator(evaluator_kind)
    predictor = TorchPredictor(checkpoint) if checkpoint else None
    rng = np.random.default_rng(seed)
    models = {'clairvoyant': lambda ctx: clairvoyant(ctx, evaluate),
              'naive': lambda ctx: naive_last_seen(ctx, evaluate),
              'uniform': lambda ctx: uniform_random(ctx, evaluate)}
    if predictor is not None:
        models['like_me_diffusion'] = lambda ctx: like_me(
            ctx, evaluate, predictor, j=j, steps=steps, rng=rng)
    tallies = defaultdict(lambda: Tally(tau=tau, floor=floor))
    started = time.monotonic()
    plies = 0
    for game_index, path in enumerate(replay_paths[:games_limit]):
        replay = json.loads(Path(path).read_text())
        frames = replay.get('frames', [])
        trackers = {True: HistoryTracker(True), False: HistoryTracker(False)}
        beliefs = {True: BeliefState(True), False: BeliefState(False)}
        for i in range(len(frames) - 1):
            frame, actual = frames[i], frames[i + 1]['move']
            side = frame['ply'] % 2 == 0
            for is_white in (True, False):
                obs = frame['white' if is_white else 'black']
                trackers[is_white].update(obs)
                beliefs[is_white].update(obs, frame['ply'], (frame['ply'] % 2 == 0) == is_white)
            ctx = MoverContext(side=side, true_board=board_from_frame(frame, side),
                               observation=frame['white' if side else 'black'],
                               tracker=trackers[side], belief_state=beliefs[side],
                               ply=frame['ply'])
            legal = [move.uci() for move in fow_moves(ctx.true_board)]
            for name, model in models.items():
                ranked = model(ctx)
                if ranked:
                    tallies[name].add(ranked, actual, legal)
            plies += 1
        if (game_index + 1) % 10 == 0:
            print(f'{game_index + 1} games, {plies} plies, '
                  f'{time.monotonic() - started:.0f}s', file=sys.stderr, flush=True)
    close()
    print(f'\nX1: opponent-move prediction ({plies} plies, '
          f'evaluator={evaluator_kind}, J={j}, T={steps})')
    for name in ('uniform', 'naive', 'clairvoyant', 'like_me_diffusion'):
        if name in tallies:
            print(f'  {name:18s} {tallies[name].row()}')
    return tallies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays', type=Path, nargs='+', required=True)
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='belief model .pt; omit to skip the diffusion contestant')
    parser.add_argument('--evaluator', choices=['material', 'stockfish'], default='material')
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--j', type=int, default=4, help='believed boards per view')
    parser.add_argument('--steps', type=int, default=4, help='diffusion steps')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tau', type=float, default=150.0,
                        help='reply softmax temperature (cp scale); fit per team offline')
    parser.add_argument('--floor', type=float, default=0.02,
                        help='uniform smoothing mass so LL never hits -inf')
    args = parser.parse_args()
    paths = []
    for entry in args.replays:
        paths.extend(sorted(entry.glob('*.json')) if entry.is_dir() else [entry])
    run(paths, args.checkpoint, args.evaluator, args.games, args.j, args.steps,
        args.seed, tau=args.tau, floor=args.floor)


if __name__ == '__main__':
    main()
