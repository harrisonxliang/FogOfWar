#!/usr/bin/env python3
"""Protocol entry point: turns the planner library into an arena bot.

Speaks the referee's stdin/stdout protocol (see uniform_random.py, the
reference implementation this loop mirrors):
    in:  'fow' | '<w|b> <base> <increment>' | 64-char observations
         (one before EACH player's move) | a bare float = our clock = our turn
    out: 'ready' | 'move <uci>'
Diagnostics go to stderr only — stdout is protocol-sacred.

Usage (and the referee invocation for local games):
    .venv/Scripts/python bot/entry.py --method A
    .venv/Scripts/python referee.py --trusted-local \
        --white ".venv/Scripts/python bot/entry.py" \
        --black ".venv/Scripts/python uniform_random.py" --output game.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np

from bot.configs import for_method
from bot.planner import plan_move
from bot.stub_tracker import StubTracker

OBSERVATION_CHARACTERS = frozenset('?-PNBRQKpnbrqk')


def budget_for(remaining: float, increment: float) -> float:
    """Simple v1 time manager: spread the base over ~30 moves, bank most of
    the increment, never commit more than a quarter of what's left."""
    return max(0.05, min(remaining / 30.0 + 0.8 * increment, remaining / 4.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', default='A', help='A|N|C (B/D pending own_worlds)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if sys.stdin.readline().strip().lower() != 'fow':
        return
    header = sys.stdin.readline().split()
    if len(header) != 3 or header[0].lower() not in ('w', 'b'):
        raise ValueError('Expected color, base time, and increment')
    side = header[0].lower() == 'w'
    increment = float(header[2])

    cfg = for_method(args.method)
    tracker = StubTracker(side)
    rng = np.random.default_rng(args.seed)
    print('ready', flush=True)

    for raw in sys.stdin:
        line = raw.strip()
        if len(line) == 64 and set(line) <= OBSERVATION_CHARACTERS:
            tracker.update(line)
            continue
        try:
            remaining = float(line)
        except ValueError:
            continue
        if tracker.current is None:
            return
        legal = tracker.legal_moves()
        if not legal:
            return  # nothing to play; the referee will adjudicate
        worlds = tracker.worlds()
        uci, diag = plan_move(
            worlds, legal, side, cfg, budget_for(remaining, increment), rng,
            danger_index=0 if tracker.danger() else None)
        tracker.record_move(chess.Move.from_uci(uci))
        print(f'[bot] ply move={uci} rounds={diag["completed_rounds"]} '
              f'clock={remaining:.1f}', file=sys.stderr, flush=True)
        print('move ' + uci, flush=True)


if __name__ == '__main__':
    main()
