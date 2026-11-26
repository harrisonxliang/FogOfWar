#!/usr/bin/env python3
"""Windows-friendly local match runner.

The course referee (referee.py) is POSIX-only at the process layer
(select.select on pipes, os.killpg), so it cannot drive subprocess bots on
Windows. This runner reuses the referee's RULES verbatim — FowBoard,
fow_moves, observation, outcome, position_key, make_frame — and replaces only
the process plumbing with blocking pipes + reader threads. Same protocol,
same replay schema, same termination and clock semantics (charge wall time,
forfeit below zero, add the increment after the move).

The arena runs the real referee on Linux against the SAME bot/entry.py — this
file exists purely so local games and the 50-clean-games gate can run here.

Usage:
    .venv/Scripts/python bot/local_match.py ^
        --white ".venv\\Scripts\\python.exe bot/entry.py --method A" ^
        --black ".venv\\Scripts\\python.exe uniform_random.py" ^
        --games 4 --base-time 60 --increment 1 --out-dir local_games
Colors swap every game (color-swapped pairs, per the working agreements).
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from referee import (FowBoard, fow_moves, make_frame, observation, outcome,
                     position_key)

READY_TIMEOUT = 20.0
MOVE_GRACE = 5.0  # hard wait beyond the clock before declaring a hang


class MatchBotFailure(Exception):
    pass


class LocalBot:
    """One subprocess bot: blocking writes, a reader thread, a line queue."""

    def __init__(self, argv, side, base, increment):
        self.side = side
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self._send('fow')
        self._send(f'{"w" if side else "b"} {base:g} {increment:g}')
        ready = self._next_line(READY_TIMEOUT)
        if ready != 'ready':
            raise MatchBotFailure(f'expected ready, got {ready!r}')

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.lines.put(line.rstrip('\n'))
        except ValueError:
            pass  # pipe closed during shutdown
        self.lines.put(None)  # EOF marker

    def _send(self, line):
        try:
            self.proc.stdin.write(line + '\n')
            self.proc.stdin.flush()
        except OSError as error:
            raise MatchBotFailure(f'bot pipe closed: {error}')

    def _next_line(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise MatchBotFailure('bot timed out (no output)')
        if line is None:
            raise MatchBotFailure('bot exited')
        return line

    def observe(self, obs):
        self._send(obs)

    def move(self, remaining):
        started = time.monotonic()
        self._send(f'{remaining:.3f}')
        while True:
            line = self._next_line(remaining + MOVE_GRACE)
            if line.startswith('move '):
                return line[5:].strip(), time.monotonic() - started
            # Tolerate stray non-protocol stdout the way a lenient reader
            # would; the real referee is stricter, so don't rely on this.

    def close(self):
        for stream in (self.proc.stdin,):
            try:
                stream.close()
            except OSError:
                pass
        try:
            self.proc.kill()
        except OSError:
            pass


def run_local_match(white_argv, black_argv, base=60.0, increment=1.0,
                    max_plies=600, match_id=None):
    board = FowBoard()
    clocks = [float(base), float(base)]
    replay = {'id': match_id or str(uuid.uuid4()),
              'white': ' '.join(white_argv), 'black': ' '.join(black_argv),
              'result': '*', 'reason': 'In progress',
              'timeControl': {'baseSeconds': base, 'incrementSeconds': increment},
              'frames': [make_frame(board, clocks)]}
    repetitions = Counter({position_key(board): 1})
    bots = []
    try:
        for side, argv in ((chess.WHITE, white_argv), (chess.BLACK, black_argv)):
            try:
                bots.append(LocalBot(argv, side, base, increment))
            except (OSError, MatchBotFailure) as error:
                replay.update(result='0-1' if side else '1-0',
                              reason=f'Bot startup failed: {error}')
                return replay
        while True:
            terminal = outcome(board, repetitions)
            if terminal:
                replay.update(zip(('result', 'reason'), terminal))
                return replay
            if len(board.move_stack) >= max_plies:
                replay.update(result='1/2-1/2', reason='Move limit reached')
                return replay
            for index, side in enumerate((chess.WHITE, chess.BLACK)):
                bots[index].observe(observation(board, side))
            index = 0 if board.turn else 1
            try:
                uci, elapsed = bots[index].move(clocks[index])
                clocks[index] -= elapsed
                if clocks[index] < 0:
                    raise MatchBotFailure('Time forfeit')
                move = chess.Move.from_uci(uci)
                if move not in fow_moves(board):
                    raise MatchBotFailure(f'Illegal move: {uci}')
            except MatchBotFailure as error:
                replay.update(result='0-1' if index == 0 else '1-0',
                              reason=str(error),
                              failedSide='white' if index == 0 else 'black')
                return replay
            board.push(move)
            clocks[index] += increment
            repetitions[position_key(board)] += 1
            replay['frames'].append(make_frame(board, clocks, uci))
    finally:
        for bot in bots:
            bot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--white', required=True)
    parser.add_argument('--black', required=True)
    parser.add_argument('--games', type=int, default=1,
                        help='colors swap every game')
    parser.add_argument('--base-time', type=float, default=60.0)
    parser.add_argument('--increment', type=float, default=1.0)
    parser.add_argument('--max-plies', type=int, default=600)
    parser.add_argument('--out-dir', type=Path, default=Path('local_games'))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sides = [args.white.split(), args.black.split()]
    score = Counter()
    for game in range(args.games):
        white, black = sides[game % 2], sides[(game + 1) % 2]
        replay = run_local_match(white, black, args.base_time, args.increment,
                                 args.max_plies, match_id=f'local-{game:03d}')
        path = args.out_dir / f'game_{game:03d}.json'
        path.write_text(json.dumps(replay, indent=2) + '\n')
        a_is_white = game % 2 == 0
        result = replay['result']
        if result == '1-0':
            score['A' if a_is_white else 'B'] += 1
        elif result == '0-1':
            score['B' if a_is_white else 'A'] += 1
        else:
            score['draw'] += 1
        print(f'game {game:03d}: {result:7s} ({len(replay["frames"]) - 1:3d} plies) '
              f'{replay["reason"]}', flush=True)
    print(f"\nA (--white cmd) {score['A']} - {score['B']} B (--black cmd), "
          f"{score['draw']} draws over {args.games} game(s)")


if __name__ == '__main__':
    main()
