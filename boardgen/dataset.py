#!/usr/bin/env python3
"""Replay JSONs (self-play or arena downloads) -> compact training dataset.

Each frame yields up to two examples (one per observer perspective), always
canonicalized so the observer plays white moving up. Examples with no hidden
squares are dropped (nothing to predict). Stored compactly as class indices;
plane expansion happens at training/inference time via encoding.build_inputs.

Usage:
    .venv/Scripts/python boardgen/dataset.py --replays data/selfplay --out data/dataset.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoding import HistoryTracker, UNKNOWN, board_string_to_classes, canonicalize


def game_examples(replay, game_index):
    rows = []
    trackers = {True: HistoryTracker(True), False: HistoryTracker(False)}
    for frame in replay['frames']:
        ply = frame['ply']
        target_raw = board_string_to_classes(frame['board'])
        for is_white in (True, False):
            raw = frame['white' if is_white else 'black']
            obs_raw = trackers[is_white].update(raw)
            obs, last_seen, since = trackers[is_white].snapshot(obs_raw)
            if not (obs == UNKNOWN).any():
                continue
            target = canonicalize(target_raw[None], is_white)[0]
            to_move = (ply % 2 == 0) == is_white
            rows.append((obs, last_seen, since, ply, to_move, target, game_index))
    return rows


def build(replay_paths, out_path):
    all_rows = []
    skipped = 0
    for game_index, path in enumerate(replay_paths):
        try:
            replay = json.loads(Path(path).read_text())
            frames = replay.get('frames')
            if not frames:
                skipped += 1
                continue
            all_rows.extend(game_examples(replay, game_index))
        except (json.JSONDecodeError, KeyError) as error:
            print(f'skipping {path}: {error}')
            skipped += 1
    if not all_rows:
        raise SystemExit('No examples produced; check --replays')
    obs, last_seen, since, ply, to_move, target, game = (np.array(column) for column in zip(*all_rows))
    np.savez_compressed(
        out_path,
        obs=obs.astype(np.uint8), last_seen=last_seen.astype(np.uint8),
        since=since.astype(np.uint8), ply=ply.astype(np.uint16),
        to_move=to_move.astype(np.uint8), target=target.astype(np.uint8),
        game=game.astype(np.uint32))
    hidden_per_example = (obs == UNKNOWN).sum(axis=1)
    print(f'{len(obs)} examples from {len(replay_paths) - skipped} games ({skipped} skipped)')
    print(f'hidden squares per example: mean {hidden_per_example.mean():.1f}, '
          f'min {hidden_per_example.min()}, max {hidden_per_example.max()}')
    print(f'wrote {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays', type=Path, nargs='+', required=True,
                        help='Replay JSON files or directories containing them')
    parser.add_argument('--out', type=Path, default=Path('data/dataset.npz'))
    args = parser.parse_args()
    paths = []
    for entry in args.replays:
        paths.extend(sorted(entry.glob('*.json')) if entry.is_dir() else [entry])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(paths, args.out)


if __name__ == '__main__':
    main()
