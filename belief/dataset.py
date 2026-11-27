"""Replays (frame format of referee.make_frame) -> .npz training shards.

This is the only module that reads the true board: to make targets and to
check that ViewTracker's viewer-only features agree with the truth.

One example = (game, frame index t, viewer). Frame t is the position after t
plies; the viewer receives that frame's observation. Both viewers, every frame.

Per example (canonical orientation, see features.py):
  cur, last, age  uint8 [64]   features from ViewTracker
  tgt             int8  [64]   0 empty / 1-6 enemy P..K on unseen squares, -100 elsewhere
  inv             uint8 [7]    enemy inventory
  ply, to_move    uint8
  metadata: game (int32 index into meta.json games), team_white/team_black
  (int16 index into meta.json teams), viewer (0 white / 1 black), t (int16),
  promo (1 if the enemy has promoted by frame t), exceeds (1 if targets need
  more of some piece type than inv allows).

Splits: by held-out teams (val/test = 10%/10% of teams) when there are
enough teams; otherwise by whole games. Never within a game. A game goes to
test if either team is a test team, else val if either is a val team.

    .venv/bin/python -m belief.dataset --src data/selfplay --out data/shards/selfplay
    .venv/bin/python -m belief.dataset --src data/raw/games --out data/shards/arena
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from belief.features import (ENEMY0, INV_START, OWN0, UNSEEN, ViewTracker, canonical_cur, inv_slot)

ROOT = Path(__file__).resolve().parent.parent
IGNORE = -100
ARRAYS = ('cur', 'last', 'age', 'tgt', 'inv', 'ply', 'to_move',
          'game', 'team_white', 'team_black', 'viewer', 't', 'promo', 'exceeds')


class FeatureCheckError(AssertionError):
    pass


def load_replay(path: Path) -> dict:
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def true_enemy_counts(canon_board: np.ndarray) -> np.ndarray:
    """canonical cur-style codes of the full board -> enemy inventory counts [7]."""
    counts = np.zeros(7, dtype=np.int16)
    for sq in np.flatnonzero(canon_board >= ENEMY0):
        counts[inv_slot(int(canon_board[sq]) - ENEMY0 + 1, int(sq))] += 1
    return counts


def game_examples(replay: dict, strict: bool = True) -> tuple[dict[str, np.ndarray], Counter]:
    """All (t, viewer) examples of one game plus check statistics."""
    frames = replay['frames']
    stats = Counter()
    rows = {k: [] for k in ('cur', 'last', 'age', 'tgt', 'inv', 'ply', 'to_move', 'viewer', 't', 'promo', 'exceeds')}
    for viewer_is_white in (True, False):
        key = 'white' if viewer_is_white else 'black'
        tracker = ViewTracker(viewer_is_white)
        enemy_promoted = False
        for t, frame in enumerate(frames):
            if frame['ply'] != t:
                raise FeatureCheckError(f'frame {t} has ply {frame["ply"]}')
            if t > 0:
                move = frame['move']
                mover_white = (t % 2 == 1)          # ply t was made by White iff t is odd
                if mover_white == viewer_is_white:
                    tracker.own_move(move)
                elif len(move) == 5:
                    enemy_promoted = True
            tracker.observe(frame[key])
            f = tracker.features()
            board = canonical_cur(frame['board'], viewer_is_white)   # full board, never '?'

            # --- checks against the true board -----------------------------
            vis = f.cur != UNSEEN
            if not np.array_equal(f.cur[vis], board[vis]):
                raise FeatureCheckError(f'{replay.get("id")} t={t} {key}: visible squares differ from truth')
            hidden = ~vis
            own_hidden = hidden & (board >= OWN0) & (board < ENEMY0)
            if own_hidden.any():
                raise FeatureCheckError(f'{replay.get("id")} t={t} {key}: own piece on an unseen square')
            # every enemy piece is visible (with the right type, checked above) or a target
            tgt = np.full(64, IGNORE, dtype=np.int8)
            enemy_hidden = hidden & (board >= ENEMY0)
            tgt[hidden] = 0
            tgt[enemy_hidden] = board[enemy_hidden] - ENEMY0 + 1
            n_enemy_true = int((board >= ENEMY0).sum())
            n_enemy_seen = int((f.cur >= ENEMY0).sum()) + int((tgt > 0).sum())
            if n_enemy_true != n_enemy_seen:
                raise FeatureCheckError(f'{replay.get("id")} t={t} {key}: enemy piece unaccounted for')

            true_inv = true_enemy_counts(board)
            inv = f.inv.astype(np.int16)
            if enemy_promoted:
                stats['promo_examples'] += 1
                stats['promo_inv_mismatch'] += int(not np.array_equal(true_inv, inv))
            elif not np.array_equal(true_inv, inv):
                stats['inv_mismatch'] += 1
                if strict:
                    raise FeatureCheckError(
                        f'{replay.get("id")} t={t} {key}: inv {inv.tolist()} != true {true_inv.tolist()}')
            exceeds = bool((true_inv > inv).any())
            stats['exceeds_inv'] += exceeds
            stats['examples'] += 1
            stats['hidden_squares'] += int(hidden.sum())

            rows['cur'].append(f.cur)
            rows['last'].append(f.last)
            rows['age'].append(f.age)
            rows['tgt'].append(tgt)
            rows['inv'].append(f.inv)
            rows['ply'].append(f.ply)
            rows['to_move'].append(f.to_move)
            rows['viewer'].append(0 if viewer_is_white else 1)
            rows['t'].append(t)
            rows['promo'].append(int(enemy_promoted))
            rows['exceeds'].append(int(exceeds))
    out = {
        'cur': np.stack(rows['cur']).astype(np.uint8),
        'last': np.stack(rows['last']).astype(np.uint8),
        'age': np.stack(rows['age']).astype(np.uint8),
        'tgt': np.stack(rows['tgt']).astype(np.int8),
        'inv': np.stack(rows['inv']).astype(np.uint8),
        'ply': np.array(rows['ply'], dtype=np.uint8),
        'to_move': np.array(rows['to_move'], dtype=np.uint8),
        'viewer': np.array(rows['viewer'], dtype=np.uint8),
        't': np.array(rows['t'], dtype=np.int16),
        'promo': np.array(rows['promo'], dtype=np.uint8),
        'exceeds': np.array(rows['exceeds'], dtype=np.uint8),
    }
    return out, stats


def _game_meta(replay: dict, path: Path) -> dict:
    return {
        'id': replay.get('id') or path.name.split('.')[0],
        'white_team': str(replay.get('whitePlayerId') or replay.get('white') or 'unknown'),
        'black_team': str(replay.get('blackPlayerId') or replay.get('black') or 'unknown'),
        'white_name': replay.get('white'), 'black_name': replay.get('black'),
        'white_version': replay.get('whiteVersion'), 'black_version': replay.get('blackVersion'),
        'result': replay.get('result'), 'reason': replay.get('reason'),
        'plies': len(replay['frames']) - 1, 'created': replay.get('createdAt'),
    }


def _work(path_str: str, strict: bool):
    path = Path(path_str)
    try:
        replay = load_replay(path)
        meta = _game_meta(replay, path)
        if replay.get('rulesVersion') not in (None, 'fow-v1'):
            return meta, None, Counter(skipped_rules=1), None
        arrays, stats = game_examples(replay, strict)
        return meta, arrays, stats, None
    except FeatureCheckError as err:
        return {'id': path.name}, None, Counter(failed_games=1), str(err)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as err:
        return {'id': path.name}, None, Counter(bad_games=1), f'{path.name}: {err!r}'


def assign_splits(metas: list[dict], val_frac: float, test_frac: float, seed: int,
                  split_by: str, min_teams: int = 10) -> tuple[list[str], dict]:
    teams = sorted({m['white_team'] for m in metas} | {m['black_team'] for m in metas})
    info = {'split_by': split_by, 'n_teams': len(teams)}
    if split_by == 'auto':
        split_by = 'team' if len(teams) >= min_teams else 'game'
        info['split_by'] = split_by
    if split_by == 'team':
        rng = random.Random(seed)
        order = teams[:]
        rng.shuffle(order)
        n_val = max(1, round(val_frac * len(teams)))
        n_test = max(1, round(test_frac * len(teams)))
        test_teams, val_teams = set(order[:n_test]), set(order[n_test:n_test + n_val])
        info.update(test_teams=sorted(test_teams), val_teams=sorted(val_teams))
        splits = []
        for m in metas:
            pair = {m['white_team'], m['black_team']}
            splits.append('test' if pair & test_teams else 'val' if pair & val_teams else 'train')
    else:
        splits = []
        for m in metas:
            u = int(hashlib.sha1(f'{seed}:{m["id"]}'.encode()).hexdigest()[:8], 16) / 2**32
            splits.append('test' if u < test_frac else 'val' if u < test_frac + val_frac else 'train')
    return splits, info


def load_shards(root: Path | str, split: str, max_examples: int | None = None) -> dict[str, np.ndarray]:
    """Concatenate all shards of one split into a dict of arrays."""
    files = sorted((Path(root) / split).glob('shard_*.npz'))
    if not files:
        raise FileNotFoundError(f'no shards in {Path(root) / split}')
    parts = {k: [] for k in ARRAYS}
    n = 0
    for f in files:
        with np.load(f) as z:
            for k in ARRAYS:
                parts[k].append(z[k])
            n += len(z['t'])
        if max_examples and n >= max_examples:
            break
    out = {k: np.concatenate(v) for k, v in parts.items()}
    if max_examples:
        out = {k: v[:max_examples] for k, v in out.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, nargs='+', required=True, help='dirs of *.json / *.json.gz replays')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-games', type=int, default=None)
    ap.add_argument('--split-by', choices=('auto', 'team', 'game'), default='auto')
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--test-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shard-examples', type=int, default=500_000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-strict', action='store_true', help='count inventory mismatches instead of failing')
    args = ap.parse_args()

    paths = sorted(p for d in args.src for p in list(d.glob('*.json')) + list(d.glob('*.json.gz')))
    if args.max_games:
        paths = paths[:args.max_games]
    print(f'{len(paths)} replay files', file=sys.stderr)

    results = []
    stats = Counter()
    errors = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (meta, arrays, st, err) in enumerate(
                pool.map(_work, map(str, paths), [not args.no_strict] * len(paths), chunksize=16), 1):
            stats.update(st)
            if err:
                errors.append(err)
            if arrays is not None:
                results.append((meta, arrays))
            if k % 1000 == 0:
                print(f'  {k}/{len(paths)} games', file=sys.stderr)
    for e in errors[:20]:
        print('  ERROR', e, file=sys.stderr)
    if stats['failed_games']:
        print(f'{stats["failed_games"]} games failed feature checks', file=sys.stderr)

    metas = [m for m, _ in results]
    splits, split_info = assign_splits(metas, args.val_frac, args.test_frac, args.seed, args.split_by)
    teams = sorted({m['white_team'] for m in metas} | {m['black_team'] for m in metas})
    team_idx = {t: i for i, t in enumerate(teams)}

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ('train', 'val', 'test'):
        (args.out / split).mkdir(exist_ok=True)
        for old in (args.out / split).glob('shard_*.npz'):
            old.unlink()
        buf, n_buf, shard, n_total, n_games = {k: [] for k in ARRAYS}, 0, 0, 0, 0
        for gi, ((meta, arrays), s) in enumerate(zip(results, splits)):
            if s != split:
                continue
            n = len(arrays['t'])
            for k, v in arrays.items():
                buf[k].append(v)
            buf['game'].append(np.full(n, gi, dtype=np.int32))
            buf['team_white'].append(np.full(n, team_idx[meta['white_team']], dtype=np.int16))
            buf['team_black'].append(np.full(n, team_idx[meta['black_team']], dtype=np.int16))
            n_buf += n
            n_games += 1
            if n_buf >= args.shard_examples:
                np.savez(args.out / split / f'shard_{shard:05d}.npz', **{k: np.concatenate(v) for k, v in buf.items()})
                n_total += n_buf
                shard, n_buf, buf = shard + 1, 0, {k: [] for k in ARRAYS}
        if n_buf:
            np.savez(args.out / split / f'shard_{shard:05d}.npz', **{k: np.concatenate(v) for k, v in buf.items()})
            n_total += n_buf
        summary[split] = {'games': n_games, 'examples': n_total}

    ex = max(stats['examples'], 1)
    report = {
        'src': [str(s) for s in args.src], 'n_files': len(paths), 'splits': summary, 'split_info': split_info,
        'checks': {
            **{k: int(v) for k, v in stats.items()},
            'inv_mismatch_rate_excl_promo': stats['inv_mismatch'] / max(ex - stats['promo_examples'], 1),
            'promo_example_rate': stats['promo_examples'] / ex,
            'exceeds_inv_rate': stats['exceeds_inv'] / ex,
            'mean_hidden_squares': stats['hidden_squares'] / ex,
        },
        'errors': errors[:200],
    }
    (args.out / 'meta.json').write_text(json.dumps({**report, 'teams': teams, 'games': metas}))
    print(json.dumps(report, indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
