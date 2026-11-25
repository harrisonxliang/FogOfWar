"""Generate local self-play games with the starter bots (smoke-test data).

Writes one replay JSON per game to data/selfplay/, in the referee's frame format,
plus the same team/version metadata keys the arena replays carry so the dataset
builder treats both sources identically.

    .venv/bin/python -m belief.selfplay --games 200 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BOTS = {
    'baseline': 'baseline.py',
    'uniform': 'uniform_random.py',
}


def play_one(index: int, white: str, black: str, out_dir: str, max_plies: int) -> str:
    from referee import run_match
    py = sys.executable
    w_argv = [py, '-u', str(ROOT / BOTS[white]), '--seed', str(2 * index + 1)]
    b_argv = [py, '-u', str(ROOT / BOTS[black]), '--seed', str(2 * index + 2)]
    game_id = f'sp{index:06d}'
    replay = run_match(w_argv, b_argv, base=180, increment=2, match_id=game_id,
                       white_name=white, black_name=black, max_plies=max_plies)
    replay.pop('privateDiagnostics', None)
    replay.pop('protocol', None)
    replay.update(whitePlayerId=f'selfplay-{white}', blackPlayerId=f'selfplay-{black}',
                  whiteVersion=0, blackVersion=0, source='selfplay')
    path = Path(out_dir) / f'{game_id}.json'
    path.write_text(json.dumps(replay))
    return f'{game_id} {white} vs {black}: {replay["result"]} {replay["reason"]} ({len(replay["frames"]) - 1} plies)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'selfplay')
    ap.add_argument('--max-plies', type=int, default=600)
    ap.add_argument('--start', type=int, default=0, help='first game index (for appending more games)')
    ap.add_argument('--pairing', choices=('bu', 'uu', 'bb'), default='bu',
                    help='bu: baseline vs uniform, alternating colours (default); uu / bb: same bot both sides')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairings = {'bu': [('baseline', 'uniform'), ('uniform', 'baseline')],
                'uu': [('uniform', 'uniform')] * 2, 'bb': [('baseline', 'baseline')] * 2}[args.pairing]
    jobs = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i in range(args.start, args.start + args.games):
            if (args.out / f'sp{i:06d}.json').exists():
                continue
            w, b = pairings[i % 2]
            jobs.append(pool.submit(play_one, i, w, b, str(args.out), args.max_plies))
        for fut in as_completed(jobs):
            print(fut.result(), flush=True)


if __name__ == '__main__':
    main()
