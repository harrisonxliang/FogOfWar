"""Throttled, cached, resumable client for the public arena replay API.

API notes (https://www.mit.edu/~6.7980/fow/api.html, checked 2026-11-24):
  GET /api/games?limit=100&sort=played&order=desc[&cursor=...]
      -> {games: [row], nextCursor, summary (first page only), playing, ...}
      row: id, white, black, whitePlayerId, blackPlayerId, whiteVersion,
           blackVersion, result, reason, plies, createdAt, ...
  GET /api/games/{id}   (id matches ^[a-z0-9]{8}$)
      -> Replay: the row's fields plus frames, rulesVersion, timeControl.
      frames[i] = {ply, board, white, black, move, clocks}: identical to
      referee.make_frame, so no adapter is needed.
Rate limits: the page asks clients for <= 1 request/s, sequential, with
exponential backoff on 429/503 and never retrying before Retry-After.

Layout under --root (default data/raw/):
  index.jsonl          one listing row per line (newest first), deduplicated
  index_state.json     cursor checkpoint so listing resumes where it stopped
  games/{id}.json.gz   one replay per file; never fetched twice

    .venv/bin/python -m belief.download --max-games 30000
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = 'https://6s890.lids.mit.edu'
ROOT = Path(__file__).resolve().parent.parent
ID_RE = re.compile(r'^[a-z0-9]{8}$')
# Normal game endings. Anything else (time forfeit, illegal move, crash, ...)
# is a bot failure; such games are skipped by default.
NORMAL_REASONS = {'King capturable', 'Threefold repetition', '50-move rule',
                  'No legal moves', 'Move limit reached'}
USER_AGENT = 'fow-belief-dataset/1.0 (course project; throttled)'


class Client:
    def __init__(self, min_interval: float = 1.0, max_retries: int = 5):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def _wait(self):
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()

    def get_json(self, path: str, params: dict | None = None):
        url = BASE + path + ('?' + urllib.parse.urlencode(params) if params else '')
        for attempt in range(self.max_retries + 1):
            self._wait()
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT,
                                                       'Accept': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as err:
                if err.code in (429, 503) and attempt < self.max_retries:
                    retry_after = float(err.headers.get('Retry-After') or 0)
                    backoff = max(retry_after, 2 ** attempt) + random.uniform(0, 1)
                    print(f'  HTTP {err.code} on {path}; sleeping {backoff:.1f}s', file=sys.stderr)
                    time.sleep(backoff)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as err:
                if attempt < self.max_retries:
                    backoff = 2 ** attempt + random.uniform(0, 1)
                    print(f'  {err!r} on {path}; sleeping {backoff:.1f}s', file=sys.stderr)
                    time.sleep(backoff)
                    continue
                raise
        raise RuntimeError('unreachable')


def eligible(row: dict, min_plies: int, include_failures: bool) -> bool:
    if not ID_RE.match(str(row.get('id', ''))):
        return False
    if row.get('result') not in ('1-0', '0-1', '1/2-1/2'):
        return False
    if (row.get('plies') or 0) < min_plies:
        return False
    return include_failures or row.get('reason') in NORMAL_REASONS


def load_index(root: Path) -> list[dict]:
    path = root / 'index.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def update_index(client: Client, root: Path, want: int, min_plies: int, include_failures: bool,
                 page_size: int = 100) -> list[dict]:
    """Page through the archive (newest first) until `want` eligible rows are indexed."""
    rows = load_index(root)
    seen = {r['id'] for r in rows}
    state_path = root / 'index_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    cursor = state.get('cursor')
    if state.get('exhausted'):
        return rows
    n_ok = sum(eligible(r, min_plies, include_failures) for r in rows)
    with open(root / 'index.jsonl', 'a') as out:
        while n_ok < want:
            params = {'limit': page_size, 'sort': 'played', 'order': 'desc'}
            if cursor:
                params['cursor'] = cursor
            try:
                page = client.get_json('/api/games', params)
            except urllib.error.HTTPError as err:
                if err.code == 400 and cursor:
                    # Stale cursor: the docs say to restart the search. Rows are
                    # deduplicated by id, so restarting only costs requests.
                    print('  cursor rejected; restarting listing from the newest page', file=sys.stderr)
                    cursor = None
                    continue
                raise
            if page.get('summary'):
                state['summary'] = page['summary']
            for row in page.get('games', []):
                if row.get('id') in seen:
                    continue
                seen.add(row['id'])
                rows.append(row)
                out.write(json.dumps(row) + '\n')
                n_ok += eligible(row, min_plies, include_failures)
            out.flush()
            cursor = page.get('nextCursor')
            state['cursor'] = cursor
            state_path.write_text(json.dumps(state))
            print(f'  indexed {len(rows)} rows, {n_ok} eligible', file=sys.stderr)
            if not cursor:
                state['exhausted'] = True
                state_path.write_text(json.dumps(state))
                break
    return rows


def fetch_replays(client: Client, root: Path, ids: list[str]) -> None:
    games = root / 'games'
    games.mkdir(parents=True, exist_ok=True)
    todo = [g for g in ids if not (games / f'{g}.json.gz').exists()]
    print(f'{len(ids) - len(todo)} replays cached, {len(todo)} to fetch', file=sys.stderr)
    t0 = time.monotonic()
    for k, game_id in enumerate(todo, 1):
        try:
            replay = client.get_json(f'/api/games/{game_id}')
        except urllib.error.HTTPError as err:
            if err.code == 404:
                print(f'  {game_id}: 404, skipping', file=sys.stderr)
                continue
            raise
        tmp = games / f'{game_id}.json.gz.tmp'
        with gzip.open(tmp, 'wt') as f:
            json.dump(replay, f)
        tmp.replace(games / f'{game_id}.json.gz')
        if k % 100 == 0 or k == len(todo):
            rate = k / (time.monotonic() - t0)
            eta = (len(todo) - k) / max(rate, 1e-9) / 3600
            print(f'  fetched {k}/{len(todo)} ({rate:.2f}/s, ETA {eta:.1f} h)', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT / 'data' / 'raw')
    ap.add_argument('--max-games', type=int, default=30000, help='eligible games to download, newest first')
    ap.add_argument('--min-plies', type=int, default=2)
    ap.add_argument('--include-failures', action='store_true',
                    help='also keep games that ended in a bot failure (forfeit, crash, illegal move)')
    ap.add_argument('--interval', type=float, default=1.0, help='seconds between requests (API asks >= 1)')
    ap.add_argument('--index-only', action='store_true')
    args = ap.parse_args()
    if args.interval < 1.0:
        ap.error('the API documentation asks for at most one request per second')
    args.root.mkdir(parents=True, exist_ok=True)
    client = Client(args.interval)
    rows = update_index(client, args.root, args.max_games, args.min_plies, args.include_failures)
    ok = [r['id'] for r in rows if eligible(r, args.min_plies, args.include_failures)][:args.max_games]
    reasons = {}
    for r in rows:
        if not eligible(r, args.min_plies, args.include_failures):
            reasons[r.get('reason')] = reasons.get(r.get('reason'), 0) + 1
    print(f'index: {len(rows)} rows, {len(ok)} selected; skipped by reason: '
          f'{dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:10])}', file=sys.stderr)
    if not args.index_only:
        fetch_replays(client, args.root, ok)


if __name__ == '__main__':
    main()
