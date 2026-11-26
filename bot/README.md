# bot/ — planner, method configs, protocol entry (David / P3)

The playing bot. Services live in `boardeval/` (evaluator, reply rule,
weighting, veto/aggregation); this package is the turn orchestration that
makes them a bot. Design decisions in `boardeval/NOTES.md`.

| File | What it owns |
| --- | --- |
| `planner.py` | The turn: fallback FIRST, then anytime widening rounds (sample worlds → shortlist → score (world, move) pairs → veto + aggregate). Incumbent replaced only at completed-round boundaries; partial rounds discarded (NOTES §5). β≥1 is the clairvoyant fast path (method A never touches the belief machinery). |
| `shortlist.py` | Deterministic candidate filter: cheap score (eval-after-their-best-reply) on every legal move, top-M survive + four guaranteed slots (hiding, scouting, captures-of-visible, king-safety). `eval_after_best_reply` is also the fallback's and fast path's core. |
| `belief_sources.py` | The `boards_for_reply` seam: `TrueBoardSource` (live, method A), `NaiveBoardSource` (needs P2 opp views, Nov 30+), `DiffusionSource` (needs P1 sampler, Dec 1). Unwired seams raise, never silently approximate. |
| `configs.py` | Methods A/N/C/B/D as frozen config dataclasses — every mode-registry axis is a named field. B/D raise until `own_worlds` exists. |
| `stub_tracker.py` | Single-world tracker until P2's enumerator: exact legal moves (reuses `uniform_random.ObservationTracker`), last-seen world reconstruction (shared with the naive X1 contestant), **both kings guaranteed** (a kingless world blinds the veto — regression-tested), approximate danger flag. |
| `entry.py` | Protocol main: stdin/stdout loop the referee speaks. `--method A|N|C`, `--seed`. Budget: `min(remaining/30 + 0.8·inc, remaining/4)`. |
| `local_match.py` | Windows-friendly match runner — referee.py is POSIX-only at the process layer (select on pipes, killpg), so this reuses the referee's *rules* verbatim and replaces only the plumbing. Same protocol, clocks, replay schema. Colors swap per game. |
| `test_planner.py` | 10 tests: registry + unwired seams, shortlist structure/slots, terminal shortcut, fallback-first, determinism, partial-round discard, stub kings, fallback veto. |

## Run

```sh
.venv/Scripts/python bot/test_planner.py

# Local games (colors swap each game):
.venv/Scripts/python bot/local_match.py \
    --white ".venv\Scripts\python.exe bot/entry.py --method A" \
    --black ".venv\Scripts\python.exe baseline.py --seed 5" \
    --games 4 --base-time 180 --increment 2
```

## Status (Nov 26)

Method A plays full protocol games end to end. First results, 60+1:
**4–0 vs baseline.py, 2–0 vs uniform_random**, both colors, no timeouts,
no illegal moves. M2 gate next: 50 clean games at 180+2, then the arena
submission with Christine (and the protocol/time-manager boundary resolved).

Swap-in points: P2's tracker replaces `StubTracker` (same call sites);
P1's sampler + P2's opp views light up methods C and N via `belief_sources`.
