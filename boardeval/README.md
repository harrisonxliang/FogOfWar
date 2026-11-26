# boardeval — the decision layer + X1 harness (David)

Everything that turns boards into decisions and results. Consumes boards from
the enumerator (Christine) and the belief model (Harrison, `boardgen/`); all
five method configs (A/N/C/B/D) run through this layer. Design decisions and
the mode registry live in [NOTES.md](NOTES.md).

## Files

| File | What it owns |
| --- | --- |
| `evaluator.py` | Position grading. `terminal_score` (king capture, always checked first), `material_eval` (material + PST + king safety + optional exact per-world fog terms, µs), `tanh_eval` ([-1,1] wrapper, ±1 terminal), `StockfishEvaluator` (depth-1 w/ qsearch; guards: terminal-first, illegal-FEN fallback, dead-engine mode flip, per-turn FEN cache), `get_eval_config` (handcrafted / sf-hybrid / sf-full — reply-side vs truth-side split). |
| `reply.py` | Opponent-move prediction. Contestants `clairvoyant`, `naive_last_seen`, `like_me` (J diffusion-believed boards), `uniform_random` share one 1-ply rule so X1 isolates beliefs. `to_distribution` (softmax at τ + uniform smoothing floor → p(r), LL never −∞), `mix_with_clairvoyant` (β mix, β=1 = method A's opponent). |
| `weighting.py` | Settled design (NOTES §3): score S with the belief model → ε-mix w̃ → select K (modes: `weighted-sample` default / `top-k` / `uniform`) → return renormalized-within-sample `agg_weights` + diagnostics (captured mass, ESS). `veto_weights` = the conservative ε_veto mix for the full-S veto. Metrics: truth-rank, truth-in-sample, ESS, captured mass. |
| `decide.py` | Aggregation `(1−λ)·weighted_mean + λ·worst` (feed `agg_weights`), sample-level mass veto with least-mass fallback, `loss_threshold` param (cp or tanh scale), `veto_mass_full_s` (ε_veto-mixed danger mass over ALL of S — the tracker seam). |
| `x1.py` | X1 harness: replays → per-ply predictions from every contestant → top-1/top-3/MRR/**log-likelihood** table (`--tau`, `--floor`). Works on self-play and arena replays (same schema). |
| `test_boardeval.py` | 14 tests per the NOTES §6 plan: veto/λ/threshold scales, full-S veto blend+floor, ε floor, selection modes + renormalized weights + danger include, reply distribution + β mix, terminal precedence, tanh bounds, mirror symmetry (fog active), fog-term toggles. |

## Run

```sh
.venv/Scripts/python boardeval/test_boardeval.py
.venv/Scripts/python boardeval/x1.py --replays data/selfplay --checkpoint checkpoints/v1/belief.pt --games 50
.venv/Scripts/python boardeval/x1.py --replays <arena-replays> --evaluator stockfish --tau 150 --floor 0.02
```

## Current smoke numbers (self-play, material evaluator, v0 belief checkpoint)

1,423 plies / 50 games: uniform .063 / naive .150 / clairvoyant .244 / like-me .244 top-1.
Clairvoyant ≈ like-me is the expected null on self-play movers — separation (or not)
on REAL opponents is the X1 result. These numbers validate plumbing, nothing else.

## Interfaces for teammates

- **Harrison**: `like_me` consumes any predictor with `__call__(planes) -> logits`
  (torch or ONNX, see `boardgen/sample.py`); `weighting` uses the same predictor's
  `score()` for world weights. New checkpoints drop in via `--checkpoint`.
- **Christine**: `weighting.sample_worlds(fens, k, rng, predictor, state, ...)`
  now returns `(indices, agg_weights, diag)` — pass the tracker's enumerated set
  in, feed `agg_weights` to `decide.aggregate`, and `decide.veto_mass_full_s`
  wants per-world danger booleans over ALL of S for the δ veto. We decided to
  WEIGHT worlds (announced; the proposal had it as an open question).

## Still to do

- Planner (M2, Nov 28): shortlist w/ guaranteed slots, PIMC fallback first,
  anytime loop, protocol entry point. Clarify protocol/time-manager boundary w/ P2.
- SF gates before relying on sf-hybrid: UCI round-trip benchmark in the 2-CPU
  container, binary policy, init budget (NOTES §1).
- BC baseline (`bc.py`): view → move policy net for X1.
- ε sweep + (ε_veto, δ) grid on real replays (NOTES §3); τ fit per team; β sweep.
- Per-team X1 breakdown + like-me-exact contestant once arena replays and P2's
  enumerator exist. Tune dials on real replays only.
