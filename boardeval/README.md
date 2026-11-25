# boardeval — the decision layer + X1 harness (David)

Everything that turns boards into decisions and results. Consumes boards from
the enumerator (Christine) and the belief model (Harrison, `boardgen/`); all
four bots in the progression run through this layer.

## Files

| File | What it owns |
| --- | --- |
| `evaluator.py` | Position grading: `material_eval` (material + FoW king danger, µs) and `StockfishEvaluator` (depth-1, guarded fallback for orthodox-illegal FoW states). |
| `reply.py` | Opponent-move prediction, one interface, four contestants: `clairvoyant`, `naive_last_seen` (never-seen squares keep their starting piece), `like_me` (J diffusion-believed boards), `uniform_random`. Shared 1-ply decision rule so X1 isolates beliefs. |
| `weighting.py` | ε-mixed belief-weighted world sampling (`p = (1−ε)·softmax(logw/τ) + ε/N`), never weights when |S| ≤ K. Metrics: truth-rank, truth-in-sample, ESS. |
| `decide.py` | Aggregation `(1−λ)·weighted_mean + λ·worst`, king-safety veto by losing-probability-mass with least-mass fallback when everything is vetoed. |
| `x1.py` | The X1 harness: replays → per-ply predictions from every contestant → top-1/top-3/MRR table. Works on self-play and arena replays (same schema). |
| `test_boardeval.py` | Edge-case tests (veto-all fallback, ε floor, λ tilt, king dynamics). |

## Run

```sh
python boardeval/test_boardeval.py
python boardeval/x1.py --replays data/selfplay --checkpoint checkpoints/v1/belief.pt --games 50
python boardeval/x1.py --replays <arena-replay-dir> --evaluator stockfish ...   # when replays + SF land
```

## Current smoke numbers (self-play, material evaluator, v0 belief checkpoint)

157 plies / 5 games: uniform .03 / naive .16 / clairvoyant .21 / like-me .21 top-1.
Clairvoyant ≈ like-me is expected on random-policy movers — separation (or not)
on REAL opponents is the X1 result. These numbers validate plumbing, nothing else.

## Interfaces for teammates

- **Harrison**: `like_me` consumes any predictor with `__call__(planes) -> logits`
  (torch or ONNX, see `boardgen/sample.py`). New checkpoints drop in via
  `--checkpoint`; nothing here changes when the model improves (7-class head
  will need a small decode update in `boardgen/sample.py`, flag me).
- **Christine**: `weighting.sample_worlds(fens, k, rng, predictor, state, ...)`
  expects the enumerated set as FEN board fields. Plug the tracker's set in and
  the weighted-vs-uniform ablation runs itself.

## Still to do

- BC baseline (`bc.py`): view → move policy net for X1.
- Stockfish binary locally + offline deeper reply rule for the real X1 runs.
- Dial tuning (λ, ε, τ, veto threshold) — real replays only, don't tune on self-play.
- Per-team X1 breakdown once arena replays exist.
