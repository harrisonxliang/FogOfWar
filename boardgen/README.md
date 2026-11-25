# boardgen — opponent-belief board generation (Person C)

Masked-diffusion model over the 64×13 board grid that predicts the hidden
squares of the true board from our observation history. One network serves
both roles from the task split:

- **`score(fen, obs_state) -> log-weight`** — Person B's learned weight
  function (one forward pass total, regardless of candidate count).
- **`generate(obs_state, n) -> boards`** — board generation for when |P|
  exceeds the enumerator's cap (iterative unmasking, MaskGIT-style).

Evaluated single-pass with everything masked, the model *is* the per-square
"model v1" from the plan; multi-step unmasking is v2. Same checkpoint, so the
v1-vs-v2 comparison (truth-in-sample at steps=1 vs 4 vs 8) is a free ablation.

## Pipeline

```sh
# 1. Self-play data (in-process, no subprocesses — works on Windows; ~56 games/s)
python boardgen/generate_games.py --games 8000 --out data/selfplay --workers 8

# 2. Replays -> compact tensors (also accepts downloaded arena replay JSONs)
python boardgen/dataset.py --replays data/selfplay --out data/dataset.npz

# 3. Train (CPU ok; --device cuda on Colab)
python boardgen/train.py --data data/dataset.npz --epochs 10 --out checkpoints/v1

# 4. Correctness smoke test
python boardgen/test_smoke.py --checkpoint checkpoints/v1/belief.pt --replay data/selfplay/game-000000.json

# 5. ONNX export + arena speed budget check (2 threads)
python boardgen/bench.py --checkpoint checkpoints/v1/belief.pt --out checkpoints/v1/belief.onnx
```

## Files

| File | Purpose |
| --- | --- |
| `generate_games.py` | In-process self-play using the referee's exact rules code; emits arena-replay-schema JSON. Policies (uniform / greedy / mixed) see only their own observation. |
| `dataset.py` | Replay JSONs → `dataset.npz` (compact class indices; train/val split by game). |
| `encoding.py` | Shared square/piece encoding, canonicalization (observer always plays white moving up), history tracking, plane building. Torch-free — the arena bot reuses it with numpy + ONNX Runtime only. |
| `model.py` | `BeliefNet` CNN (~1M params at 96ch/6blocks; 323k at 64ch/4blocks). |
| `train.py` | Masked-diffusion training; validation reports full-masking log-loss/accuracy vs a predict-empty baseline. |
| `sample.py` | `BeliefState`, `score()`, `generate()`, torch/ONNX predictors. |
| `bench.py` | ONNX export with numerical verification + speed benchmark. |

## Contract with Person B (sampler)

`weight_fn(fen: str, obs: Observation) -> float` in log-space. Adapter:
build a `BeliefState`, feed it every raw 64-char observation line we have
received this game (in ply order), then call
`score(predictor, state, [fen_board_field, ...])` → `(N,)` log-weights.
Scoring any number of candidates costs **one** forward pass (~26 ms for 2,000
boards), so there is no reason to subsample before weighting.

## Integration with Misty (pending repo access)

- `generate()` takes a `filter_fn(fen_board) -> bool` hook — wire Misty's
  `BeliefHardFacts` checks there and report the acceptance rate (it's a result
  for the write-up).
- Downsampling: in `_maybe_downsample` / `_rust_downsample`, replace random
  truncation with generated-and-validated boards kept ∝ model probability.

## Speed on 2 CPU threads (local measurements, untrained-weights valid)

| Operation | 64ch/4bl | 96ch/6bl | Budget |
| --- | --- | --- | --- |
| score 2,000 boards | 26 ms | 26 ms | 500 ms |
| generate 256, 4 steps | 195 ms | ~1.4 s | 500 ms |

The first unmasking step shares one forward pass across the whole batch.
If the big model's accuracy is worth it, run it at `n=128, steps=4` or ship
the small one for generation and the big one for scoring — scoring cost is
independent of model size at these scales.

## Known gaps / next

- Dataset realism: self-play policies are weak randomized heuristics; mix in
  arena replays (same schema, just drop the files into `--replays`) and
  Misty self-play once the fork is available.
- `score()` scores the *first* FEN field only (piece placement); castling
  rights / en passant are not modeled.
- Quantization (int8) is an option if the arena box is slower than local.
