# P3 design notes — things to be aware of and test

Running list for the decision layer (evaluator, reply rule, weighting, veto/aggregation, X1).
Decisions agreed Nov 25; update as they change. Companion to README.md.

---

## 0. Mode registry (every switch we're creating — keep this current)

Rule: every mode is a named config value, never a hidden code branch. A new mode gets a
row here, a config entry, and a test. **Bold = default.**

| Axis | Options | Purpose |
|---|---|---|
| Method (top level) | A / N / **C** / B / D | The five research configs; one file each, consumed by P2's harness |
| Evaluator | **handcrafted** / sf-hybrid / sf-full | sf-hybrid = SF `go depth 1` on true-board V only; sf-full offline-only; also the degradation rung (sf-hybrid → handcrafted under clock pressure) |
| World selection | **weighted-sample** / top-k / uniform | weighted-sample = WOR ∝ w̃ (Gumbel top-K); top-k = deterministic truncation by w̃ (purist estimator, no sampling noise); uniform = proposal default, ablation baseline |
| Aggregation | **renorm-weighted mean** / plain mean / +λ tilt | plain mean is the ablation pair for weighting; λ off by default |
| Veto | **ε-mixed mass (δ=1%)** / count / hard (δ=0) | count = ε_veto→1 limit; hard veto for ablation only |
| Reply belief source | TrueBoard / NaiveBoard / **Diffusion** | = the A/N/C axis; dials τ (softmax), β (clairvoyant mix) ride on top |
| ε | sampling ε / ε_veto | two separate knobs, fit separately (§3) |

---

## 1. Evaluator: two live modes + hybrid split

**Decision:** the bot ships two evaluator modes behind one `eval(board) -> [-1, 1]` interface:

- `handcrafted` — material (P100 N300 B300 R500 Q900) + PSTs + king safety + two fog
  terms, squashed by `tanh(cp/600)`; ±1 short-circuit on king capture. Microseconds/call.
- `sf-hybrid` — Stockfish `go depth 1` (NOT static eval — depth 1 includes qsearch, which
  resolves hanging pieces and replaces the hand-rolled capture-only quiescence) used ONLY
  for true-board scoring in V(x,a). Reply scoring on believed boards stays handcrafted.
- `sf-full` — offline/ablation only, too slow live.

**Why the split:** reply scoring (~2,400 evals/move: K×M pairs × ~30 replies × J boards) only
shapes p(r), which is smoothed by τ-softmax, β-mix, and averaging over J and K — eval errors
wash out, and the boards are diffusion guesses anyway. True-board scoring (~1,200 evals) is
unsmoothed and directly picks our move. Put the expensive evaluator where errors are
unsmoothed and calls are fewest. SF over UCI ≈ 0.3–1 ms/call → full substitution eats the
whole 2–5 s budget; hybrid costs ~0.4–1.2 s.

**Gates to clear BEFORE relying on sf-hybrid (ideally before M2, Nov 28):**
- [ ] Benchmark ~1,000 UCI round-trips (`position fen` + `go depth 1`) single-threaded in the
      2-CPU / 4 GiB Docker container. Pipe overhead on 2 cores is the unknown, not eval speed.
- [ ] Confirm shipping a Stockfish binary is allowed under the 5-minute offline-build rule,
      and engine spawn + NNUE load fits the 15 s init budget.
- [ ] Per-call timeout on the UCI pipe; a hung engine must cost one eval, never the move.

**Guards (mandatory, in this order):**
1. King-capturable check BEFORE any SF call → ±1. FoW positions are routinely illegal in
   standard chess (kings en prise/adjacent); SF behavior on illegal FENs is undefined.
2. Any SF rejection/crash/timeout → fall back to handcrafted for that position only.
3. SF process death mid-game → flip mode to handcrafted, don't crash.
4. Per-turn eval cache keyed on FEN so anytime-loop re-visits are free.

**Degradation ladder (time manager owns the switch):** sf-hybrid → handcrafted → PIMC
fallback move. SF mode is a runtime quality knob with a known ~1 s cost, not a build-time
choice. If any gate fails, handcrafted ships for M2 and SF stays offline-only.

**Handcrafted eval requirements:** side-symmetric (eval from white = −eval from black —
unit test this, spec in §2b), deterministic, defined on standard-chess-illegal positions,
µs-fast. Sweep the tanh scale over {300, 600, 1200}.

**Fog-term semantics (clarified Nov 25 — do NOT assume worst case inside the eval):**
the evaluator always receives a FULL-INFORMATION board (a sampled world x, or true x′).
Both fog terms are exact per-world computations:
- "our visibility" = given THIS world's enemy placement, compute their vision mask from the
  rules and count which of our key pieces fall inside it.
- "hidden attackers" = given this world, count attacks toward our king that originate from
  squares outside OUR vision mask (threats we wouldn't see coming in this world).
We never assume they could have pieces on every square we can't see — that uncertainty is
handled ABOVE the evaluator: averaging over K sampled worlds, and the δ-mass veto over all
of S. Pessimism belongs in the veto/λ tilt, not the eval; putting it in both double-counts
fear and makes the bot passive.

**Quiescence in handcrafted: WANT, not must — with a promotion trigger.** As long as
sf-hybrid clears its gates, qsearch comes free (SF `go depth 1`) on the true-board side,
which is the only place it matters much — the reply side is smoothed by τ/β/averaging.
It PROMOTES TO MUST if the SF gates fail (binary disallowed, or pipe overhead blows the
budget): handcrafted-everywhere with zero capture resolution hangs pieces, which is fatal
in FoW. Cheap middle option if promoted: static exchange evaluation (SEE) on the
destination square, or capture-only 1-ply on the moved piece — most of the benefit at a
fraction of qsearch cost. Decision point = the SF gate benchmark, before Nov 28.

---

## 2. Consistency checking (IMPORTANT)

### 2a. Hybrid eval consistency — the big one
In `sf-hybrid`, p(r) comes from handcrafted eval (believed boards) but the payoffs it weights
come from SF (true boards). Two definitions of "good" in one expectation. Failure mode: two
replies look equal to handcrafted (softmax gives ~0.4 each) but SF knows one crushes us —
expected values get computed with probabilities from the crude metric and payoffs from the
sharp one. Bites hardest near softmax boundaries, on tactics/king-safety positions — exactly
where SF-with-qsearch and a static hand eval diverge most.

- [ ] Offline on replay positions: compute p(r) under handcrafted vs SF on the SAME believed
      boards. Report top-1 agreement and KL divergence.
- [ ] X1 log-likelihood: sf-hybrid vs handcrafted-everywhere. If hybrid degrades, that's the
      red flag.
- [ ] One X7-style head-to-head: sf-hybrid vs handcrafted bot, does play strength move?
- Fixes in order of cheapness: raise τ (soften p(r)) → raise β (hedge to clairvoyant) →
  run SF on just the top few replies per pair to re-rank.
- Partially fine by design: the opponent isn't Stockfish, so a cruder lens on THEIR choice is
  defensible, and all five methods share the hybrid so A/N/C/B/D comparisons still isolate
  the belief source.

### 2b. Other consistency checks
- [ ] **Eval symmetry unit test (required):** for random boards, build the color-mirror —
      flip the board vertically, swap piece colors, flip side to move, mirror castling/ep
      state — and assert eval(mirror) == −eval(original) to float tolerance. Include boards
      with fog terms active (hidden attackers, visible key pieces) since those are the terms
      most likely to be accidentally asymmetric. Also assert eval(board, white) ==
      −eval(board, black) on the SAME board.
- [ ] SF-vs-handcrafted sign agreement on quiet positions (they should rarely disagree on who
      is better by ≥2 pawns; log the disagreements, they're PST bugs or SF-guard bugs).
- [ ] Correlated-error check: the SAME diffusion model weights our worlds (§3) and generates
      opponent beliefs (§4). Its errors hit both places at once. The ε floor exists for this;
      never remove it. Test: inject a deliberately miscalibrated model, confirm the true board
      still gets sampled (truth-in-sample stays > 0).
- [ ] Determinism: same observation history + same seed → same move, both modes.

---

## 3. Weighting (DECIDED: we weight; announce to team — proposal lists it as open)

Mechanism: belief model's `score()` — one fully-masked forward pass over OUR view, then
log w(x) = Σ_hidden log p(piece on square). One ONNX batch scores the candidate set
(~21 ms / 2,000 boards, 2 threads).

- Pipeline: candidates from tracker → score → normalize → **ε-mix with uniform
  (ε = 0.25–0.5, mandatory — see 2b)** → sample K without replacement
  (+ force-include one danger world when flagged).
- **SETTLED DESIGN (Nov 25): randomized truncation + renormalized weighted mean.**
  (a) score all S → ŵ; ε-mix → w̃ = (1−ε)ŵ + ε/|S|;
  (b) SELECT K without replacement ∝ w̃ (Gumbel top-K), force-include danger world;
  (c) AGGREGATE with w̃ renormalized within the sample:
      V(a) = Σ w̃(x)V(x,a) / Σ w̃(x) over picked worlds.
  Not double-counting: the double-count trap applies to with-replacement FREQUENCY
  sampling, where multiplicity already encodes weight. Without replacement at K=8 each
  board appears once, so frequency can't carry plausibility — selection just picks WHICH
  truncation of S we can afford; the weights act exactly once, inside it. The force-
  included danger world enters with its own w̃ like any other pick.
  Diagnostics per turn: captured mass (Σ w̃ over the K picked) and ESS.
  Writeup footnote: sample-then-renormalize is a mildly biased estimate of the full-S
  weighted mean (deterministic top-K is the purist alternative); sampling buys diversity
  against model error, which we value more.
- **ε determination:** 0.25–0.5 is the proposal's prior, not derived.
- [ ] **TODO — ε hyperparameter sweep (offline, held-out replays):** ε ∈
      {0, 0.1, 0.25, 0.5, 1}; maximize held-out log w̃(true board) and truth-in-sample@K;
      report captured-mass@K per ε. Never ship ε = 0 (the floor IS the safety property).
      Needs only true boards from replays — no tracker dependency.
- [ ] **TODO — ε_veto sweep, separate objective:** coarse grid (ε_veto, δ) ∈
      {0.25, 0.5, 1} × {0.5%, 1%, 2%} — they jointly set the firing condition. Metric:
      false-negative rate on moves that truly lose the king (exact from true board),
      subject to a false-positive (over-veto) budget on safe moves. DEPENDS on P2's exact
      sets at replay positions; if they slip, ship prior ε_veto = 0.5 and defer.
- **Sweep cost note:** model scoring (ŵ) is ε-independent — score replays ONCE, cache,
  re-blend per ε. Both sweeps are grid arithmetic on the same cached pass. The two ε's
  feed disjoint computations (selection/aggregation vs veto), so two independent 1D
  sweeps are sound — no joint ε×ε grid needed.
- **ε_veto logic:** the ε-mixed danger mass decomposes as
      mass(danger) = (1−ε)·ŵ(danger) + ε·(count(danger)/|S|),
  i.e. a blend of the model-mass veto and the old count veto, with guaranteed floor
  ε · count-share. The model can suppress the danger mass at most down to that floor.
  Veto errors are asymmetric: false negative (model hides a real king-loss world, move
  approved) loses the game; false positive (bogus danger world vetoes a fine move) costs
  a little strength. So ε_veto sits closer to 1 (count-like, model-proof) than sampling ε,
  which is tuned for estimate quality, not survival.
- Selection modes: **weighted-sample (default)** vs top-k (deterministic truncation) —
  see mode registry §0; both share w̃, aggregation, and diagnostics.
- Never weight SELECTION when |S| ≤ K (take everything) — but weights still feed:
- Weighted aggregation: V(a) = weight-weighted mean over worlds (λ tilt optional on top).
- Weighted veto: δ threshold on probability MASS of king-capture worlds, not count —
  one implausible world can no longer veto a good move. NOT double-weighting: the veto
  runs over all of S with no sampling step in front, so weights are used exactly once,
  as a measure. Use the ε-MIXED weights here, not raw model weights: the veto is the
  last line of defense, and a miscalibrated model must never be able to weight a real
  king-loss world down to invisibility (floor = ε/|S| mass per world).
- Caveat to state once in the writeup: Σ per-square log-probs assumes square independence —
  a pseudo-likelihood, fine for ranking, not a true joint probability.
- [ ] Metrics (weighting.py): truth-rank, truth-in-sample, ESS on replays.
- [ ] Ablation: uniform vs weighted in X1 + false-veto rate.
- [ ] Interface ask to P2: sample_worlds needs to expose candidates + view context to score.

---

## 4. Reply rule — the invariant

**Predict their reply on what they BELIEVE; settle the value on what's TRUE.**

- p(r): their ~30 legal replies on true x′ (they always know their own moves), each scored
  from THEIR side on the J believed boards, averaged over J, softmax at τ, β-mixed with
  their clairvoyant best reply on x′.
- NOT minimax vs their true-board optimum — that's clairvoyant (method A / β = 1).
  Their "best" move may be objectively bad; exploiting that is the thesis.
- [ ] Uniform smoothing floor on p(r) so X1 log-likelihood never hits −∞.
- [ ] τ is fit per team OFFLINE for X1 reporting; at play time we likely don't know the
      opponent (protocol open question) → need a pooled default τ.
- Belief source is the config switch: TrueBoard = A, NaiveBoard = N, Diffusion = C.

---

## 5. Latency / anytime budget (worked-example shape)

- Budget ~2–5 s/move. PIMC fallback computed FIRST — a move is always ready.
- Tracker update tens of ms; diffusion batch ~0.25 s; handcrafted evals ~15 ms;
  sf-hybrid adds ~0.4–1.2 s; spare time → widen the shortlist (anytime loop).
- [ ] Clock checks between diffusion batches and between eval blocks.
- [ ] Dec 1 integration gate: method C end-to-end in budget, diffusion step < ~0.5 s/move.
- **Planner status (Nov 26): BUILT** — `bot/` package (planner, shortlist, belief
  sources, configs, stub tracker, protocol entry, local match runner). Method A plays
  full protocol games: 4–0 vs baseline.py, 2–0 vs uniform_random at 60+1, both colors,
  no timeouts. Next: 50 clean games at 180+2 (M2 gate), arena submission with P2.
- **Windows note:** referee.py is POSIX-only at the process layer (select-on-pipes,
  killpg) — local games run through `bot/local_match.py`, which reuses the referee's
  rules verbatim and swaps only the plumbing. The arena's real referee runs the same
  `bot/entry.py` unchanged.
- **Stub-world lesson (regression-tested):** a reconstructed world missing the enemy
  king evaluates +WIN for every move → the veto goes blind → arbitrary king moves
  (observed: walked onto a remembered bishop's diagonal). Any world handed to the
  evaluator MUST contain both kings. Worth re-checking against P2's enumerator too —
  every enumerated world should carry both kings by construction; assert it anyway.
- **Anytime incumbent rules (planner):** the PIMC fallback is the floor, computed once;
  each COMPLETED widening round (bigger K/M) replaces the incumbent move; a stop signal
  plays the current incumbent. Two hard rules: (1) never adopt a half-finished round —
  a partial V matrix compares moves over different world sets, which is biased;
  (2) never compare scores ACROSS rounds — the widest completed round supersedes,
  even if its top score is numerically lower. Bonus: each round re-runs the veto with
  more worlds, so widening improves safety too. Test: stop mid-round → previous
  incumbent returned, not the partial round's argmax.

---

## 6. Test plan — thorough, per module (expand test_boardeval.py's current 6)

### evaluator.py
- [ ] Terminal precedence: king-capturable → exactly ±1 for both colors, and it fires
      BEFORE any other term (and before SF in hybrid mode). Construct a position that is
      simultaneously up a queen and has our king en prise; assert −1.
- [ ] Symmetry (spec in §2b).
- [ ] Material: hand-built positions with known cp (expose the raw cp pre-tanh for
      testability); each piece value; captures change cp by exactly the piece value.
- [ ] PST sanity: knight on rim < knight in center, all else equal; advancing a passed
      pawn monotonically increases score.
- [ ] Fog terms, constructed pairs: same position with an attacker inside vs outside our
      vision → hidden-attacker penalty toggles; key piece inside vs outside their vision
      mask → visibility term toggles. Each term tested in isolation (zero the others).
- [ ] Bounds + scale: output always in [−1, 1]; tanh scale parameter {300, 600, 1200}
      changes magnitude, never sign or ranking of two positions.
- [ ] Legal-in-FoW-only positions (kings adjacent, king en prise for the side NOT to
      move): no exception, sane output.

### SF wrapper (when built)
- [ ] Illegal FEN → handcrafted fallback for that position, no crash, result still bounded.
- [ ] Per-call timeout fires → fallback; the move decision still completes.
- [ ] Engine process killed mid-turn → mode flips to handcrafted, game continues.
- [ ] Eval cache: second request for the same FEN does zero UCI round-trips (counter/mock).
- [ ] Sign agreement with handcrafted on quiet positions where |cp| ≥ 200; log and eyeball
      every disagreement (each one is a PST bug, a guard bug, or a genuine SF-divergence).
- [ ] Determinism: same FEN twice → same score (single thread, fixed options).

### reply.py
- [ ] p(r) is a valid distribution: non-negative, sums to 1, smoothing floor > 0 on every
      legal reply (no −∞ log-likelihood possible).
- [ ] Limits: τ → 0 approaches argmax of the believed-board scores; τ large → uniform.
- [ ] β = 1 reproduces the clairvoyant reply exactly; β = 0 ignores it entirely.
- [ ] Degenerate-belief equivalence: when the believed board IS the true board
      (TrueBoard source), like-me ranking == clairvoyant ranking.
- [ ] Legal-move source: replies are enumerated on true x′, never on believed boards —
      construct a case where a believed board changes their legal move set and assert the
      enumerated set came from x′.

### weighting.py
- [ ] ε-mix limits: ε = 1 → uniform sampling; ε = 0 → pure model weights; empirical
      sample frequencies match expected within tolerance over many draws.
- [ ] |S| ≤ K → all worlds returned unweighted, no sampling path taken.
- [ ] Truth never starved (§2b): with a deliberately adversarial scorer that puts the true
      board last, truth-in-sample stays > 0 across repeated draws (the ε floor working).
- [ ] Numerics: log-domain accumulation over 64 squares — no underflow; scoring is
      batch-order invariant.
- [ ] Metrics: truth-rank / truth-in-sample / ESS verified against hand-computed values
      on tiny fixtures.

### decide.py (aggregation + veto)
- [ ] Weighted mean matches hand computation on a known V matrix; uniform weights
      reproduce the plain mean exactly.
- [ ] Renormalized-within-sample mean (settled design §3): fixture where the picked K
      have unequal w̃ — assert V uses w̃ renormalized over the SAMPLE (not raw ŵ, not
      uniform); ε = 1 edge case → reduces to plain mean over the sample.
- [ ] λ tilt: λ = 0 → mean; λ = 1 → the defined worst-case blend; monotone in between.
- [ ] Veto mass: king-capture mass > δ → vetoed; ≤ δ → survives; a constructed case where
      weighted mass and world COUNT disagree (one implausible danger world) proves the
      mass version is live.
- [ ] All-vetoed → least-exposed fallback returns the minimum-mass move; deterministic
      tie-break.

### x1.py
- [ ] Holdout hygiene: whole teams held out — assert no team id appears in both splits.
- [ ] Metrics on tiny hand-built fixtures with known answers: top-1, top-3, MRR,
      log-likelihood (including the smoothing-floor case where the actual move had raw
      p = 0).
- [ ] All contestants wired and reported per team: uniform, naive, clairvoyant, like-me
      exact (once P2's enumerator lands), like-me diffusion, BC.

### planner (when built — M2-critical)
- [ ] Shortlist contents: top-M naive plus ALL four guaranteed slots (hiding, scouting,
      captures of visible pieces, king-safety fixes), deduplicated; budget-allows → every
      legal move.
- [ ] Danger world force-included when the tracker flag is set.
- [ ] Fallback-first ordering: a stop signal at ANY instant after the clock line yields a
      legal move (the precomputed PIMC fallback), never None, never illegal.
- [ ] Anytime loop: stop mid-widening returns best-so-far.

### Integration / golden / property
- [ ] End-to-end determinism: fixed replay + fixed seed → identical move sequence, in
      BOTH eval modes.
- [ ] Golden games: method A vs baseline.py and uniform_random.py at fixed seed → stored
      move lists; any refactor diffs against them.
- [ ] Referee cross-check at bot level: every move we emit is legal per referee.py (on top
      of P2's engine equality tests).
- [ ] Fuzz: random boards through the evaluator (no exceptions, bounded output); random
      observation sequences through the pipeline with the stub tracker.
- [ ] Timing: per-stage budget assertions on the dev machine with margin; no move over
      budget across a full local game.
- [ ] 50 clean local games at 180+2 before any arena submission (working agreement).
