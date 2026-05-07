# Operational Lessons — do not relearn

Distilled from the single-agent run history (≈200 trials). Load-bearing
for anyone mutating `train_gpt.py`.

## Before you propose

- **Read recent `results.tsv` rows**. If the last five trials in your
  domain all tripped `size_blocked`, your next edit almost certainly
  needs to shrink, not grow.
- **`val_bpb:F` alone is weak signal.** The primary regex is
  `quantized_ttt val_loss:… val_bpb:F` — that's the full pipeline
  (quantization + sliding window + TTT). A fallback pre-quantization
  number is easy to confuse with the real metric; don't propose a keep
  from a pre-TTT value.
- **Crash ≠ penalty.** Preflight crashes and GPU-side crashes are cheap
  and informative; don't add defensive scaffolding to avoid them unless
  the cause is clearly out-of-scope (OOM, CUDA init).

## Size gate (pre-run + post-run)

- `run_trial.sh` runs `SMOKE_TEST=1` before the real train and greps
  `smoke_pack_bytes: total=N` from the log. If `N > 16 000 000`, the
  real run is skipped and the trial is classified `DQ_SIZE`. No GPU time
  is burned past preflight (~60 s).
- Smoke uses 2-batch GPTQ calibration vs full calibration at real-run.
  Smoke bytes drift ±KB from real; head-side calibration is NOT yet
  implemented in multi_agent, so the gate is raw-compare.
- A real-run `DQ_SIZE` after a passing smoke means GPTQ's full
  calibration produced more entropy than the 2-batch estimate. Treat
  this as a legitimate post-run failure; tighten quantization or
  compression.

## Time budgets

- Train self-caps via `MAX_WALLCLOCK_SECONDS=600`. `train_budget_overrun`
  means an edit broke the cap logic (the wallclock check at the top of
  the training loop, the `stopping_early: wallclock_cap` exit, or
  time-fractional schedule anchoring). Audit those three sites.
- Eval has NO in-file cap — `run_trainer.py` enforces a strict 600s eval
  budget (SIGTERM the torchrun process group once post-train elapsed >
  600s, emits `--- EVAL_TIMEOUT after 600s ---`). Outer bash `timeout
  2200s` is the backstop (600 train + 600 eval + 1000 compile headroom).
  `eval_budget_overrun` / `DQ_EVAL` usually means a TTT or sliding-window
  change blew the budget. Compile time is unbounded by the rules.

## Crash interpretation

- `notes` on a crash row carries a compact excerpt: deepest
  `train_gpt.py:line` + exception class + first 200 chars of message.
  Read the excerpt before re-proposing; blind retries on sliding-window
  shape bugs cost a lot of iters.
- `preflight_crash` means the head-side syntax or pre-run size check
  failed — no GPU used. Fix and re-propose immediately.
- If the combined log contains `--- OUTER_TIMEOUT ---` the node was
  killed by `run_trial.sh`'s backstop; check for hangs in GPTQ, sliding
  window, or TTT.

## Parallel-agent etiquette

- Other specialists read the blackboard simultaneously. Don't assume
  `best.json` will still be best at your `submit_trial` time — there
  may be a newer `keep` row between your read and your submit. The
  harness handles this correctly (baseline is re-snapped at
  `record_trial` time), so there's nothing you need to do, but don't
  plan proposals that assume a stable baseline across a long reasoning
  window.
- When two specialists propose the same idea in parallel, both trials
  run. The second lands as `discard` naturally since it's delta vs the
  (now-improved) best. This is fine — no explicit deduplication needed.

## Things that look like bugs but aren't

- `smoke_pack_bytes: code=C model=M total=T` line missing in the real
  run: correct — smoke lines are only emitted when `SMOKE_TEST=1`.
- `run_seed0.jsonl` has `"val_bpb": null` on preflight-aborted rows:
  correct — preflight didn't reach a metric.
- Compile time 60–120 s on the first trial after node cold-start,
  10–30 s after: correct — `torch.compile` caches on disk.

## Lessons from the 500-trial run (2026-04-20 to 2026-04-23, baseline 1.0810 → 1.073978)

506 trials, 10 specialists, 30 keeps total. Keep-rate decayed 11% → 2% from
exp_100 to exp_500 as the stack saturated. Abandoning the 1.0810 stack for a
new ~1.03 bpb baseline, so the lessons below are filtered for portability.

### Swarm-operational

- **Keep-rate collapses fast on a saturating stack** (general). 7% / 11% /
  7% / 3% / 2% across exp_000–499 in 100-trial windows. Once per-domain
  lessons accumulate and every specialist is proposing 3rd-order tweaks,
  the search degenerates into noise. Plan for it: rotate the baseline or
  the specialist roster when Δ-per-keep drops below iteration variance.
- **Size-gate incidence grows with compression headroom shrinking**
  (general). size_blocked share rose from ~13% (exp_000–199) to ~48%
  (exp_400–524). Any proposal adding params late in the run needs
  explicit byte accounting — specialists largely failed at this.
- **Per-domain "depth" varies 5× in productive keeps** (general).
  `opt` landed the biggest single win (Δ=-0.001115 exp_088). `quant` kept
  most often (5/53) but with tiny Δ per keep. `curr` kept 2/46 with near-
  noise Δ — low ROI domain. `eval` kept 1/47 (the kept change was a TTT
  hyper, not an eval-specific knob) — `eval` as a distinct axis was weak;
  consider merging eval/ttt under a single specialist on the new stack.
- **`arch` proposals have a bimodal crash/keep distribution** (general).
  7 crashes + 12 size_blocks vs only 3 keeps (53 total). Architecture
  edits need an explicit preflight invariant (shape round-trip + small-batch
  forward) or they burn iterations on thread-through bugs.
- **Parallel specialists re-propose the same idea in waves** (general).
  EMA_decay 0.9965→0.998 was re-tried as "orphan revival" on 6+ different
  baselines (exp_003/083/100/119/184/242/393/447). Most failed because the
  win didn't compose. Add explicit "already tried on this branch" de-dup to
  the blackboard — don't rely on specialists noticing.
- **`harness_abort` in this run was almost entirely one infra event**
  (1.0810-specific): a scheduler-side cache miss quarantined ~42 trials in a single
  wave. Not a specialist-behavior lesson.
- **`eval_budget_overrun` is concentrated in TTT-touching edits** (general).
  26 cases, dominated by `eval`/`ttt`/`loss` trials that added inner SGD
  steps, shrank chunks, or enabled Nesterov+reset combos. Budget-cost
  pre-estimation (epochs×chunks×batch_size vs current) should be required
  before submitting a TTT mutation.

### Per-domain directional heuristics

- **TTT: more inner adaptation helped; momentum resets did not** (general).
  Confirmed keeps: +1 epoch/chunk (exp_014, exp_017), Nesterov on TTT SGD
  (exp_151), fwd/rev traversal alternation (exp_264), freeze small-numel
  control scalars during TTT (exp_491). Hurt or no-signal across ≥8
  trials: per-chunk momentum-buffer resets (exp_064/089/113/171/213/232/
  253/255/305/327/392/480/501/503). Direction: TTT likes *more* continuity
  and *more* adaptation, not less.
- **GPTQ: finer-grained, gentler damping helped; structural replacements
  did not** (general). Wins: block_size 128→64→32, multiplicative Hessian
  damping, monotone downward damp strength (exp_108/212/261/280/377).
  Losses: max-abs scale, MAD scale, column re-ordering, closed-form
  post-quant scale refinement (exp_145/092/125/319). On a new stack,
  start by sweeping block_size down and damping down before touching
  calibration structure.
- **Muon momentum: shorten warmup, raise the deep-warmdown floor**
  (general). Keeps: warmup_fraction shrink (exp_088), cosine-cool during
  warmdown (exp_010), raise momentum floor at lr_scale<0.25 (exp_324).
  Probe-up-peak losses: 0.99→0.995 (exp_138, 433). Probe-down-peak losses:
  0.99→0.97, 0.985 (exp_094, 394). Peak momentum knob appears locally
  optimal on the 1.0810 stack — likely re-optimal on stronger baselines.
- **Weight-decay: split by parameter role helped; zeroing didn't** (general).
  Keeps: per-type split (exp_164), MLP-only DOWN step (exp_336), scalar-
  group-only DOWN step (exp_177). Losses: scalar-WD-to-zero (exp_062/087/
  257/313), attn-WD DOWN (exp_414/489).
- **Curriculum: Latin-square/stratified shard sampling helped exactly
  twice, then saturated** (general). exp_142, exp_196. 8 subsequent
  low-discrepancy variants (Van der Corput, golden-ratio, Weyl, bit-
  reversal, random) all discarded or size-blocked. Signal: one pass of
  "less-variance sample order" is worth trying; iterating further is not.
- **Loss: gentle monotone position-weighted CE helped; shape exponent
  didn't matter much** (general). linear 0.9→1.1 (exp_156), u² (exp_186),
  then higher exponents (u⁴, cubic) all neutral/hurt. Steeper spans (0.5→
  1.5, 0.8→1.2) hurt. On a new baseline, test one gentle PW-CE variant
  and move on — don't sweep shape.
- **Token embedding: single-feature gain helped; biases and per-vocab
  knobs hurt** (general). Keeps: per-feature embed_scale (exp_190, 267,
  298). Losses/size-blocks across ≥10 trials: per-vocab logit_bias,
  output_bias, embed_bias (exp_023/166/219/249/288/307/317/361/365/399/
  486/496/505). Per-vocab 8192-param additions almost always tripped
  the size gate.
- **`opt` LR/clip anneals almost always underperform the static baseline**
  (general). cosine/linear grad_clip anneal, beta1/beta2 anneal, LR-floor
  schedules — mostly discard with +0.0001..+0.001. Only lockstep Muon
  momentum cool and peak-momentum floor kept. Direction: schedule the
  thing that has a known physical coupling (momentum↔LR under
  Newton-Schulz); stop rescheduling every hyper in sight.
- **`meta` was heavily redundant with `opt`** (general). Most `meta`
  trials were "raise/lower AdamW beta/eps/wd/warmdown_frac" — 32 discards,
  3 tiny keeps. Collapsing meta into opt on the new stack would save
  ~10% of iteration budget.

### Explicit anti-patterns (do not retry)

- **Dropout in any form** (general). MLP hidden dropout, residual dropout,
  DropPath, token dropout, feature dropout, x0-path dropout: 10 trials,
  0 keeps, mean +0.003 bpb, repeatedly size-blocked or discarded
  (exp_012/040/050/155/198/293/302/430/476 and more). This baseline class
  is not dropout-friendly; reassess only if the new baseline already uses
  dropout.
- **NormFormer / sandwich / sub-LN** (general). 5 attempts, 3 crashes, 2
  discards (exp_090/098/206/286/398/403). Every attempt to add an extra
  RMSNorm inside the block broke something or added bytes without help.
  Skip until there's a specific reason the new stack's activation
  statistics demand it.
- **V-norm (adding rms_norm on V to match Q/K-norm)** (general). 3 trials,
  0 keeps (exp_150/235/296). Symmetry argument is tempting and wrong here.
- **Multi-token prediction / 2-ahead aux CE** (general). 3 trials, all
  size_blocked with bpb +0.01..+0.02 (exp_281/308/478). The extra logits
  path or head changes compress badly; defer unless the new stack already
  ships an MTP head.
- **Per-vocab learnable parameters (logit_bias, embed_bias [V], output
  bias)** (general). ≥10 attempts, all discard or size_blocked. Vocab-size
  params are compression-hostile on small-model regimes.
- **SwiGLU / squared-ReLU swaps on a silu²-Primer MLP** (1.0810-specific,
  but directional). 8 trials, 0 keeps, often size-blocked. Signal: don't
  swap the activation family to match a "standard" — tune what's in place.
- **Logit temperature in eval/scoring** (general). 29 trials touching
  T=0.95..1.05; only 3 keeps, all tiny, all on narrow branches. The eval
  softmax temp is a weak knob; one probe early, then skip.
- **EMA-decay orphan revival** (general). exp_003's 0.9965→0.998 kept on
  the baseline but failed to stack on 6+ downstream bases. General lesson:
  "proven on an old parent" keeps often *don't* re-apply cleanly on
  distant descendants. Require the specialist to test on the *current*
  branch before claiming it's unvalidated-elsewhere.
- **Per-chunk TTT momentum reset** (general). ≥12 attempts across multiple
  specialists, 0 keeps. Stop proposing this.
- **Label smoothing ≥0.002** (general). 6 trials, 0 keeps, all +0.002..
  +0.006 (exp_026/042/222/246/338/428). Small-model CE doesn't want it.
- **Focal / anti-focal CE reweighting** (general). 13 focal, 7 anti-focal,
  0 durable keeps. Hurts small-model CE calibration here.

### Harness pitfalls

- **size_blocked margins are tiny near the best** (general). Late in the
  run, trials routinely beat the best on raw val_bpb (e.g. exp_450 at
  1.073985, exp_493 at 1.073997) then size-blocked by <1 KB. Without
  head-side GPTQ calibration in smoke, specialists can't predict which
  change will bust the gate. Either implement head-side smoke GPTQ or
  require every proposal >5 kB of new params/bytes to pass a byte budget.
- **Smoke-pass-then-real-run-size_blocked is a real failure class**
  (general). The 2-batch GPTQ smoke underestimates full-calibration entropy
  by ~KB margin. On a stack with tighter headroom this becomes worse; plan
  to invest in aligned-calibration smoke before scaling iterations.
- **`test` hypothesis slipped through** (general). exp_316 had hypothesis
  text "test" and still ran (discarded, +0.001819). Add a min-hypothesis-
  length guard in the proposer.
- **Compile cache invalidation on certain `arch` edits is silent**
  (general). Several arch crashes (exp_001/206/286/320/321/398/401) share
  "thread-through shape change" signatures — suggests compile trace
  invalidation with a post-compile runtime surface. Worth checking on new
  stack: does a preflight eager forward catch these before compile?

### Top-5 keeps: mechanism-only (for porting evaluation)

- **exp_491 (TTT, Δ=-0.000043)** — Freeze very-small-numel control scalars
  during the TTT inner loop (general, mechanism-only). The *class* of
  move: "during test-time adaptation, don't let per-head/per-channel
  control gains drift, only let weight matrices adapt." Re-evaluate
  on any baseline that uses TTT + small scalar gain tensors.
- **exp_475 (arch, Δ=-0.000066)** — Bifurcate a per-head scalar gain
  across RoPE-vs-NoPE head-dim regions (1.0810-specific-ish, but
  directional). Mechanism: "when a region of the head has different
  inductive bias (rope vs no-rope, or any region-split), a single
  gain-per-head averages two distinct optima; split by region."
  Applies wherever partial-RoPE or per-head-region gains exist.
- **exp_377 / exp_280 (quant, Δ=-0.000084 / -0.000054)** — Multiplicative
  signal-proportional Hessian damping with *monotone-downward* damp
  strength (general). Mechanism: small multiplicative `H_ii *= 1+ε` is
  strictly better than large additive `H_ii += α·mean(H)` for GPTQ
  stability; and once multiplicative, probe ε downward. Re-test on any
  GPTQ-in-pipeline stack.
- **exp_336 (meta, Δ=-0.000109)** — *Down*-probe on split-by-type Muon
  WD after the split was established (general). Mechanism: when you
  split a regularization coefficient by parameter role, the old joint
  optimum is generally too strong for one sub-group. Probe DOWN on each
  new sub-group before probing UP.
- **exp_324 (opt, Δ=-0.000202)** — Floor momentum-warmdown blend at a
  non-zero value so the deepest-warmdown tail still has meaningful
  momentum (general). Mechanism: lockstep-to-LR schedules that hit zero
  at end-of-training are over-aggressive; a small floor preserves late-
  phase refinement. Re-applicable to any schedule that cools a
  secondary hyper in lockstep with LR.
