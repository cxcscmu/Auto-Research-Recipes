# Seed Stack — multi_596 (val_bpb ≈ 1.072, our-node verified)

Starting point. `multi_agent/train_gpt.py` is **multi_596 verbatim**
(rebased apr-25 from the canonical `train_gpt_multi_596.py` at repo
root). Every specialist's baseline is this file; mutations happen
from here outward.

See `pr_library/INDEX.md` (already injected into your system prompt)
for the curated, compliance-pre-screened PR landscape, and the
`read_pr_library` / `read_pr_source` tools for porting techniques from
those PRs onto this baseline.

## Seed baseline (our-node 8 GPUs verified)

| Metric | Value |
|---|---|
| `val_bpb` (post-quant + sliding eval + score-first TTT) | **≈ 1.072** (3-seed VALID; one-node variance ~0.001) |
| Artifact size (code + model) | ≈ 15.99–16.00 MB (tight margin) |
| Train wall | ≈ 600 s (capped by `MAX_WALLCLOCK_SECONDS`) |
| Eval wall | ≈ 575–595 s (varies; some seeds DQ_EVAL by ~5–10 s — node jitter) |
| Hardware | 8 GPUs 80 GB SXM, PyTorch 2.9.1+cu128, FA3 |

This baseline is **fully compliant** with the official Parameter Golf
rules: score-first TTT only (no pre-quant TTT), SP8192 + fineweb10B
default data, no CaseOps tokenizer, no byte-sidecar, no
non-record-track tricks. It reproduces ±0.001 val_bpb across seeds.

## Components (all in `train_gpt.py`)

1. **Tokenizer** — SentencePiece, `vocab_size=8192`. Model file
   `fineweb_8192_bpe.model`. Dataset `fineweb10B_sp8192/`.
2. **Architecture** — 11 transformer blocks; `model_dim=512`,
   `num_heads=8`, `num_kv_heads=4` (GQA stride-2 KV), `mlp_mult=4.0`
   (hidden=2048).
3. **3-Layer Recurrence** — `num_loops=2`, `loop_start=3`, `loop_end=5`.
   Layers 3-5 invoked 3× per forward pass; `ln_scale_factor` divided
   by `sqrt(num_loops+1)` to equalize residual contribution.
4. **Parallel Residuals** — `parallel_residual_start=7`. Layers 7-10
   compute attention + MLP in parallel branches summed into the
   residual.
5. **QK-Gain (bifurcated [H,2])** — learnable per-head Q scale
   initialized at 5.25; col-0 gates RoPE region, col-1 gates NoPE
   region (partial RoPE with `rope_dims=16`).
6. **Per-head attention-output gate** — sigmoid gate on attention
   output, zero-init so step-0 is transparent.
7. **XSA on all 11 layers** — exclusive-self-attention (subtract
   normalized-V projection of the output).
8. **MuonEq-R optimizer** — row-normalized Muon (`muon_row_normalize=1`)
   with split MLP weight decay (`muon_wd=0.095`, `muon_wd_mlp=0.115`).
9. **Score-first TTT** (legal per official README) — `ttt_enabled=1`,
   `ttt_lr=0.005`, `ttt_epochs=4`. Per chunk: score val tokens with
   `torch.no_grad()` first, then SGD-adapt on that chunk, advance.
10. **Sliding-window eval** — `eval_stride=64`.
11. **GPTQ SDClip int6 (matrix) / int8 (embed)** — per-row scales,
    Hessian damping 1.01, `matrix_clip_sigmas=12.85`,
    `embed_clip_sigmas=20.0`, block_size=32.
12. **Byte-shuffle stride-2 + Brotli-11** — quant blob compression.
13. **Tied embeddings** — `tok_emb` shared with logit projection;
    `lm_head` is `None`.

## Tunable size escape hatch

`LOWBIT_LAYERS` env var supports per-pattern bit override on top of
the int6 matrix default. Format: `pattern:bits[,pattern:bits...]`.

```
LOWBIT_LAYERS="blocks.8.mlp.proj:5"               # 1-matrix int5 demo
LOWBIT_LAYERS="blocks.{4,5,6}.mlp.{fc,proj}:5"    # 6-matrix int5
```

Off by default (empty string → all matrices stay int6). Used as
last-resort if seed variance pushes us over the 16 MB cap; expected
val_bpb cost +0.001-0.005 per matrix at int5.

## Byte budget

| Part | Bytes |
|---|---|
| Code (packed lzma-RAW + base85) | ≈ 19,600 |
| Code (unpacked, served by our node) | ≈ 70,400 |
| Model (GPTQ int6 + brotli-11) | ≈ 15,975,000 |
| Headroom vs 16 MB cap | ≈ 0–5,000 bytes (variable per seed) |

The tight cap is the **binding constraint**. Any seed-increase edit
(deeper model, wider MLP, more embed params) must be paired with a
quant/compression saving. Always check `size_project` before
submitting.

## Known failure modes

- **Tight cap, seed-variable.** Quant blob varies ~1-3 KB stdev
  across seeds. A bad seed can tip us over. Watch `size_project`
  output for the actual margin; if smoke pack > 16,000,000, try
  `LOWBIT_LAYERS="blocks.8.mlp.proj:5"` as escape.
- **Eval near-overrun.** `eval_s` lands 575–595 s typically; node
  jitter can push it past 600 → `DQ_EVAL`. Trim `ttt_epochs` or
  `eval_stride` if persistent.
- **torch.compile + STE.** Standard compile vs quantized-inference
  hazard — the STE branch must reach the compiled path for GPTQ
  outputs to match train behavior. See `INIT.md`.
- **AllReduce non-determinism.** 8 GPUs reduction order is
  non-deterministic; same `SEED=42` gives ±0.001 val_bpb across
  re-runs. Plan multi-seed verify before declaring an improvement
  significant; the official 0.005-nat threshold is far above this
  noise floor.

## Porting techniques (apr-25 model)

Wholesale-rebase to a PR is no longer supported (the `rebase_to_pr`
tool was removed). The PR library is now a curated technique-donor
catalogue (n=68 visible, all pass 7 compliance gates). Workflow:

1. Scan `pr_library/INDEX.md` for a PR whose `top specialists` column
   includes your domain.
2. `read_pr_library(pr_number)` to read the per-technique rationale
   (each technique tagged `[specialist / risk]`) and the compliance
   gate breakdown.
3. `read_pr_source(pr_number, path)` for implementation detail when
   the summary isn't specific enough.
4. Port a single technique by hand-edit onto your current train_gpt.py.

PR numbers absent from INDEX (e.g. #1758, #1370, #1738, #1350, #1564)
were intentionally archived — they fail one or more hard-legality gates
(pre-quant TTT, val-token leak, sub-Shannon, missing FLA dependency,
etc). `read_pr_library(N)` for an archived PR returns
`not in the library`; don't retry.
