# Research Knowledge Base
*Auto-generated — the `## Key Insights` section at the bottom is preserved.*

## Research Tree
Full tree is at `/home/user/auto-research/magent_state_pg_lineage_A/blackboard/tree.tsv` (TSV; columns: `exp_id, parent_exp, depth, path, specialist, status, val_bpb, delta_vs_best, hypothesis`; preorder-sorted so siblings are adjacent). Slice it from Bash, e.g.:

```
# subtree rooted at exp_042 (all descendants + the node itself)
awk -F '\t' 'NR==1 || $4 ~ /(^|\/)042(\/|$)/' /home/user/auto-research/magent_state_pg_lineage_A/blackboard/tree.tsv
# direct children of exp_042
awk -F '\t' 'NR==1 || $2=="042"' /home/user/auto-research/magent_state_pg_lineage_A/blackboard/tree.tsv
# only kept trials, sorted by val_bpb
awk -F '\t' 'NR==1 || $6=="keep"' /home/user/auto-research/magent_state_pg_lineage_A/blackboard/tree.tsv | sort -t $'\t' -k7,7g
```

**Current-best lineage** (root → best):
```
exp_000 [baseline, baseline, bpb=1.081000] PR #1758 seed (CaseOps + PreQuant TTT LR=1e-3 Unfrozen, 3-seed claim)
   └─ exp_014 [opt, keep, bpb=1.078777, Δ=-0.000120] Split Muon MLP weight decay (muon_wd=0.095 attn, muon_wd_mlp=0.115 MLP) + reduce ttt_epochs from 3 t
      └─ exp_030 [arch, keep, bpb=1.078041, Δ=-0.000736] Bifurcate per-head q_gain into [H, 2]: col-0 scales the RoPE-rotated head-dim slice (positional chan
         └─ exp_045 [opt, keep, bpb=1.076881, Δ=-0.001160] Set min_lr=0.10 so the linear warmdown LR floor is 10% of peak (not 0), preventing completely frozen
            └─ exp_054 [meta, keep, bpb=1.076735, Δ=-0.000054] Raise warmdown LR floor min_lr=0.10→0.15: exp_045 just established that a non-zero floor compounds w
               └─ exp_064 [meta, keep, bpb=1.076697, Δ=-0.000038] Raise muon_momentum_warmup_fraction 0.33→0.45: extends momentum ramp from 198s to 270s, keeping Muon
                  └─ exp_082 [eval, keep, bpb=1.076538, Δ=-0.000045] Per-chunk NLL-adaptive TTT learning rate: after score-first evaluation of each chunk, scale the cosi
                     └─ exp_092 [ttt, keep, bpb=1.076397, Δ=-0.000141] Combine exp_075's recurrence-block gradient equalization (rescale blocks 3-5 grad by 1/(num_loops+1)
                        └─ exp_109 [curr, keep, bpb=1.074752, Δ=-0.001645] Dynamic small→large batch schedule: use half-size batch (393k tokens) for the pre-warmdown phase (0–
                           └─ exp_130 [quant, keep, bpb=1.074702, Δ=-0.000050] Loss-weighted GPTQ Hessian calibration: two-pass approach computes per-batch CE loss then re-accumul
                              └─ exp_157 [reg, keep, bpb=1.074207, Δ=-0.000495] Lower embed_wd from 0.085 to 0.05: in the severely undertrained 10-minute regime with tiny embed ini
                                 └─ exp_176 [opt, keep, bpb=1.073142, Δ=-0.001065] Muon momentum cooldown 0.99→0.95 after warmup: once the momentum warmup completes at 33% of training
                                    └─ exp_201 [ttt, keep, bpb=1.073136, Δ=-0.000006] Restore exp_092's TTT improvements onto exp_176: add nesterov=True to TTT SGD and recurrence-block g  ← BEST
```

## Recent Activity (last 30)

| exp | specialist | status | val_bpb | hypothesis |
|-----|------------|--------|---------|------------|
| 200 | quant | discard | 1.077172 | Combined freq+loss GPTQ Hessian weighting: multiply existing per-token loss w... |
| 199 | curr | eval_budget_overrun | 1.099951 | Mask first 64 positions from training loss to align gradient with stride-64 s... |
| 198 | arch | size_blocked | 1.071377 | test |
| 197 | tok | discard | 1.073881 | Replace additive output_shift (exp_187, +0.000170 BPB penalty) with multiplic... |
| 196 | meta | discard | 1.073947 | Raise muon_momentum_warmup_start 0.92→0.95: untouched higher-side bracket of... |
| 195 | loss | discard | 1.123512 | Byte-weighted CE training loss: weight each token's NLL by its actual encoded... |
| 194 | reg | discard | 1.073655 | Lower muon_wd_mlp from 0.115 to 0.095 to equalize MLP and attention weight de... |
| 193 | opt | discard | 1.073718 | Delay Muon momentum cooldown start from frac=0.33 to frac=0.70: hold momentum... |
| 192 | quant | crash | — | Combined freq+loss GPTQ Hessian weighting: multiply existing per-token loss w... |
| 191 | eval | discard | 1.716202 | Raise ttt_lr from 0.005 to 0.008: with AdamW(beta1=0) the effective per-step... |
| 190 | quant | size_blocked | 1.078024 | Int4+percentile for all matrix layers: use MSE-optimal percentile scale selec... |
| 189 | ttt | discard | 1.074228 | Add SLOT-style logit-bias correction to TTT scoring: a [vocab_size=8192] floa... |
| 188 | curr | discard | 1.073911 | Stratified shard sampling: replace proportional-remaining shard selection wit... |
| 187 | tok | discard | 1.073312 | Add learnable output_shift [model_dim=512] zero-init, applied to x before the... |
| 186 | arch | eval_budget_overrun | 1.073352 | V-RMSNorm symmetric to existing Q/K norm (HybridNorm 2503.04598, 2025): add F... |
| 185 | meta | discard | 1.075725 | Raise AdamW BETA2 from 0.95 to 0.98 — untouched in 179 trials (only beta1 tes... |
| 184 | opt | discard | 1.074140 | Muon momentum cooldown 0.99→0.90 (deeper): extend exp_176's winning cooldown... |
| 183 | reg | discard | 1.084735 | Split zero-init output projections (attn.proj + mlp.proj) into a dedicated Mu... |
| 182 | quant | size_blocked | 1.067220 | MSE-optimal percentile scale selection (exp_170 approach) for all layers, plu... |
| 181 | loss | discard | 1.084269 | Remove exp_172's harmful z-loss and add label smoothing ε=0.02: z-loss at 1e-... |
| 180 | eval | discard | 1.549940 | Switch TTT optimizer from SGD+momentum to AdamW(betas=(0.0,0.9), weight_decay... |
| 179 | meta | discard | 1.076099 | Lower enable_looping_at 0.35→0.28 to align loop activation with the batch-swi... |
| 178 | tok | discard | 1.075296 | Add per-dimension causal bigram mixing (bigram_mix=[512], zero-init): uses F.... |
| 177 | arch | eval_budget_overrun | 1.076976 | RWKV-V5+ token shift on attention input: per-channel learnable lerp between R... |
| 176 | opt | keep | 1.073142 | Muon momentum cooldown 0.99→0.95 after warmup: once the momentum warmup compl... |
| 175 | ttt | discard | 1.087726 | Replace standard CE with focal loss (γ=2) in TTT inner training loop: focal_w... |
| 174 | curr | discard | 1.074228 | Quarter-batch phase 1 (196k tokens, 12 seqs/GPU) for 0–28% of training: same... |
| 173 | reg | discard | 1.083402 | Stochastic depth with linear depth scaling (max_rate=0.1): layer i drops its... |

## Key Insights
*(Add manually — this section is preserved across regenerations.)*
