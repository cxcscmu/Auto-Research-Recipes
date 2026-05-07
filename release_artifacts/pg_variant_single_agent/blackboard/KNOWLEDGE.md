# Research Knowledge Base
*Auto-generated — the `## Key Insights` section at the bottom is preserved.*

## Research Tree
Full tree is at `/home/user/auto-research/magent_state_pg_single_C/blackboard/tree.tsv` (TSV; columns: `exp_id, parent_exp, depth, path, specialist, status, val_bpb, delta_vs_best, hypothesis`; preorder-sorted so siblings are adjacent). Slice it from Bash, e.g.:

```
# subtree rooted at exp_042 (all descendants + the node itself)
awk -F '\t' 'NR==1 || $4 ~ /(^|\/)042(\/|$)/' /home/user/auto-research/magent_state_pg_single_C/blackboard/tree.tsv
# direct children of exp_042
awk -F '\t' 'NR==1 || $2=="042"' /home/user/auto-research/magent_state_pg_single_C/blackboard/tree.tsv
# only kept trials, sorted by val_bpb
awk -F '\t' 'NR==1 || $6=="keep"' /home/user/auto-research/magent_state_pg_single_C/blackboard/tree.tsv | sort -t $'\t' -k7,7g
```

**Current-best lineage** (root → best):
```
exp_000 [baseline, baseline, bpb=1.081000] single-agent generalist baseline (PG 1.0810 SOTA stack)
   └─ exp_002 [generalist, eval_budget_overrun, bpb=1.080426] Freeze tok_emb during score-first TTT (tied embedding == output projection; per-chunk SGD on it over
      └─ exp_003 [generalist, keep, bpb=1.079370, Δ=-0.001630] Skip standalone quantized_sliding_window eval when TTT is enabled (TTT itself does sliding-window sc
         └─ exp_004 [generalist, keep, bpb=1.078965, Δ=-0.000405] Raise TTT cosine LR floor from 0 to 0.4·ttt_lr so late val chunks (where the model has accumulated u
            └─ exp_047 [generalist, keep, bpb=1.078023, Δ=-0.000942] Raise TTT cosine LR floor 0.4→0.7: exp_004 proved 0→0.4 helps via more late-chunk drift; pushing 0.7
               └─ exp_048 [generalist, keep, bpb=1.078012, Δ=-0.000011] Push TTT cosine LR floor 0.7→1.0 (= constant LR per chunk): tests asymptote of the now-confirmed sup
                  └─ exp_056 [generalist, keep, bpb=1.077861, Δ=-0.000151] Replace linear LR warmdown with cosine decay (same endpoints: 1.0→min_lr=0.1), keeping LR higher for
                     └─ exp_076 [generalist, keep, bpb=1.077585, Δ=-0.000276] Isolate Muon WD split: attn/other matrices use muon_wd=0.095, MLP matrices use muon_wd_mlp=0.115 (th
                        └─ exp_078 [generalist, keep, bpb=1.077522, Δ=-0.000063] Paired Head Muon: apply Newton-Schulz orthogonalization independently to Q/K matrices in head-pair b
                           └─ exp_080 [generalist, keep, bpb=1.077488, Δ=-0.000034] Async CPU batch prefetch: move mmap data loading to a background thread (queue.Queue maxsize=3) so C
                              └─ exp_081 [generalist, keep, bpb=1.076421, Δ=-0.001067] ResFormer value residual (arxiv 2410.17897): thread block[0]'s V projection of x0 to all attention l
                                 └─ exp_146 [generalist, keep, bpb=1.076144, Δ=-0.000277] Bundle of 4 accumulated improvements over exp_081 (all infrastructure-failed in prior sessions): (1) 
                                    └─ exp_147 [generalist, keep, bpb=1.076019, Δ=-0.000125] Remove nesterov=True from TTT SGD (revert to heavy-ball momentum): exp_005 showed Nesterov added +0.
                                       └─ exp_156 [generalist, keep, bpb=1.075543, Δ=-0.000476] Bifurcated q_gain [H,2]: separate learnable temperature scales for RoPE dims (position-aware, first 
                                          └─ exp_159 [generalist, keep, bpb=1.075418, Δ=-0.000125] Per-KV-head V-residual gate: expand v_res_scale from shape (1,) to (num_kv_heads=4,) so each KV head
                                             └─ exp_171 [generalist, keep, bpb=1.075384, Δ=-0.000034] FreqGPTQ: bias GPTQ Hessian calibration toward high-frequency tokens (top-100 get sqrt(2)× activatio  ← BEST

**Best's direct children (27):**
  • exp_172 [generalist, discard, bpb=1.079314, Δ=+0.003930] KV temporal shift mixing: add per-KV-head learnable tanh-bounded scalars (init 0=transparent) that b
  • exp_173 [generalist, discard, bpb=1.077695, Δ=+0.002311] V normalization (RMSNorm on V before v_res addition): symmetric with existing QK normalization, maki
  • exp_174 [generalist, size_blocked, bpb=1.077518] Muon-VS (Variance-Scaled Muon, arxiv 2601.14603): add per-row exponential-average squared-norm track
  • exp_175 [generalist, discard, bpb=1.077911, Δ=+0.002527] Continuous frequency-proportional GPTQ weighting: replace binary top-100 FreqGPTQ (only 1.2% of voca
  • exp_176 [generalist, discard, bpb=1.077752, Δ=+0.002368] One-sided FreqGPTQ: change clamp(0.5, 2.0) to clamp(1.0, 2.0) so rare tokens never get penalized (we
  • exp_177 [generalist, size_blocked, bpb=1.075595] Enable depth recurrence earlier (enable_looping_at 0.35→0.25): activates the 3-layer loop (layers 3-
  • exp_178 [generalist, size_blocked, bpb=1.075363] Remove full-Hessian GPTQ scale refinement (s*=(W^T H Q)/(Q^T H Q)): exp_170 showed this feature cost
  • exp_179 [generalist, discard, bpb=1.075765, Δ=+0.000381] Increase GPTQ calibration batches 64→96: 50% more Hessian data gives a more accurate H=X^T X estimat
  • exp_180 [generalist, size_blocked, bpb=1.079140] Learnable per-block MLP negative slope (PReLU²): replace fixed leaky_relu(0.5)^2 with a learnable ne
  • exp_181 [generalist, size_blocked, bpb=1.076819] Replace MLP activation from leaky_relu(0.5)² to ReLU² (pure squared ReLU, negative_slope=0→0): maxim
  • exp_182 [generalist, size_blocked, bpb=1.075614] Learned per-block MLP activation mixing: add a scalar act_mix=nn.Parameter(tensor(-4.0)) per block s
  • exp_183 [generalist, discard, bpb=1.216216, Δ=+0.140832] AdamW TTT (betas=(0.9,0.95), lr=0.001): replace SGD with AdamW for per-parameter adaptive learning r
  • exp_184 [generalist, discard, bpb=1.079045, Δ=+0.003661] Z-loss regularization (alpha=1e-4): add 1e-4 * mean(log(sum(exp(logits)))^2) to training CE; penaliz
  • exp_185 [generalist, discard, bpb=1.081399, Δ=+0.006015] LAWA (Latest Weight Averaging): replace EMA with uniform average of 5 explicit model snapshots taken
  • exp_186 [generalist, discard, bpb=1.079847, Δ=+0.004463] Last-4-blocks TTT (blocks 7-10 only) with ttt_epochs=5: restrict TTT adaptation to the last 4 blocks
  • exp_189 [generalist, size_blocked, bpb=1.075426] Remove full-Hessian GPTQ scale refinement (s*=(W^T H Q)/(Q^T H Q)): exp_178 confirmed this feature h
  • exp_190 [generalist, size_blocked, bpb=1.078456] Focal loss (gamma=0.5) during training: down-weight easy tokens (high predicted probability) and up-
  • exp_191 [generalist, size_blocked, bpb=1.080347] Deeper LR warmdown: reduce min_lr from 0.05 to 0.01 (decay to 1% of peak instead of 5%), allowing th
  • exp_192 [generalist, discard, bpb=1.075401, Δ=+0.000017] Remove full-Hessian GPTQ scale refinement (s*=(W^T H Q)/(Q^T H Q)): exp_178 measured BPB=1.075363 fr
  • exp_193 [generalist, size_blocked, bpb=1.076566] ExoFormer K residual (arxiv 2601.08131): add block[0]'s K projection of normalized x0 as a per-KV-he
  • exp_194 [generalist, size_blocked, bpb=1.077065] Q-residual anchor (ExoFormer-style): thread block[0]'s Q projection of normalized x0 as a per-head g
  • exp_195 [generalist, eval_budget_overrun, bpb=—] MuonTTT: replace SGD+momentum TTT optimizer with Muon-style NS-normalized momentum — accumulate grad
  • exp_196 [generalist, eval_budget_overrun, bpb=—] MuonTTT with ttt_epochs=2: NS-normalized momentum TTT (LaCT arXiv 2505.23884) using zeropower_via_ne
  • exp_197 [generalist, eval_budget_overrun, bpb=—] MuonTTT with min_dim=64 gate: NS-normalized momentum TTT applied only to large weight matrices (both
  • exp_198 [generalist, discard, bpb=1.317916, Δ=+0.242532] MuonTTT ns_steps=2 diagnostic: reduce NS iterations from 5 to 2 to test if NS compute is the source 
  • exp_199 [generalist, size_blocked, bpb=1.075137] Increase warmdown_frac from 0.72 to 0.78: spend 78% of the 600s training budget in cosine LR decay (
  • exp_200 [generalist, size_blocked, bpb=1.075522] Increase warmdown_frac from 0.72 to 0.75: spend 75% of training budget in cosine LR decay instead of
```

## Recent Activity (last 30)

| exp | specialist | status | val_bpb | hypothesis |
|-----|------------|--------|---------|------------|
| 200 | generalist | size_blocked | 1.075522 | Increase warmdown_frac from 0.72 to 0.75: spend 75% of training budget in cos... |
| 199 | generalist | size_blocked | 1.075137 | Increase warmdown_frac from 0.72 to 0.78: spend 78% of the 600s training budg... |
| 198 | generalist | discard | 1.317916 | MuonTTT ns_steps=2 diagnostic: reduce NS iterations from 5 to 2 to test if NS... |
| 197 | generalist | eval_budget_overrun | — | MuonTTT with min_dim=64 gate: NS-normalized momentum TTT applied only to larg... |
| 196 | generalist | eval_budget_overrun | — | MuonTTT with ttt_epochs=2: NS-normalized momentum TTT (LaCT arXiv 2505.23884)... |
| 195 | generalist | eval_budget_overrun | — | MuonTTT: replace SGD+momentum TTT optimizer with Muon-style NS-normalized mom... |
| 194 | generalist | size_blocked | 1.077065 | Q-residual anchor (ExoFormer-style): thread block[0]'s Q projection of normal... |
| 193 | generalist | size_blocked | 1.076566 | ExoFormer K residual (arxiv 2601.08131): add block[0]'s K projection of norma... |
| 192 | generalist | discard | 1.075401 | Remove full-Hessian GPTQ scale refinement (s*=(W^T H Q)/(Q^T H Q)): exp_178 m... |
| 191 | generalist | size_blocked | 1.080347 | Deeper LR warmdown: reduce min_lr from 0.05 to 0.01 (decay to 1% of peak inst... |
| 190 | generalist | size_blocked | 1.078456 | Focal loss (gamma=0.5) during training: down-weight easy tokens (high predict... |
| 189 | generalist | size_blocked | 1.075426 | Remove full-Hessian GPTQ scale refinement (s*=(W^T H Q)/(Q^T H Q)): exp_178 c... |
| 188 | generalist | discard | 1.079168 | FreqGPTQ stronger boost: increase top-100 token activation multiplier from sq... |
| 187 | generalist | discard | 1.080083 | Increase rope_dims from 16 to 24 (37.5% of head_dim=64) to give the model ric... |
| 186 | generalist | discard | 1.079847 | Last-4-blocks TTT (blocks 7-10 only) with ttt_epochs=5: restrict TTT adaptati... |
| 185 | generalist | discard | 1.081399 | LAWA (Latest Weight Averaging): replace EMA with uniform average of 5 explici... |
| 184 | generalist | discard | 1.079045 | Z-loss regularization (alpha=1e-4): add 1e-4 * mean(log(sum(exp(logits)))^2)... |
| 183 | generalist | discard | 1.216216 | AdamW TTT (betas=(0.9,0.95), lr=0.001): replace SGD with AdamW for per-parame... |
| 182 | generalist | size_blocked | 1.075614 | Learned per-block MLP activation mixing: add a scalar act_mix=nn.Parameter(te... |
| 181 | generalist | size_blocked | 1.076819 | Replace MLP activation from leaky_relu(0.5)² to ReLU² (pure squared ReLU, neg... |
| 180 | generalist | size_blocked | 1.079140 | Learnable per-block MLP negative slope (PReLU²): replace fixed leaky_relu(0.5... |
| 179 | generalist | discard | 1.075765 | Increase GPTQ calibration batches 64→96: 50% more Hessian data gives a more a... |
| 178 | generalist | size_blocked | 1.075363 | Remove full-Hessian GPTQ scale refinement (s*=(W^T H Q)/(Q^T H Q)): exp_170 s... |
| 177 | generalist | size_blocked | 1.075595 | Enable depth recurrence earlier (enable_looping_at 0.35→0.25): activates the... |
| 176 | generalist | discard | 1.077752 | One-sided FreqGPTQ: change clamp(0.5, 2.0) to clamp(1.0, 2.0) so rare tokens... |
| 175 | generalist | discard | 1.077911 | Continuous frequency-proportional GPTQ weighting: replace binary top-100 Freq... |
| 174 | generalist | size_blocked | 1.077518 | Muon-VS (Variance-Scaled Muon, arxiv 2601.14603): add per-row exponential-ave... |
| 173 | generalist | discard | 1.077695 | V normalization (RMSNorm on V before v_res addition): symmetric with existing... |
| 172 | generalist | discard | 1.079314 | KV temporal shift mixing: add per-KV-head learnable tanh-bounded scalars (ini... |
| 171 | generalist | keep | 1.075384 | FreqGPTQ: bias GPTQ Hessian calibration toward high-frequency tokens (top-100... |

## Key Insights
*(Add manually — this section is preserved across regenerations.)*
