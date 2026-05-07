# Research Knowledge Base
*Auto-generated — the `## Key Insights` section at the bottom is preserved.*

## Research Tree
Full tree is at `/home/user/auto-research/magent_state_nc/blackboard/tree.tsv` (TSV; columns: `exp_id, parent_exp, depth, path, specialist, status, core_metric, delta_vs_best, hypothesis`; preorder-sorted so siblings are adjacent). Slice it from Bash, e.g.:

```
# subtree rooted at exp_042 (all descendants + the node itself)
awk -F '\t' 'NR==1 || $4 ~ /(^|\/)042(\/|$)/' /home/user/auto-research/magent_state_nc/blackboard/tree.tsv
# direct children of exp_042
awk -F '\t' 'NR==1 || $2=="042"' /home/user/auto-research/magent_state_nc/blackboard/tree.tsv
# only kept trials, sorted by core_metric
awk -F '\t' 'NR==1 || $6=="keep"' /home/user/auto-research/magent_state_nc/blackboard/tree.tsv | sort -t $'\t' -k7,7g
```

**Current-best lineage** (root → best):
```
exp_000 [baseline, baseline, core=0.161800] calibrated baseline (1 trial, avg=0.161800)
   └─ exp_007 [sys, keep, core=0.169500, Δ=+0.007700] Switch --window-pattern from SSSL to L: with PyTorch SDPA (no FA3 on environment), any sliding-window mask f
      └─ exp_020 [sys, keep, core=0.202900, Δ=+0.033400] Exploit window=L speedup: increase --target-param-data-ratio from 12 to 100, training 8x more tokens
         └─ exp_025 [sched, keep, core=0.224100, Δ=+0.010200] Push --target-param-data-ratio from 100→130 on top of exp_020 (window=L, ratio=100, 0.2029): 30% mor
            └─ exp_156 [arch, keep, core=0.224400, Δ=+0.000300] Add learnable logit_bias (32768 params, zero-init) applied after lm_head before softcap, on clean ex  ← BEST

**Best's direct children (65):**
  • exp_161 [arch, discard, core=0.198500, Δ=-0.025900] Add learnable per-feature output norm scale (ln_f_scale, 768 params, ones-init) applied after final 
  • exp_162 [arch, preflight_crash, core=—] Add learnable per-layer attention temperature (qk_scale, 12 params, init=1.2) in CausalSelfAttention
  • exp_163 [sys, train_budget_overrun, core=—] Skip all intermediate val_bpb (eval-every=-1) and CORE evals (core-metric-every=99999, final still t
  • exp_165 [opt, discard, core=0.218800, Δ=-0.005600] Raise logit_bias LR from 0.008 (unembedding_lr) to 0.04 and beta2 from 0.96 to 0.99 on exp_156 base:
  • exp_166 [data, discard, core=0.205500, Δ=-0.018900] Increase BOS-bestfit buffer_size 1000→4000 on clean exp_156 base: 4× more candidates dramatically in
  • exp_167 [arch, discard, core=0.207500, Δ=-0.016900] Add learnable per-layer attention temperature (qk_scale, 12 params, init=1.2) in CausalSelfAttention
  • exp_168 [sys, discard, core=0.210500, Δ=-0.013900] Skip all intermediate val_bpb (eval-every=-1) and CORE evals (core-metric-every=99999) to reclaim ev
  • exp_169 [opt, discard, core=0.209900, Δ=-0.014500] Apply C-AdamW cautious masking to adamw_step_fused: only update elements where exp_avg and grad agre
  • exp_171 [sys, discard, core=0.208300, Δ=-0.016100] Skip val_bpb evals (eval-every=-1, saves ~7-8 min) and redirect budget to ratio 130→140 (+7.7% token
  • exp_172 [data, discard, core=0.201600, Δ=-0.022800] Clip documents to row_capacity (T+1=2049 tokens) before buffering: docs longer than row_capacity can
  • exp_173 [arch, discard, core=0.206400, Δ=-0.018000] Key offset in attention: second half of each head's key dims at position t ← position t-1's key valu
  • exp_174 [sched, discard, core=0.207800, Δ=-0.016600] Change final-lr-frac from 0.05→0.0 on clean exp_156 base: true linear decay-to-zero for the 65% warm
  • exp_175 [opt, discard, core=0.216400, Δ=-0.008000] Raise lm_head and logit_bias AdamW beta2 from 0.96→0.98 on clean exp_156 base: lm_head receives cros
  • exp_176 [arch, discard, core=0.204000, Δ=-0.020400] Replace relu^2 MLP with SwiGLU (8/3 expansion, same param+FLOP count): 3*(n_embd*(8/3)*n_embd) = 2*(
  • exp_177 [data, discard, core=0.203400, Δ=-0.021000] Mask loss at cross-document boundary positions (set target=-1, ignore_index=-1 in gpt.py): the last 
  • exp_178 [sys, discard, core=0.203100, Δ=-0.021300] Add mode="max-autotune-no-cudagraphs" to torch.compile: Inductor benchmarks GEMM/attention kernel im
  • exp_179 [sched, discard, core=0.206200, Δ=-0.018200] Add AdamW β₂ warmdown (initial→0.999 linearly during warmdown) on clean exp_156 base: at low LR, gra
  • exp_180 [opt, discard, core=0.201000, Δ=-0.023400] Remove cautious WD mask from muon_step_fused: switch from mask=(g*params>=0) gated WD to standard de
  • exp_181 [arch, discard, core=0.200000, Δ=-0.024400] Extend value embeddings (VE/ResFormer) from alternating 6 layers to all 12 layers: change has_ve() t
  • exp_182 [data, discard, core=0.191800, Δ=-0.032600] Revert exp_177's cross-document boundary loss masking: remove all_bos_positions tracking + cpu_targe
  • exp_183 [sys, discard, core=0.211300, Δ=-0.013100] Revert max-autotune-no-cudagraphs compile regression (exp_178 added it, −0.021 vs exp_156). PyTorch 
  • exp_184 [sched, discard, core=0.198100, Δ=-0.026300] Increase --warmup-steps from 40→200 on clean exp_156 base: at ratio=130 (≈27,274 steps), 40-step war
  • exp_185 [opt, discard, core=0.203200, Δ=-0.021200] Raise value_embeds AdamW LR from 0.5×→1.0× embedding_lr (0.15→0.30) on clean exp_156 base: value_emb
  • exp_186 [arch, discard, core=0.200800, Δ=-0.023600] Add post-norm (sandwich norm) to Block.forward: wrap each sub-layer output with norm() before residu
  • exp_187 [sys, discard, core=0.211400, Δ=-0.013000] Enable FP8 tensorwise training (--fp8) on exp_156 base, excluding lm_head from FP8 conversion to pro
  • exp_188 [data, discard, core=0.203800, Δ=-0.020600] Add document tail recycling: when a document is cropped to fill the remaining row space, prepend BOS
  • exp_189 [sched, discard, core=0.181600, Δ=-0.042800] Switch Muon weight-decay schedule from cosine-decay-to-zero to constant (weight_decay_scaled) on cle
  • exp_190 [opt, discard, core=0.210600, Δ=-0.013800] Raise Muon NorMuon variance-reduction beta2 from 0.9→0.95 on clean exp_156 base: the NorMuon paper (
  • exp_191 [arch, train_budget_overrun, core=—] Differential Attention (arXiv:2410.05258, ICLR 2025): split Q,K,V along head_dim into halves (64-dim
  • exp_192 [sched, preflight_crash, core=—] Reduce warmdown-ratio 0.65→0.50 and change LR warmdown from linear to sqrt shape on exp_156 base: sq
  • exp_193 [sched, preflight_crash, core=—] Reduce warmdown-ratio 0.65→0.50 and change LR warmdown from linear to sqrt shape on exp_156 base: sq
  • exp_194 [arch, preflight_crash, core=—] Add per-head attention output gate (12×n_head Linear, AdamW) on clean exp_156 base: sigmoid gate con
  • exp_195 [sched, preflight_crash, core=—] Replace linear LR warmdown with cosine warmdown (half-cosine from 1.0→final_lr_frac over warmdown_ra
  • exp_196 [sys, preflight_crash, core=—] Reduce --eval-tokens 10× (80→8 ×524288) to reclaim ~6-10 min of val-bpb eval overhead, redirecting b
  • exp_197 [sched, preflight_crash, core=—] Replace linear LR warmdown with cosine warmdown (half-cosine from 1.0→final_lr_frac over warmdown_ra
  • exp_198 [sys, preflight_crash, core=—] Reduce --eval-tokens 10× (80→8×524288, saves ~6-10 min) and redirect budget to ratio 130→155 (+19% t
  • exp_199 [opt, preflight_crash, core=—] Add stateless MuonEq-R/C pre-orthogonalization equilibration to muon_step_fused: for tall matrices n
  • exp_200 [opt, preflight_crash, core=—] Add stateless MuonEq-R/C pre-orthogonalization equilibration to muon_step_fused: for tall matrices n
```

## Recent Activity (last 30)

| exp | specialist | status | core_metric | hypothesis |
|-----|------------|--------|---------|------------|
| 200 | opt | preflight_crash | — | Add stateless MuonEq-R/C pre-orthogonalization equilibration to muon_step_fus... |
| 199 | opt | preflight_crash | — | Add stateless MuonEq-R/C pre-orthogonalization equilibration to muon_step_fus... |
| 198 | sys | preflight_crash | — | Reduce --eval-tokens 10× (80→8×524288, saves ~6-10 min) and redirect budget t... |

## Key Insights
*(Add manually — this section is preserved across regenerations.)*
