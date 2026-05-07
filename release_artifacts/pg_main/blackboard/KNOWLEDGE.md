# Research Knowledge Base
*Auto-generated — the `## Key Insights` section at the bottom is preserved.*

## Research Tree
Full tree is at `/home/user/auto-research/magent_state/blackboard/tree.tsv` (TSV; columns: `exp_id, parent_exp, depth, path, specialist, status, val_bpb, delta_vs_best, hypothesis`; preorder-sorted so siblings are adjacent). Slice it from Bash, e.g.:

```
# subtree rooted at exp_042 (all descendants + the node itself)
awk -F '\t' 'NR==1 || $4 ~ /(^|\/)042(\/|$)/' /home/user/auto-research/magent_state/blackboard/tree.tsv
# direct children of exp_042
awk -F '\t' 'NR==1 || $2=="042"' /home/user/auto-research/magent_state/blackboard/tree.tsv
# only kept trials, sorted by val_bpb
awk -F '\t' 'NR==1 || $6=="keep"' /home/user/auto-research/magent_state/blackboard/tree.tsv | sort -t $'\t' -k7,7g
```

**Current-best lineage** (root → best):
```
exp_000 [baseline, baseline, bpb=1.0810] 1.0810 SOTA reference
   └─ exp_005 [arch, keep, bpb=1.078859, Δ=-0.000788] Increase partial RoPE coverage: rope_dims 16→32 (half of head_dim=64, the common 50/50 RoPE/NoPE spl
      └─ exp_017 [eval, keep, bpb=1.078059, Δ=-0.000413] Increase TTT epochs per chunk 3→4: more per-chunk SGD steps let the model adapt more thoroughly on e
         └─ exp_088 [opt, keep, bpb=1.076944, Δ=-0.001115] Shrink muon_momentum_warmup_fraction 0.33→0.22 so Muon momentum reaches peak 0.99 BEFORE warmdown be
            └─ exp_108 [quant, keep, bpb=1.076865, Δ=-0.000079] Shrink GPTQ block_size 128→64 for finer error propagation (2× more error-update events per matrix, e
               └─ exp_142 [curr, keep, bpb=1.076563, Δ=-0.000028] Latin-square deterministic phase cycling in ShuffledSequenceLoader: per-shard, cycle through 8 evenl
                  └─ exp_151 [ttt, keep, bpb=1.076291, Δ=-0.000272] Enable Nesterov momentum on the TTT SGD optimizer (nesterov=True) so the ~16 SGD steps/chunk lookahe
                     └─ exp_164 [reg, keep, bpb=1.076067, Δ=-0.000024] Split Muon weight decay by parameter TYPE: keep muon_wd=0.095 on attention matrices, raise to 0.125 
                        └─ exp_177 [meta, keep, bpb=1.075858, Δ=-0.000038] Reduce adam_wd 0.02→0.005 (4×) on scalar-params-only AdamW group: ln_scale (init=1.0), qk_gain (init
                           └─ exp_196 [curr, keep, bpb=1.075246, Δ=-0.000087] Stratified per-batch shard sampling in ShuffledSequenceLoader.next_batch (largest-remainder method):
                              └─ exp_212 [quant, keep, bpb=1.075022, Δ=-0.000224] Raise GPTQ Hessian damping 0.01→0.02 (2×): exp_016 already showed 0.01→0.003 hurt bpb by +0.000354, 
                                 └─ exp_245 [arch, keep, bpb=1.074750, Δ=-0.000272] Recurrence-aware ln_scale correction: divide ln_scale_factor by sqrt(num_loops+1)=sqrt(3) for looped
                                    └─ exp_267 [tok, keep, bpb=1.074604, Δ=-0.000036] Port exp_190's proven per-feature input-side embed_scale (shape=[embedding_dim], init=1.0) onto the 
                                       └─ exp_280 [quant, keep, bpb=1.074550, Δ=-0.000054] Replace flat additive Hessian damping (0.02·mean(diag)) with signal-proportional multiplicative damp
                                          └─ exp_298 [tok, keep, bpb=1.074482, Δ=-0.000068] Move embed_scale INSIDE the post-tok_emb RMSNorm: replace `F.rms_norm(x) * embed_scale` with `F.rms_
                                             └─ exp_324 [opt, keep, bpb=1.074280, Δ=-0.000202] Floor Muon-momentum warmdown blend at 0.25 (momentum floor 0.9375 vs 0.92) so the deepest ~18% of tr
                                                └─ exp_336 [meta, keep, bpb=1.074171, Δ=-0.000109] Lower muon_wd_mlp 0.125 → 0.115 (−8%) on MLP-only Muon WD group: probes the untested DOWN direction 
                                                   └─ exp_377 [quant, keep, bpb=1.074087, Δ=-0.000084] Lower GPTQ multiplicative Hessian damping 1.02→1.01 (−50% damping strength): pure scalar downward pr
                                                      └─ exp_475 [arch, keep, bpb=1.074021, Δ=-0.000066] Bifurcated q_gain [num_heads, 2]: one scalar per head for RoPE region (first rope_dims=16 of head_di
                                                         └─ exp_491 [ttt, keep, bpb=1.073978, Δ=-0.000043] Freeze control-scalar params (numel < 10000) during TTT: q_gain [H,2]=16, embed_scale 512, skip_weig
                                                            └─ exp_538 [arch, keep, bpb=1.073466, Δ=-0.000512] PR1667 per-head data-dependent attention-output gate as raw nn.Parameter
                                                               └─ exp_596 [loss, keep, bpb=1.072251, Δ=-0.000188] Retry exp_587 TTT-only z-loss (lambda=1e-5 * logsumexp^2 on eval_val_ttt inner CE) which measured va
                                                                  └─ exp_750 [opt, keep, bpb=1.072210, Δ=-0.000036] Decouple Muon momentum warmdown cool target from warmup_start: hard-code 0.93 on the mm_blend cool l  ← BEST

**Best's direct children (151):**
  • exp_759 [opt, discard, bpb=1.072612, Δ=+0.000402] Per-group Muon momentum cool target — attn (group 0) keeps 0.93 (kept exp_750), mlp (group 1) deepen
  • exp_760 [meta, size_blocked, bpb=1.072797] Per-group BETA2 split on optimizer_tok (tied tok_emb): lower beta2 0.95 to 0.93 at L908 only, leavin
  • exp_761 [reg, size_blocked, bpb=1.072514] Per-(chunk,epoch) random shuffle of TTT inner-loop batch start indices via seeded torch.Generator: r
  • exp_762 [eval, size_blocked, bpb=1.072392] Symmetric (doubly-centered) Gradient Centralization on TTT SGD: subtract BOTH per-row mean AND per-c
  • exp_763 [ttt, size_blocked, bpb=1.072280] Mean-preserving front-loaded z-loss schedule across 4 TTT epochs.
  • exp_764 [curr, size_blocked, bpb=1.073102] Replace random self.rng.shuffle(shard_plan) with deterministic Bresenham largest-error interleave: e
  • exp_765 [loss, size_blocked, bpb=1.072457] Gate kept exp_596 TTT z-loss to epochs>=1 only.
  • exp_766 [tok, size_blocked, bpb=1.075950] WARP-inspired word-start additive bias [embedding_dim=512, init=0] applied to tok_emb output ONLY on
  • exp_767 [quant, discard, bpb=1.072636, Δ=+0.000426] Eval-aligned position weighting on GPTQ Hessian: weight last-64 calibration positions by 10x to alig
  • exp_768 [arch, size_blocked, bpb=1.076752] Block-major depth recurrence: loop order [3,3,3,4,4,4,5,5,5] instead of [3,4,5,3,4,5,3,4,5] — each l
  • exp_769 [ttt, size_blocked, bpb=1.073654] Re-introduce exp_588's TTT chunk-boundary momentum_buffer halving (mb.mul_(0.5) for ci>0): proven Δ=
  • exp_770 [opt, discard, bpb=1.076457, Δ=+0.004247] Anchor Muon weight_decay shrinkage during warmdown by setting wd_dyn = wd / max(lr_scale, 0.5) per s
  • exp_771 [loss, discard, bpb=1.073297, Δ=+0.001087] Replace TTT-only z-loss L2 form (1e-5*logsumexp.pow(2).mean) with L1 form (1e-4*logsumexp.mean) on e
  • exp_772 [eval, size_blocked, bpb=1.072482] Score-phase logit softening (mult 0.99, T=1.0101) on BOTH eval_val_sliding and eval_val_ttt score-ph
  • exp_773 [reg, size_blocked, bpb=1.073251] Lower muon_wd_mlp 0.115 → 0.108 (-6.1% DOWN extension probe of exp_336's kept -8% DOWN move on this 
  • exp_774 [curr, discard, bpb=1.073200, Δ=+0.000990] Change Latin-square phase bucket count 8 to 11 (prime, coprime with seq_len=2048).
  • exp_775 [tok, discard, bpb=1.075152, Δ=+0.002942] Tied-symmetric gauge — reuse input-side embed_scale [512] at the tied LM head readout via F.linear(x
  • exp_776 [meta, size_blocked, bpb=1.073412] Raise mm_blend floor 0.25→0.27 — untested midpoint between current best floor=0.25 kept and exp_562 
  • exp_777 [quant, eval_budget_overrun, bpb=1.072454] Fisher-weighted GPTQ Hessian: weight per-token x⊗x by ||∂L/∂y_t||² (output-side gradient norm) on at
  • exp_778 [arch, size_blocked, bpb=1.072971] Per-pass loop offset: single learnable [model_dim] vector (zero-init), modulated by hard-coded linea
  • exp_779 [opt, size_blocked, bpb=1.073478] Per-group Muon mm_blend SHAPE split: attn (g0) uses sqrt-with-floor (max(lr_scale**0.5, 0.25)), mlp 
  • exp_780 [ttt, size_blocked, bpb=1.073582] Leaky-anchor pull-back at TTT chunk boundary lerp params 0.5pct toward post-quant initial weights af
  • exp_781 [tok, discard, bpb=1.074700, Δ=+0.002490] Add untied per-feature output head_scale [embedding_dim, init=1.0] applied AFTER final_norm at tied 
  • exp_782 [eval, size_blocked, bpb=1.073617] Freeze the 3 looped blocks (loop_start..loop_end = 3..5) during eval-time TTT SGD, so adaptation onl
  • exp_783 [curr, size_blocked, bpb=1.073100] Mean-preserving token-id-stratified train CE: w=0.9+0.2*(target_id/8191) probes Gap 5 rare-token upw
  • exp_784 [loss, discard, bpb=1.073470, Δ=+0.001260] Replace TTT-only z-loss L2 form (1e-5*logsumexp^2.mean) with L4 form (6e-8*logsumexp^4.mean) — coef 
  • exp_785 [meta, size_blocked, bpb=1.073114] grad_clip_norm 0.3→0.28 — untested gentle DOWN midpoint between current 0.3 and exp_525's size_block
  • exp_786 [reg, size_blocked, bpb=1.072267] Tighten TTT inner-loop grad_clip_norm 1.0 → 0.8: untested middle-DOWN probe between exp_244/180/351 
  • exp_787 [quant, size_blocked, bpb=1.071334] Narrow AWQ-style per-row scale-grid refinement (alpha in {0.95, 1.0, 1.05}) before GPTQ inner loop, 
  • exp_788 [arch, size_blocked, bpb=1.072608] Asymmetric LayerScale init for parallel-residual blocks (7-10): mlp_scale starts at 0.5 instead of 1
  • exp_789 [ttt, eval_budget_overrun, bpb=1.072417] Mean-preserving per-epoch linear LR decay within each TTT chunk (1.25x→0.75x of cos_lr across the 4 
  • exp_790 [opt, size_blocked, bpb=1.072797] Schedule optimizer_tok (AdamW on tied tok_emb) beta2 from 0.95 (peak-LR) up to 0.99 (deepest tail) l
  • exp_791 [quant, size_blocked, bpb=1.071846] Threshold-gated AWQ-style per-row scale refinement: keep grid alpha in {0.95,1.0,1.05} but only swap
  • exp_792 [meta, size_blocked, bpb=1.071951] SCALAR_LR 0.02 → 0.018 (gentle DOWN -10%, untested midpoint between current and exp_204's failed -25
  • exp_793 [arch, size_blocked, bpb=1.072819] Milder asymmetric init for parallel-residual blocks (7-10): mlp_scale=0.8 (vs 1.0 baseline, attn=1.0
  • exp_794 [opt, size_blocked, bpb=1.073069] Per-group LR floor on optimizer_tok only at 5% of base_lr — keeps tied tok_emb's rare-token rows sti
  • exp_795 [eval, size_blocked, bpb=1.072801] Stack Gradient Centralization (zero-mean per-row gradient on 2D+ ttt_params before clip+step) onto c
  • exp_796 [loss, discard, bpb=1.079557, Δ=+0.007347] Temperature-scale TTT inner CE: F.cross_entropy(_lf/1.1,...) — softens TTT per-chunk targets to T=1
  • exp_797 [quant, size_blocked, bpb=1.071147] Pair brotli lgwin=22→24 (untried alone on current parent; estimated 150–500KB savings on the ~16MB c
  • exp_798 [curr, size_blocked, bpb=1.074176] Stride+1 packing in _reset_shard: stride from seq_len (2048) to seq_len+1 (2049). Decorrelates seq-p
  • exp_799 [tok, discard, bpb=2.557347, Δ=+1.485137] Forward-time per-row RMS-equalized tied-readout: F.linear(x, F.rms_norm(tok_emb.weight, (D,)) * tied
  • exp_800 [reg, discard, bpb=1.072391, Δ=+0.000181] adam_wd 0.005 to 0.006 gentle UP midpoint byte-neutral probe
  • exp_801 [ttt, size_blocked, bpb=1.079181] Hard-mask TTT inner CE to last 128 positions of each 2048-token training sequence; aligns adaptation
  • exp_802 [meta, size_blocked, bpb=1.072196] ADAM_EPS 1e-8 → 1e-7 (untouched UP direction): only DOWN was probed (exp_304 1e-8→1e-9 discard +0.00
  • exp_803 [arch, discard, bpb=1.073728, Δ=+0.001518] Loop-iteration-aligned U-Net split: shift num_enc 8->9 (len(all_indices)//2 + 1) so encoder ends at 
  • exp_804 [opt, size_blocked, bpb=1.072548] Per-group Muon matrix_lr split: introduce matrix_lr_mlp=0.023 for MLP Muon group while attn group ke
  • exp_805 [reg, size_blocked, bpb=1.073592] TTT input-token dropout p=0.03 mask-to-0 before forward_logits in eval_val_ttt train phase: token-le
  • exp_806 [ttt, discard, bpb=1.077586, Δ=+0.005376] Recurrence-aware per-block-group TTT LR: divide TTT lr by sqrt(num_loops+1)=sqrt(3) on looped blocks
  • exp_807 [loss, discard, bpb=1.073486, Δ=+0.001276] Mismatch-gated mean-preserving TTT z-loss: replace `1e-5*lse².mean()` with `1e-5*sum(lse²·1[ŷ≠y])/su
  • exp_808 [tok, size_blocked, bpb=1.075415] Halve gradient on tied tok_emb classifier path (W_cls = 0.5*W + 0.5*W.detach()): lookup path retains
  • exp_809 [quant, size_blocked, bpb=1.081207] Stochastic (unbiased) rounding in GPTQ inner loop: replace torch.round(w_col/sf) with torch.floor(w_
  • exp_810 [meta, discard, bpb=1.075748, Δ=+0.003538] Enable amsgrad=True on optimizer_tok (tied tok_emb AdamW) only — caps v_t monotonically non-decreasi
  • exp_811 [eval, discard, bpb=1.072790, Δ=+0.000580] Replace per-row GC with per-column GC on TTT SGD: attacks per-input-feature bias mode. exp_762 doubl
  • exp_812 [arch, size_blocked, bpb=1.073935] Swap MLP activation silu²→gelu² (Primer-style x²-family kept, but use GELU's sharper shoulder vs SiL
  • exp_813 [curr, discard, bpb=1.074293, Δ=+0.002083] Predecessor-special CE mask in train_loader: y[x<3]=-100 drops CE on positions where input token is 
  • exp_814 [opt, size_blocked, bpb=1.073217] Cool target 0.93→0.94 on Muon momentum mm_blend (single-literal byte-neutral): probe untested UP mid
  • exp_815 [ttt, size_blocked, bpb=1.077717] Hard-mask TTT z-loss to last-128 positions to match CE scope: replace logsumexp.mean() over all 2048
  • exp_816 [loss, discard, bpb=1.074947, Δ=+0.002737] Byte-neutral quartic TTT z-loss: 1e-5*lse.pow(2).mean() to 1e-7*lse.pow(4).mean() at L1401
  • exp_817 [reg, size_blocked, bpb=1.072721] Add weight_decay=5e-4 to eval-TTT SGD (the only WD=0 optimizer in the pipeline) — clean retry of exp
  • exp_818 [quant, size_blocked, bpb=1.072053] H-magnitude-proportional GPTQ damping: replace uniform H.diag.mul_(1.01) with H.diag.mul_(1 + 0.01 *
  • exp_819 [meta, size_blocked, bpb=1.072611] Per-group ADAM_EPS split: optimizer_tok (tied tok_emb) eps 1e-8→1e-7 only, optimizer_scalar unchange
  • exp_820 [tok, size_blocked, bpb=1.072325] Port exp_711 (embed_scale ONLY on residual baseline x0; entry-x to block 0 plain unit-RMS) onto exp_
  • exp_821 [arch, size_blocked, bpb=1.073302] Mirror of kept exp_245's recurrence-aware ln_scale damping, applied to parallel-residual blocks: div
  • exp_822 [curr, size_blocked, bpb=1.073631] Latin-square phase bucket count 8→6 (untested between failed 5 and kept 8): coupon-collector coverag
  • exp_823 [opt, discard, bpb=1.072485, Δ=+0.000275] Polar Express per-iteration NS coefficients (PR #1787 / PR #1344): replace fixed (3.4445,-4.775,2.03
  • exp_824 [eval, size_blocked, bpb=1.072466] Raise TTT-param numel threshold 10000 to 200000, freezing K/V projections (c_k, c_v at 131,072 each)
  • exp_825 [loss, discard, bpb=1.072266, Δ=+0.000056] Stack a dormant-by-default entropy-floor hinge on the TTT inner CE: 1e-4 * F.relu(1.0 - H(p)).mean()
  • exp_826 [reg, size_blocked, bpb=1.072387] Extend exp_491's selective-freeze policy to also exclude tied tok_emb [8192,512]=4M params during ev
  • exp_827 [opt, crash, bpb=—] Recurrence-aware Muon LR: scale main-train Muon base_lr by 1/sqrt(num_loops+1)=0.577 on looped block
  • exp_828 [quant, size_blocked, bpb=1.073554] Half-amplitude rescue of exp_818's size_blocked H-magnitude-proportional damping (bpb=1.072053, Δ=-0
  • exp_829 [ttt, size_blocked, bpb=1.073632] Replace global L2-norm gradient clip (1.0) with per-element value clip (0.1) during TTT SGD: caps ou
  • exp_830 [arch, crash, bpb=—] Per-head sub-LN: F.rms_norm(y, (head_dim,)) on attention output between XSA and attn_out_gate, decou
  • exp_831 [meta, size_blocked, bpb=1.072416] SCALAR_LR 0.02 → 0.019 (gentle -5% DOWN midpoint between current and exp_792's failed -10% size_bloc
  • exp_832 [tok, size_blocked, bpb=1.073465] Reparameterize embed_scale [D] from linear (init=1) to log-exp (init=0, applied as torch.exp(log_s))
  • exp_833 [curr, size_blocked, bpb=1.073724] Length curriculum: mask y positions seq_len//2.. with -100 for first 25% of train wallclock, forcing
  • exp_834 [eval, discard, bpb=1.072545, Δ=+0.000335] Replace TTT cosine LR decay with linear (1-ci/(N-1)) across chunks — preserves total LR mass but no 
  • exp_835 [opt, size_blocked, bpb=1.075299] Recurrence-aware Muon LR (bugfix retry of exp_827): scale main-train Muon base_lr by 1/sqrt(num_loop
  • exp_836 [loss, size_blocked, bpb=1.073303] Gentle focal CE (γ=0.25, mean-preserving) on TTT inner CE only — first focal probe on TTT axis (prio
  • exp_837 [reg, size_blocked, bpb=1.073053] SGLD-style Gaussian gradient noise injected into TTT inner-loop SGD (alpha=h.ttt_lr*0.01, added pre-
  • exp_838 [quant, size_blocked, bpb=1.071830] Tight 3-point per-row GPTQ scale grid {0.96, 1.0, 1.04} on top of current best — partial rescue of e
  • exp_839 [ttt, size_blocked, bpb=1.073632] Re-add exp_588's chunk-boundary momentum_buffer halving (mb.mul_(0.5) for ci>0) onto the new exp_750
  • exp_840 [opt, discard, bpb=1.075173, Δ=+0.002963] Replace fixed Newton-Schulz coefficients (3.4445,-4.775,2.0315) with 5 per-iteration minimax-optimal
  • exp_841 [tok, eval_budget_overrun, bpb=1.076039] Add learnable per-feature bias [embedding_dim=512, init=0] after the embed-RMSNorm, completing the a
  • exp_842 [quant, size_blocked, bpb=1.076450] FreqGPTQ: weight GPTQ Hessian by sqrt(token_frequency/mean_freq) so high-frequency tokens (which dom
  • exp_843 [arch, eval_budget_overrun, bpb=1.075364] In parallel-residual blocks (7-10), MLP branch reads pre-mix x while attn keeps reading x_in: asymme
  • exp_844 [reg, discard, bpb=1.082079, Δ=+0.009869] Stochastic depth with linear schedule (max_p=0.05 at last layer): randomly replace a block's output 
  • exp_845 [meta, discard, bpb=1.074837, Δ=+0.002627] Per-group ADAM_EPS split: raise scalar-group AdamW eps 1e-8 to 1e-7 only, leave optimizer_tok at h.a
  • exp_846 [ttt, discard, bpb=1.074403, Δ=+0.002193] Add Gradient Centralization to TTT SGD: zero-mean each 2D+ parameter gradient across input dims afte
  • exp_847 [eval, discard, bpb=1.074369, Δ=+0.002159] Fuse all_reduce loop with per-row Gradient Centralization (g.sub_(g.mean(1,True)) for 2D TTT params)
  • exp_848 [loss, discard, bpb=1.109208, Δ=+0.036998] Byte-weighted TTT CE (via val_data.base_bytes_lut + has_leading_space correction, normalized to mean
  • exp_849 [curr, discard, bpb=1.074681, Δ=+0.002471] Replace per-shard _phase_epoch+_phase_order lists with a single global phase counter: all shard rese
  • exp_850 [opt, discard, bpb=1.076416, Δ=+0.004206] Replace linear LR warmdown with cosine shape: same 0→0 LR range and same total LR integral (0.5), bu
  • exp_851 [tok, discard, bpb=1.076551, Δ=+0.004341] Add output-side per-feature embed_scale_out [512, init=1.0] multiplied onto x just before the tied L
  • exp_852 [meta, discard, bpb=1.075101, Δ=+0.002891] TIED_EMBED_LR 0.03 → 0.028 (gentle DOWN -6.7%): half-step retry of exp_252's failed -13.3% probe (de
  • exp_853 [quant, discard, bpb=1.074783, Δ=+0.002573] Split GPTQ clip sigmas by layer type: MLP matrices use 15.0σ (wider, preserves outlier weights) whil
  • exp_854 [arch, eval_budget_overrun, bpb=1.074421] Q-only region-split RMSNorm: normalize Q's RoPE region [...,:rope_dims=16] and NoPE region [...,rope
  • exp_855 [loss, discard, bpb=1.079554, Δ=+0.007344] Add label_smoothing=0.02 to training CE: reduces overconfidence during training, mild regularization
  • exp_856 [ttt, discard, bpb=1.074941, Δ=+0.002731] Cosine-decay TTT SGD momentum over chunks (0.9→0.45, floored at 0.5×ttt_momentum) mirroring existing
  • exp_857 [reg, discard, bpb=1.082208, Δ=+0.009998] Muon WD taper: linearly reduce Muon weight_decay from 1.0× to 0.5× during training fraction 0.70→1.0
  • exp_858 [eval, eval_budget_overrun, bpb=1.073728] Halve TTT chunk size 65536→32768: doubles adaptation granularity (finer local specialization per 32K
  • exp_859 [curr, discard, bpb=1.074366, Δ=+0.002156] Gentle linear position-weight 0.9→1.1 on training CE loss (model.forward only; TTT and sliding eval 
  • exp_860 [tok, size_blocked, bpb=1.074181] Add learnable per-feature additive bias (embed_bias [512], init=0) after the embedding RMSNorm, comp
  • exp_861 [curr, crash, bpb=—] Dynamic batch-size curriculum: half-batch (393K tokens) during the pre-warmdown phase (first 28% of 
  • exp_862 [opt, size_blocked, bpb=1.115030] Disable Nesterov lookahead on Muon during warmdown (set nesterov=False when lr_scale<1.0): near conv
  • exp_863 [meta, discard, bpb=1.072400, Δ=+0.000190] TIED_EMBED_INIT_STD 0.005 → 0.006 (+20% UP, byte-neutral): genuinely untouched axis across 850+ tria
  • exp_864 [quant, discard, bpb=1.074517, Δ=+0.002307] Frequency-weighted GPTQ Hessian: pre-scan calibration batches to count token frequencies, then up-we
  • exp_865 [arch, eval_budget_overrun, bpb=1.074554] k_gain[Hkv,2] init=1.0 mirrors exp_475 q_gain[H,2] on K side
  • exp_866 [ttt, discard, bpb=1.078642, Δ=+0.006432] Focal loss (γ=2) for TTT inner objective: replace CE with per-token reweighted loss (_ce * (1-exp(-_
  • exp_867 [loss, discard, bpb=1.072324, Δ=+0.000114] Ramp TTT z-loss coefficient linearly from 2.5e-6 (epoch 0) to 1e-5 (epoch 3) via `1e-5*(_ep+1)/h.ttt
  • exp_868 [eval, eval_budget_overrun, bpb=1.075093] Cosine-decay TTT z-loss coefficient (1e-5→0 over chunks): cold early chunks get full logsumexp^2 reg
  • exp_869 [reg, discard, bpb=1.072353, Δ=+0.000143] Split looped-block attn WD into third Muon group: blocks [loop_start=3..loop_end=5] get muon_wd_loop
  • exp_870 [arch, crash, bpb=—] ResFormer Value Residual Learning (arXiv 2410.17897, ACL 2025): per-layer learnable scalar λ blends 
  • exp_871 [arch, crash, bpb=—] Universal-Transformer-style per-loop-pass additive bias [num_loops+1=3, model_dim=512] added to the 
  • exp_872 [quant, discard, bpb=1.072687, Δ=+0.000477] Halve GPTQ block_size 32→16: 2× more error-propagation flushes per matrix, continuing the proven 128
  • exp_873 [opt, discard, bpb=1.072811, Δ=+0.000601] Raise EMA decay from 0.9965 to 0.9975: extends the weight-averaging window from ~286 steps to ~400 s
  • exp_874 [eval, discard, bpb=1.072294, Δ=+0.000084] Stack GC (proven exp_746 keep) + within-chunk TTT cosine LR warmdown (ep_lr = cos_lr * 0.5*(1+cos(π*
  • exp_875 [meta, discard, bpb=1.072965, Δ=+0.000755] TIED_EMBED_INIT_STD 0.005 to 0.004 (-20% DOWN, byte-neutral): symmetric mirror of exp_863's +20% UP 
  • exp_876 [loss, discard, bpb=1.072554, Δ=+0.000344] Self-anchored TTT z-loss: replace 1e-5*logsumexp^2.mean() with 1e-5*(logsumexp - anchor).^2.mean() w
  • exp_878 [curr, discard, bpb=1.072820, Δ=+0.000610] Inverted linear position ramp on training CE: w[t]=1.25-0.5*t/(T-1) upweights early/short-context po
  • exp_879 [tok, discard, bpb=1.072515, Δ=+0.000305] Decouple embed_scale from x0: apply embed_scale only to working stream x entering block 0, while res
  • exp_880 [arch, size_blocked, bpb=1.072880] Universal-Transformer-style per-loop-pass additive bias [num_loops+1=3, model_dim=512] added to the 
  • exp_881 [ttt, discard, bpb=1.072970, Δ=+0.000760] TTT MLP-only adaptation: freeze all attention (Q/K/V/O) across all blocks during TTT inner loop, ada
  • exp_882 [opt, discard, bpb=1.072522, Δ=+0.000312] Raise muon_wd (attn group) from 0.095 to 0.105 (~10% UP): attn WD has never been independently probe
  • exp_883 [meta, discard, bpb=1.072729, Δ=+0.000519] ENABLE_LOOPING_AT 0.35 to 0.40 (gentle +14% UP probe on a genuinely untouched UP direction): only pr
  • exp_884 [eval, discard, bpb=1.072957, Δ=+0.000747] Stack chunk-boundary TTT momentum-buffer halving (mul_(0.5) at ci>0, exp_588 technique, Δ=-0.000727 
  • exp_885 [arch, discard, bpb=1.073904, Δ=+0.001694] Asymmetric RoPE coverage: looped blocks 3-5 get rope_dims=32 (2× the global default) while non-loope
  • exp_886 [tok, discard, bpb=1.074825, Δ=+0.002615] Add output-side per-feature head_scale [dim=512, init=1.0] applied to x before the tied LM head proj
  • exp_887 [reg, discard, bpb=1.073658, Δ=+0.001448] Lower muon_wd_mlp 0.115 → 0.108 (−6.1%), continuing exp_336's proven DOWN direction (0.125→0.115 was
  • exp_888 [quant, discard, bpb=1.072256, Δ=+0.000046] 3-level quantile GPTQ damping (correct direction): top-20% H-diag cols get 1.015x, middle 60% keep 1
  • exp_889 [curr, discard, bpb=1.074786, Δ=+0.002576] Replace Latin-square 8-bucket phase cycling with golden-ratio Weyl sequence: phase=int((si+k)*φ%1*ma
  • exp_890 [loss, discard, bpb=1.072465, Δ=+0.000255] Entropy-floor hinge at 4.0 nats stacked on TTT z-loss: 1e-4*F.relu(4-H(p)).mean() where H(p)=lse-sum
  • exp_891 [ttt, eval_budget_overrun, bpb=1.072598] Re-introduce exp_588's TTT chunk-boundary momentum_buffer halving (mb.mul_(0.5) for ci>0): proven Δ=
  • exp_892 [opt, discard, bpb=1.072896, Δ=+0.000686] Shorten warmdown_frac from 0.72 to 0.67: expands the stable peak-LR+peak-momentum window from 6% to 
  • exp_893 [meta, discard, bpb=1.072999, Δ=+0.000789] ADAM_EPS 1e-8 → 2e-8 (2× UP gentle midpoint between current 1e-8 and exp_802's 1e-7 size_blocked-at-
  • exp_894 [quant, discard, bpb=1.072365, Δ=+0.000155] Continuous proportional H damping (exp_818's formula: hd *= 1+0.01*hd/mean(hd), mean=1.01) paired wi
  • exp_895 [eval, discard, bpb=1.072261, Δ=+0.000051] Raise TTT gradient clip threshold 1.0→2.0: exp_244 showed tighter clip (0.5) hurt BPB by +0.000489, 
  • exp_896 [arch, size_blocked, bpb=1.072806] Switch attn-output gate input from x[..., :12] (pre-attn input) to v.reshape(...,-1)[..., :12] (V pr
  • exp_897 [reg, discard, bpb=1.073045, Δ=+0.000835] Zero WD on attn_out_gate_w: split 88 zero-init per-head sigmoid gate params out of scalar AdamW grou
  • exp_898 [tok, size_blocked, bpb=1.073582] Add output-side gauge-fixed out_scale [dim=512, init=1.0] applied INSIDE final_norm (F.rms_norm(x * 
  • exp_899 [curr, discard, bpb=1.073384, Δ=+0.001174] Port exp_186's u² position-weighted CE (pw=0.9+0.2*u², normalized to mean 1.0) onto exp_750 baseline
  • exp_900 [ttt, discard, bpb=1.072974, Δ=+0.000764] Alternate TTT batch-traversal order across per-chunk epochs (fwd on even, rev on odd): decorrelates 
```

## Recent Activity (last 30)

| exp | specialist | status | val_bpb | hypothesis |
|-----|------------|--------|---------|------------|
| 900 | ttt | discard | 1.072974 | Alternate TTT batch-traversal order across per-chunk epochs (fwd on even, rev... |
| 899 | curr | discard | 1.073384 | Port exp_186's u² position-weighted CE (pw=0.9+0.2*u², normalized to mean 1.0... |
| 898 | tok | size_blocked | 1.073582 | Add output-side gauge-fixed out_scale [dim=512, init=1.0] applied INSIDE fina... |
| 897 | reg | discard | 1.073045 | Zero WD on attn_out_gate_w: split 88 zero-init per-head sigmoid gate params o... |
| 896 | arch | size_blocked | 1.072806 | Switch attn-output gate input from x[..., :12] (pre-attn input) to v.reshape(... |
| 895 | eval | discard | 1.072261 | Raise TTT gradient clip threshold 1.0→2.0: exp_244 showed tighter clip (0.5)... |
| 894 | quant | discard | 1.072365 | Continuous proportional H damping (exp_818's formula: hd *= 1+0.01*hd/mean(hd... |
| 893 | meta | discard | 1.072999 | ADAM_EPS 1e-8 → 2e-8 (2× UP gentle midpoint between current 1e-8 and exp_802'... |
| 892 | opt | discard | 1.072896 | Shorten warmdown_frac from 0.72 to 0.67: expands the stable peak-LR+peak-mome... |
| 891 | ttt | eval_budget_overrun | 1.072598 | Re-introduce exp_588's TTT chunk-boundary momentum_buffer halving (mb.mul_(0.... |
| 890 | loss | discard | 1.072465 | Entropy-floor hinge at 4.0 nats stacked on TTT z-loss: 1e-4*F.relu(4-H(p)).me... |
| 889 | curr | discard | 1.074786 | Replace Latin-square 8-bucket phase cycling with golden-ratio Weyl sequence:... |
| 888 | quant | discard | 1.072256 | 3-level quantile GPTQ damping (correct direction): top-20% H-diag cols get 1.... |
| 887 | reg | discard | 1.073658 | Lower muon_wd_mlp 0.115 → 0.108 (−6.1%), continuing exp_336's proven DOWN dir... |
| 886 | tok | discard | 1.074825 | Add output-side per-feature head_scale [dim=512, init=1.0] applied to x befor... |
| 885 | arch | discard | 1.073904 | Asymmetric RoPE coverage: looped blocks 3-5 get rope_dims=32 (2× the global d... |
| 884 | eval | discard | 1.072957 | Stack chunk-boundary TTT momentum-buffer halving (mul_(0.5) at ci>0, exp_588... |
| 883 | meta | discard | 1.072729 | ENABLE_LOOPING_AT 0.35 to 0.40 (gentle +14% UP probe on a genuinely untouched... |
| 882 | opt | discard | 1.072522 | Raise muon_wd (attn group) from 0.095 to 0.105 (~10% UP): attn WD has never b... |
| 881 | ttt | discard | 1.072970 | TTT MLP-only adaptation: freeze all attention (Q/K/V/O) across all blocks dur... |

## Key Insights
*(Add manually — this section is preserved across regenerations.)*
