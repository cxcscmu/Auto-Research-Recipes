# Research Knowledge Base
*Auto-generated — the `## Key Insights` section at the bottom is preserved.*

## Research Tree
Full tree is at `/home/user/auto-research/magent_state_pg_genmulti_D/blackboard/tree.tsv` (TSV; columns: `exp_id, parent_exp, depth, path, specialist, status, val_bpb, delta_vs_best, hypothesis`; preorder-sorted so siblings are adjacent). Slice it from Bash, e.g.:

```
# subtree rooted at exp_042 (all descendants + the node itself)
awk -F '\t' 'NR==1 || $4 ~ /(^|\/)042(\/|$)/' /home/user/auto-research/magent_state_pg_genmulti_D/blackboard/tree.tsv
# direct children of exp_042
awk -F '\t' 'NR==1 || $2=="042"' /home/user/auto-research/magent_state_pg_genmulti_D/blackboard/tree.tsv
# only kept trials, sorted by val_bpb
awk -F '\t' 'NR==1 || $6=="keep"' /home/user/auto-research/magent_state_pg_genmulti_D/blackboard/tree.tsv | sort -t $'\t' -k7,7g
```

**Current-best lineage** (root → best):
```
exp_000 [baseline, baseline, bpb=1.081000] generic-multi-agent baseline (PG 1.0810 SOTA stack, 10× generic)
   └─ exp_023 [genf, keep, bpb=1.079016, Δ=-0.001984] Polar Express per-iteration NS coefficients for Muon + ttt_epochs 3→2 to clear the systemic eval_bud
      └─ exp_037 [geni, keep, bpb=1.078936, Δ=-0.000080] Reduce GPTQ Hessian damping 0.01→0.005 on top of exp_023 (Polar Express NS + ttt_epochs=2); lower da
         └─ exp_131 [gene, keep, bpb=1.078062, Δ=-0.000874] Skip "quantized" eval when TTT enabled (~40s saved) + constant TTT LR (no cosine decay; uniform adap
            └─ exp_146 [genh, keep, bpb=1.076155, Δ=-0.001744] Add sparse per-head attention gate (arXiv:2505.06708, Qwen GatedAttn, NeurIPS 2025 Oral) to all 11 b
               └─ exp_164 [gena, keep, bpb=1.075317, Δ=-0.000313] Add min_lr=0.1 on top of exp_146 (attention gate baseline): floors warmdown LR at 10% of peak, same 
                  └─ exp_176 [gena, size_blocked, bpb=1.073593] ResFormer value residual (arXiv:2410.17897) on top of exp_164: pre-compute v0 from block-0's c_v app
                     └─ exp_186 [gena, keep, bpb=1.074571, Δ=-0.000322] ResFormer value residual (exp_176 follow-up): add LOWBIT_LAYERS support and default to int5 for bloc
                        └─ exp_204 [geni, keep, bpb=1.073695, Δ=-0.000647] Add scalar input-conditioned MLP output gate (from exp_169 branch) to exp_186 (value residual base):
                           └─ exp_215 [gena, keep, bpb=1.072768, Δ=-0.000223] Bifurcated QK-gain [H,2]: independent temperature scales for RoPE region (positional dims 0-15) vs N
                              └─ exp_240 [genb, keep, bpb=1.072713, Δ=-0.000055] Per-head value residual alpha: upgrade scalar vr_alpha per-layer to shape [num_kv_heads=4], so each   ← BEST

**Best's direct children (11):**
```

## Recent Activity (last 30)

| exp | specialist | status | val_bpb | hypothesis |
|-----|------------|--------|---------|------------|

## Key Insights
*(Add manually — this section is preserved across regenerations.)*
