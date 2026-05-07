# Research Knowledge Base
*Auto-generated — the `## Key Insights` section at the bottom is preserved.*

## Research Tree
Full tree is at `/home/user/auto-research/magent_state_pg_nolineage_B/blackboard/tree.tsv` (TSV; columns: `exp_id, parent_exp, depth, path, specialist, status, val_bpb, delta_vs_best, hypothesis`; preorder-sorted so siblings are adjacent). Slice it from Bash, e.g.:

```
# subtree rooted at exp_042 (all descendants + the node itself)
awk -F '\t' 'NR==1 || $4 ~ /(^|\/)042(\/|$)/' /home/user/auto-research/magent_state_pg_nolineage_B/blackboard/tree.tsv
# direct children of exp_042
awk -F '\t' 'NR==1 || $2=="042"' /home/user/auto-research/magent_state_pg_nolineage_B/blackboard/tree.tsv
# only kept trials, sorted by val_bpb
awk -F '\t' 'NR==1 || $6=="keep"' /home/user/auto-research/magent_state_pg_nolineage_B/blackboard/tree.tsv | sort -t $'\t' -k7,7g
```

**Current-best lineage** (root → best):
```
exp_075 [opt, keep, bpb=1.077413, Δ=-0.000186] Raise min_lr warmdown floor from 0.004 to 0.10 so late-training EMA steps see 10% of peak LR instead  ← BEST

**Best's direct children (1):**
  • exp_132 [loss, discard, bpb=1.121509, Δ=+0.044096] Add label smoothing eps=0.1 to the CE loss: redistributes 10% of target mass uniformly, which per ne
```

## Recent Activity (last 30)

| exp | specialist | status | val_bpb | hypothesis |
|-----|------------|--------|---------|------------|
| 200 | meta | size_blocked | 1.082119 | Bump scalar_lr 0.02 → 0.030 (1.36× matrix_lr) so dense-gradient 1D modulators... |
| 199 | eval | discard | 1.078465 | Revert ttt_epochs to 3 (4 was marginally worse) and reduce eval_stride from 6... |
| 198 | ttt | discard | 1.159667 | Replace TTT optimizer from SGD+Nesterov (lr=0.005) to AdamW (lr=0.001, wd=0.0... |
| 197 | arch | crash | — | Add Universal-Transformer per-iteration step embedding to the 3-layer recurre... |
| 196 | curr | crash | — | Extend half-batch curriculum from frac=0.35 to frac=0.65 (decoupled from loop... |
| 195 | arch | eval_budget_overrun | 1.090102 | Reshuffle layer-wise RoPE schedule from [24,24,24,16,16,16,16,16,8,8,8] to [2... |
| 194 | reg | eval_budget_overrun | 1.090155 | Mechanism swap: replace weak mlp_global_drop=0.02 (activation noise in non-lo... |
| 193 | quant | eval_budget_overrun | 1.080911 | Use float64 for GPTQ Hessian Cholesky factorization instead of float32, impro... |
| 192 | opt | discard | 1.097275 | Set min_lr=0.0 (linear decay-to-zero) instead of 0.02, following arxiv 2502.1... |
| 191 | loss | crash | — | Focal-weighted NLL (gamma=0.5) via log_softmax+gather+detach (fullgraph-compi... |
| 190 | loss | crash | — | Replace flat cross-entropy with focal-weighted NLL (gamma=0.5): tokens where... |
| 189 | meta | size_blocked | 1.082358 | Bump tied_embed_init_std from 0.014 to 0.020 (GPT-2 default) so rare-token em... |
| 188 | eval | discard | 1.078784 | Skip torch.compile(eval_model) + eval_val('quantized') + eval_val_sliding whe... |
| 187 | curr | eval_budget_overrun | 1.078959 | Extend batch-curriculum half-batch phase from 20% to 35% of wallclock (aligne... |
| 186 | tok | eval_budget_overrun | 1.083905 | Fix fourgram channel to use pure normalized embeddings (consistent with other... |
| 185 | ttt | discard | 1.084026 | Replace standard cross-entropy with focal loss (gamma=2) in TTT adaptation st... |
| 184 | loss | discard | 1.079481 | Remove label_smoothing entirely (0.05→0.0) from F.cross_entropy: pure NLL tra... |
| 183 | arch | eval_budget_overrun | 1.088186 | Long-short SWA: layers 0,1,2,6,7,8 use FA3 causal-local window=512; recurrenc... |
| 182 | opt | discard | 1.097575 | Replace cosine warmdown (min_lr=0.10) with linear warmdown (min_lr=0.02): per... |
| 181 | quant | size_blocked | 1.075468 | Reduce SDClip sigmas for MLP (12.85→9.5) and attention (12.0→9.0) to improve... |
| 180 | reg | eval_budget_overrun | 1.094077 | Extend WD taper to the scalar optimizer (q_gain, resid_mix, attn_scale, mlp_s... |
| 179 | eval | eval_budget_overrun | 1.097271 | Skip redundant eval_val_sliding compile+scoring when TTT is enabled (saves ~1... |
| 178 | meta | size_blocked | 1.082639 | Lower ema_decay 0.999→0.997 (half-life ~693→~230 steps) to remove ~9% random-... |
| 177 | tok | eval_budget_overrun | 1.084170 | Fix fourgram channel consistency: compute fourgram_emb from pure RMSNorm'd em... |
| 176 | curr | crash | — | Extend batch-curriculum half-batch phase from 20% to 70% of wallclock, based... |
| 175 | ttt | discard | 1.079240 | Reduce ttt_freeze_layers from 5 to 3 (SGD TTT unchanged), adapting the depth-... |
| 174 | loss | discard | 1.088974 | Scheduled label smoothing: anneal the training label-smoothing from 0.05 to 0... |
| 173 | opt | discard | 1.083560 | Start LAWA at the midpoint of the warmdown phase (frac=0.71) instead of at th... |
| 172 | arch | crash | — | Long-short SWA: layers 0,1,2,6,7,8 use FA3 causal-local window=512; recurrenc... |
| 171 | curr | eval_budget_overrun | 1.079291 | Decouple stride warmdown from LR warmdown: new sampler_warmdown_frac=0.80 set... |

## Key Insights
*(Add manually — this section is preserved across regenerations.)*
