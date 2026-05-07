# Research Knowledge Base
*Auto-generated — the `## Key Insights` section at the bottom is preserved.*

## Research Tree
Full tree is at `/home/user/auto-research/magent_state_cifar/blackboard/tree.tsv` (TSV; columns: `exp_id, parent_exp, depth, path, specialist, status, train_s, delta_vs_best, hypothesis`; preorder-sorted so siblings are adjacent). Slice it from Bash, e.g.:

```
# subtree rooted at exp_042 (all descendants + the node itself)
awk -F '\t' 'NR==1 || $4 ~ /(^|\/)042(\/|$)/' /home/user/auto-research/magent_state_cifar/blackboard/tree.tsv
# direct children of exp_042
awk -F '\t' 'NR==1 || $2=="042"' /home/user/auto-research/magent_state_cifar/blackboard/tree.tsv
# only kept trials, sorted by train_s
awk -F '\t' 'NR==1 || $6=="keep"' /home/user/auto-research/magent_state_cifar/blackboard/tree.tsv | sort -t $'\t' -k7,7g
```

**Current-best lineage** (root → best):
```
exp_000 [baseline, baseline, time=26.356000] calibrated baseline (1 trial, avg=26.356000)
   └─ exp_030 [reg, keep, time=25.225900, Δ=-0.211000] train_epochs 43 + eval every 5 epochs (skip 33 of 43 intermediate val_acc calls): ~1.3s wallclock sa
      └─ exp_070 [opt, keep, time=25.146400, Δ=-0.079500] 42ep + lr=11 + warmup fraction 10%→5% (peak LR in ~2 epochs vs ~4): exp_060 (42ep+lr=11+warmup=10%)   ← BEST

**Best's direct children (23):**
  • exp_075 [loss, disqualified, time=—] 41ep + ls=0.15 on exp_070 base (42ep+lr=11+warmup=5%): sharper targets via ls 0.20→0.15 may compensa
  • exp_076 [aug, disqualified, time=—] Reduce proxy training from 42→25 epochs (proxy_train_epochs=25): generates 1225 masks cycled 1.68× o
  • exp_077 [opt, disqualified, time=—] 41ep + lr=12: exp_070 (42ep+lr=11+warmup=5%) barely passes gate; lr=12 (+9%) × 41ep (−2.4% steps) gi
  • exp_078 [arch, preflight_crash, time=—] Replace proxy model with self-paced loss caching: eliminate model_proxy compile+training entirely; m
  • exp_079 [loss, disqualified, time=—] 42ep + annealed label smoothing 0.30→0.10 (avg=0.20, same as baseline): redistributes regularization
  • exp_080 [opt, disqualified, time=—] 41ep + lr=12 + warmup 5%→2.5%: exp_077 (41ep+lr=12+warmup=5%) got acc=0.9596 (same miss as exp_060).
  • exp_081 [arch, disqualified, time=—] Replace proxy model with self-paced loss caching (fix float16→float32 cast): eliminate model_proxy c
  • exp_082 [reg, disqualified, time=—] 41 epochs + Lookahead every-3-steps (vs every-5): 67% more Lookahead updates in the warmdown phase (
  • exp_083 [arch, disqualified, time=—] Proxy compile mode max-autotune→reduce-overhead: exp_081 showed proxy costs ~3.45s total (21.7s with
  • exp_084 [aug, disqualified, time=—] 41ep + progressive cutout removal: train with cutout=12 for epochs 1-36, then cutout=0 for final 5 e
  • exp_085 [loss, disqualified, time=—] 41ep + Poly-1 loss ε=1.0 on exp_070 base (lr=11, warmup=5%): PolyLoss adds ε*(1-p_t) to CE, sharpeni
  • exp_086 [opt, disqualified, time=—] 41ep + lr=11 + cosine cooldown (5% warmup → cosine decay): exp_077 (41ep+lr=11+triangular) was disqu
  • exp_087 [reg, disqualified, time=—] 41 epochs + Lookahead every-2-steps (vs every-3 in exp_082): exp_082 gave acc=0.9593, gap=0.0007 bel
  • exp_088 [loss, disqualified, time=—] Proxy loss label_smoothing 0.20→0.0: proxy's sole job is hard-example ranking via loss ordering; lab
  • exp_089 [aug, disqualified, time=—] Proxy trains 42→35 epochs (saves ~0.575s; 1.2× mask cycling vs 1.68× for disqualified exp_076@25ep):
  • exp_090 [reg, disqualified, time=—] 41 epochs + stochastic depth drop_path_rate=0.1 on main model ConvGroup residuals: per-sample binary
  • exp_091 [arch, disqualified, time=—] Proxy depth=2→1: single Conv per ConvGroup stage in the proxy (widths {32,64,64} unchanged). Halves 
  • exp_092 [loss, disqualified, time=—] 41ep + ls=0.1 on exp_070 base: airbench94 uses ls=0.1 with this architecture; strong aug (cutout=12+
  • exp_093 [arch, disqualified, time=—] Proxy widths {32,64,64}→{24,48,48} (depth=2 unchanged): exp_091 showed depth=1 {32,64,64} saves ~0.8
  • exp_094 [opt, preflight_crash, time=—] 42ep + lr=11 + warmup=5% (exp_070 base) + Muon-style Newton-Schulz gradient orthogonalization for 2D
  • exp_095 [reg, disqualified, time=—] 41 epochs + gradient clipping max_norm=1.0: clip_grad_norm_ before optimizer.step() stabilizes lr=11
  • exp_096 [aug, disqualified, time=—] 42ep + batch Mixup (alpha=0.2) replacing cutout=12: remove spatial masking from loader (cutout→0), a
  • exp_097 [opt, disqualified, time=—] 42ep + lr=11 + warmup=5% + Muon-style Newton-Schulz gradient orthogonalization for 2D weights (fix: 
```

## Recent Activity (last 30)

| exp | specialist | status | train_s | hypothesis |
|-----|------------|--------|---------|------------|
| 097 | opt | disqualified | — | 42ep + lr=11 + warmup=5% + Muon-style Newton-Schulz gradient orthogonalizatio... |
| 096 | aug | disqualified | — | 42ep + batch Mixup (alpha=0.2) replacing cutout=12: remove spatial masking fr... |
| 095 | reg | disqualified | — | 41 epochs + gradient clipping max_norm=1.0: clip_grad_norm_ before optimizer.... |
| 094 | opt | preflight_crash | — | 42ep + lr=11 + warmup=5% (exp_070 base) + Muon-style Newton-Schulz gradient o... |
| 093 | arch | disqualified | — | Proxy widths {32,64,64}→{24,48,48} (depth=2 unchanged): exp_091 showed depth=... |
| 092 | loss | disqualified | — | 41ep + ls=0.1 on exp_070 base: airbench94 uses ls=0.1 with this architecture;... |
| 091 | arch | disqualified | — | Proxy depth=2→1: single Conv per ConvGroup stage in the proxy (widths {32,64,... |
| 090 | reg | disqualified | — | 41 epochs + stochastic depth drop_path_rate=0.1 on main model ConvGroup resid... |
| 089 | aug | disqualified | — | Proxy trains 42→35 epochs (saves ~0.575s; 1.2× mask cycling vs 1.68× for disq... |
| 088 | loss | disqualified | — | Proxy loss label_smoothing 0.20→0.0: proxy's sole job is hard-example ranking... |
| 087 | reg | disqualified | — | 41 epochs + Lookahead every-2-steps (vs every-3 in exp_082): exp_082 gave acc... |
| 086 | opt | disqualified | — | 41ep + lr=11 + cosine cooldown (5% warmup → cosine decay): exp_077 (41ep+lr=1... |
| 085 | loss | disqualified | — | 41ep + Poly-1 loss ε=1.0 on exp_070 base (lr=11, warmup=5%): PolyLoss adds ε*... |
| 084 | aug | disqualified | — | 41ep + progressive cutout removal: train with cutout=12 for epochs 1-36, then... |
| 083 | arch | disqualified | — | Proxy compile mode max-autotune→reduce-overhead: exp_081 showed proxy costs ~... |
| 082 | reg | disqualified | — | 41 epochs + Lookahead every-3-steps (vs every-5): 67% more Lookahead updates... |
| 081 | arch | disqualified | — | Replace proxy model with self-paced loss caching (fix float16→float32 cast):... |
| 080 | opt | disqualified | — | 41ep + lr=12 + warmup 5%→2.5%: exp_077 (41ep+lr=12+warmup=5%) got acc=0.9596... |
| 079 | loss | disqualified | — | 42ep + annealed label smoothing 0.30→0.10 (avg=0.20, same as baseline): redis... |
| 078 | arch | preflight_crash | — | Replace proxy model with self-paced loss caching: eliminate model_proxy compi... |
| 077 | opt | disqualified | — | 41ep + lr=12: exp_070 (42ep+lr=11+warmup=5%) barely passes gate; lr=12 (+9%)... |
| 076 | aug | disqualified | — | Reduce proxy training from 42→25 epochs (proxy_train_epochs=25): generates 12... |
| 075 | loss | disqualified | — | 41ep + ls=0.15 on exp_070 base (42ep+lr=11+warmup=5%): sharper targets via ls... |
| 074 | reg | disqualified | — | 42ep + SWA (uniform avg of epochs 37-41) + BN running-stat recompute after SW... |
| 073 | arch | disqualified | — | Proxy training shortened 43→20 epochs with mask cycling: proxy generates 960... |
| 072 | loss | disqualified | — | batch_size_masked=448 + PolyLoss ε=2.0: exp_068 (448, CE) gave acc=0.9577 (−0... |
| 071 | aug | disqualified | — | Reduce proxy training from 43→10 epochs, cycle masks over main training steps... |
| 070 | opt | keep | 25.146400 | 42ep + lr=11 + warmup fraction 10%→5% (peak LR in ~2 epochs vs ~4): exp_060 (... |
| 069 | reg | disqualified | — | 42 epochs + SWA uniform average of last 5 epoch snapshots (epochs 37-41): rep... |
| 068 | loss | disqualified | — | batch_size_masked 512→448 (−12.5% per-step main-model compute): trains on the... |

## Key Insights
*(Add manually — this section is preserved across regenerations.)*
