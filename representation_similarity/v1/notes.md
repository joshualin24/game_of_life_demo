# v1 — hybrid symmetry adapter on frozen V8 (pooled readout)

First attempt at: *use frozen V8's final representation to build a map that
produces embeddings which better respect GoL's symmetries (toroidal
translation + D4) while keeping t→t+1 accuracy.*

## Design (v1)

- **Frozen** V8 `CNNTransformerV4`; target = `tf_L4` (100×64 tokens, pre-output).
- **Adapter** `z = h(tf_L4)` — per-token residual MLP (`adapter.py::Adapter`).
- **Pooled readout** `ψ(x) = mean_tok(z)` — one 64-vec per grid. This is what the
  symmetry losses act on ("option a").
- **Transformation model** (`adapter.py::SymParams`): translations as a periodic
  content-independent displacement `δ(dr,dc)`; D4 as 7 learned linear operators
  `ρ(r)` (`ρ(e)=I`). Hybrid.
- **Decoder** `d(z) → next-state logits` — v1 runs used a **fresh** head (mirror
  of `patch_head`). Accuracy metric = F1 of `d∘h` vs true `F(x)`.
- **Losses**: `BCE(d(z),F(x))` + `λ·(` translation-displacement consistency +
  D4 linear-equivariance + mixed-element + soft group law `)` + pred-aug on
  transformed views.

Files: `adapter.py` (model), `train.py`, `diag.py` (post-hoc full diagnostic).
Run:  `python train.py --epochs N ...`  → `results/train_log.txt`,
`results/metrics.json`, `checkpoints/best.pt`.

## Runs

| run | change | outcome |
|-----|--------|---------|
| run1 | λ=1, uniform translation-offset sampling | symmetry terms ~1% of loss; sub-patch offsets sampled ~7% of the time → no sub-patch effect. Killed ep 1. |
| run2 | λ=25/15/25, 65% of trans samples drawn from the 16 sub-patch phases | `ρ(r)` frozen at identity (no gradient at `ρ=I`); eval blind to `rel_move`. Killed ep 2. |
| run3 | separate 15× LR for `ρ`, λ=40/50/40, eval logs `rel_move` | **completed to ep 12 then degraded** (no LR decay / no λ warmup). Best = **ep 7**. |

## v1 result (run3, best = epoch 7)

`diag.py` on `checkpoints/run3_best.pt`, and a discriminability check:

| property | raw V8 pooled `tf_L4` | + adapter (ep 7) |
|---|---|---|
| sub-patch translation `rel_move` = ‖Δ‖/‖ψ‖ | 0.110 | 0.031 |
| D4 `id_resid` | 0.110 | 0.030 |
| D4 with learned `ρ(r)` | — | 0.033 (*worse* than identity) |
| D4 best-fit linear | ~0.120 | 0.022 |
| mean ‖ψ‖ | 11.34 | 3.19 (shrank 3.6×) |
| unrelated-pair dist ‖ψ(x)−ψ(x′)‖ | 3.27 | 0.78 |
| **sym / unrelated ratio** (sub-patch) | **0.377** | **0.122** |
| mean pairwise cosine across grids | 0.955 | 0.959 |
| next-state **F1** | 0.998 | **0.964** |

### Reading

1. The adapter makes `ψ` respect both symmetries by pushing it toward
   **invariance**, not toward a structured (nonzero, consistent) displacement —
   `dir_cons≈0`, learned `δ≈0`.
2. **Learned-`ρ` half is moot for a pooled readout.** Once `ψ` is near-D4-
   invariant, `ρ=I` is optimal; the trained operator only adds noise. A global
   mean vector has no orientation-dependent content to carry equivariantly. The
   hybrid only has teeth at the token level (a future "option b").
3. **Honest symmetry metric = sym-move / unrelated-move**: 0.377 → 0.122, a **3×**
   improvement. Collapse-robust. Part of the raw `rel_move` 3.6× was `ψ` shrinking.
4. **Not a collapse**: unrelated grids stay 0.78 apart (24% of ‖ψ‖) and the
   directional spread (mean pairwise cosine) is unchanged from raw V8 — two
   unrelated distributions do *not* get identical embeddings. But there is a
   global ~3.6× contraction of `ψ` to watch.
5. **Accuracy cost ~3.4 F1** (0.998 → 0.964, mostly recall) — with a fresh
   decoder that also confounds "undertrained head" with "symmetry cost".

## Next (planned): "Design A" clean run

- **Decoder = V8's frozen `patch_head`**; adapter = pure residual with zero-init
  blocks → `z = tf_L4` at init → F1 starts at 0.998. Read the F1↔symmetry frontier.
- Add a **discriminability regularizer**: keep `E‖ψ(x)−ψ(x′)‖` near its raw-V8
  value so "invariance" can't be earned by shrinking everything.
- Metrics: **sym/unrelated ratio** primary, plus F1 and mean ‖ψ‖ every epoch.
- λ warmup (0→target over ~5 ep) + cosine LR decay. Checkpoint select hard-
  prioritizes F1.
