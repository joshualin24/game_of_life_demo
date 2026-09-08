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

## Design A — run 4 (frozen `patch_head` + near-identity adapter + disc reg)

30 epochs, λ 20/20/20 warmed 0→1 over 5 ep, cosine LR, `λ_disc`=1.
Checkpoint select = min sym/unrel ratio among F1≥0.995 epochs → **epoch 16**.
`train.py` defaults now = this config. Log: `results/run4.out`,
`results/train_log.txt`; diagnostic: `results/run4_diag.txt`.

| | raw V8 pooled `tf_L4` | v1 adapter (ep 16) |
|---|---|---|
| next-state **F1** | 0.9985 | **0.9990** (prec .9998 / rec .9981) |
| sub-patch `rel_move` = ‖Δ‖/‖ψ‖ | 0.110 | 0.051 |
| D4 `id_resid` | 0.110 | 0.047 |
| D4 learned `ρ(r)` | — | 0.042 (now beats identity; bestlin 0.035) |
| unrelated-pair dist ‖ψ(x)−ψ(x′)‖ | 3.27 | **3.56** (held / up) |
| **sym / unrelated ratio** — sub-patch | **0.377** | **0.064** |
| **sym / unrelated ratio** — D4 | **0.370** | **0.062** |
| mean ‖ψ‖ | 11.34 | 4.76 |
| mean pairwise cosine across grids | 0.955 | 0.777 |
| effective dim (participation ratio) | 2.4 | 1.8 |

### Result

**Accuracy fully maintained** (F1 0.999 = V8) **while a symmetry transform now
perturbs `ψ` ~6× less than an unrelated grid does** (sym/unrel 0.37 → 0.06).

**Not a collapse.** The discriminability regularizer held: unrelated-grid L2
distance is 3.56 (up from 3.27), and directional spread *increased* (mean
pairwise cosine 0.955 → 0.777 — grids fan out more). `‖ψ‖` shrank 11.3 → 4.8
because the large uninformative shared/DC component was removed.

### Flags / open

1. **Pooled `ψ` effective dim 2.4 → 1.8** — it's become a compact symmetry-clean
   *summary*; the rich content the decoder needs still lives in the token rep
   `z` (which the frozen `patch_head` reads). Worth probing whether 1.8-dim `ψ`
   loses anything task-relevant.
2. **Mechanism is still invariance**, not structured displacement (`dir_cons`≈0,
   learned `δ`≈0). D4's learned `ρ(r)` finally beats identity but only slightly.
3. **`z` itself not measured** — we only made the *pooled* readout symmetric.
   Does `z` (100×64) also become more symmetric? And how do V8's intermediate
   layers look through this adapter? (the "how do layers react" question)

### Next candidates

- Probe (1)/(3): measure symmetry + discriminability of `z` (not just pooled),
  and of each `tf_Lk` fed through the adapter.
- v2 = token-level ("option b"): make `z` itself equivariant — where the D4
  operator and 4-cell patch-permutation are non-trivial, not just invariance.
- Try pushing further (higher λ / more epochs) to see if sym/unrel < 0.06 is
  reachable before F1 gives.
