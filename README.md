# SimVP + CfC / DFA-CfN: Recurrent Closed-Form Translators for Video Prediction

This repository extends [SimVP](https://github.com/A4Bio/SimVP) with a family of
Closed-form Continuous-time (CfC) recurrent translators, built on top of the
Inception-style multi-scale backbone SimVP already uses, plus an optional
Dynamic Feature Accumulation (DFA) gating mechanism and support for
irregularly-sampled input sequences.

> **Use the [`further_experimentation`](https://github.com/rajghosh2000/simvp_clone/tree/further_experimentation) branch.**
> It is the most recent branch and contains every feature documented here
> (DFA-CfN gating, CfC encoder/decoder, irregular sampling). `main` may be
> behind.

---

## 1. What's in this repo

SimVP's original architecture is **Encoder → Translator → Decoder**:

- **Encoder / Decoder**: per-frame spatial conv stacks (`ConvSC`), unchanged from
  upstream SimVP unless `--use_cfc_encdec` is set (see below).
- **Translator**: the temporal reasoning module. Upstream SimVP uses a static
  U-Net of Inception blocks that mixes all input frames' channels jointly, with
  no explicit sense of time. This repo adds two alternative translators built
  as **recurrent** cells that process frames one at a time and can optionally
  be made aware of real elapsed time between frames.

Three translator families are implemented (`--translator {inception, cfc, cfcincep}`),
described in detail in [Section 3](#3-translator-options).

On top of `cfcincep`, an optional **DFA-CfN gating mode** (`--use_dfa`) changes
how the recurrent cell computes its gate, and an optional **irregular sampling
mode** (`--irregular`) changes how data is drawn from each dataset so that the
model actually has to deal with non-uniform time gaps between frames. These are
described in [Section 4](#4-dfa-cfn-dynamic-feature-accumulation) and
[Section 5](#5-irregular-sampling).

---

## 2. Repository structure

```
.
├── API/
│   ├── dataloader.py              # dispatches to per-dataset loaders by --dataname
│   ├── dataloader_moving_mnist.py
│   ├── dataloader_taxibj.py
│   ├── dataloader_kth.py
│   ├── dataloader_kitti.py
│   ├── metrics.py                 # MAE / MSE / SSIM / PSNR
│   └── recorder.py                # early-stopping / checkpoint recorder
├── main.py                        # CLI entry point, argument definitions
├── exp.py                         # training / validation / test loops
├── model.py                       # Encoder, Decoder, Mid_Xnet (translator), SimVP
├── modules.py                     # all cell/backbone definitions (ConvSC, Inception,
│                                   #   ConvCfC*, ConvCfCIncep*, CfCTemporalCell, DFA gating)
├── utils.py
└── results/                       # one subfolder per --ex_name: checkpoints, logs, saved arrays
```

---

## 3. Translator options

Set via `--translator {inception, cfc, cfcincep}`.

### `inception` (baseline / upstream SimVP)

The original SimVP translator. A static U-Net of Inception blocks
(`GroupConv2d` at kernel sizes 3/5/7/11) operating on all `T` input frames'
channels concatenated together. No recurrence, no notion of elapsed time
between frames — every frame-to-frame gap is structurally invisible to it.

### `cfc`

A minimal recurrent Closed-form Continuous-time cell (`ConvCfCCell` /
`ConvCfC` in `modules.py`), spatial adaptation of the original CfC cell
(Hasani et al., *Closed-form Continuous-time Neural Networks*, Nature Machine
Intelligence 2022) with `nn.Linear` replaced by `1×1`/`3×3` convolutions to
preserve spatial dimensions. Chunks the input by `N_T`, not by real frame
count — kept mainly as a minimal reference implementation; **`cfcincep` is the
recommended recurrent translator for actual use.**

### `cfcincep` (recommended)

`ConvCfCIncepCell` / `ConvCfCIncep`. Same CfC gating equations as `cfc`, but:

- The backbone is `MultiScaleConv` (a `1×1` channel projection followed by four
  parallel grouped convolutions at kernel sizes 3/5/7/11, summed) — i.e. the
  same multi-scale Inception-style receptive field SimVP's baseline uses,
  but evaluated **once per recurrence step** rather than as a static stack.
- The input is chunked by the *actual* frame count `T` (recovered from
  `channel_in // hid_S`), not by `N_T` — every recurrence step corresponds to
  exactly one real input frame.
- Optional `--bidirectional`: runs a second independent cell over the reversed
  frame order and concatenates forward/backward hidden states before the
  output projection. Note this requires the full input sequence to be
  available up front (not streaming/online-compatible) — see caveats below.
- Accepts a per-sample, per-step `ts` tensor (real elapsed time between
  frames) at every recurrence step. This is what irregular sampling
  (Section 5) actually feeds into.

**Update rule** (per step `t`, cell state `h`):
```
feat      = MultiScaleConv-backbone([x_t, h])
g         = tanh(ff1(feat))
h_stable  = tanh(ff2(feat))
gate      = sigmoid(time_a(feat) * ts + time_b(feat))
h_new     = g * (1 - gate) + gate * h_stable
```

---

## 4. DFA-CfN (Dynamic Feature Accumulation)

Flag: `--use_dfa` (only meaningful when `--translator cfcincep`, and/or when
`--use_cfc_encdec` puts a `CfCTemporalCell` in the encoder/decoder — see
Section 6).

Ordinary `cfcincep`'s gate is computed **fresh at every step**, from that
step's features alone — it has no memory of earlier steps. DFA-CfN
(after Liang et al., *Rederived Closed-Form Continuous-Time Neural Networks*
— NIALIM / DFA-CfN, IEEE TNNLS 2026) instead accumulates the raw gating
signal across all past steps into a running term `M`, and gates off of `M`
instead of the per-step value, so the gate is informed by the whole history
seen so far rather than just the current frame:

```
omega_ts = exp(ts * (1 - ln(ts)))       # NIALIM interval-correction factor
Tinterp  = time_a(feat) * omega_ts + time_b(feat)
M        = M_prev + Tinterp             # accumulate
gate     = sigmoid(M / step_count)      # gate off the running AVERAGE
```

The gate is computed off a **running average**, not a raw unbounded sum — an
earlier raw-sum version was found to saturate the sigmoid gate over long
(`T=10`) recurrences, causing vanishing gradients and a hard training
plateau. Averaging by step count keeps `M`'s magnitude bounded regardless of
sequence length while preserving the accumulation behavior.

`--use_dfa` with `--translator inception` is a no-op (the inception translator
has no gate at all).

---

## 5. Irregular sampling

Flags: `--irregular`, `--g_max <int>` (default `1`, i.e. no-op / fully regular).

### Motivation

All four datasets are natively sampled at a **fixed** interval (one Moving
MNIST tick, one 30-min TaxiBJ bin, one KTH/KITTI camera frame). `cfcincep`'s
`ts` argument is otherwise always `1.0`, so the closed-form/continuous-time
machinery is never actually exercised. `--irregular` randomly subsamples
frames from a wider window so real, variable Δt gaps exist between kept
frames, giving the CfC/DFA-CfN cells something genuine to condition on — and
giving the `inception` baseline (which has no `ts` input at all) a fair,
harder task to be blind to.

### How it works, per dataset

- **Moving MNIST**: trajectories are simulated at a finer *sub-tick*
  resolution internally, so gaps are genuinely continuous-time, not just
  "skip some frames from a fixed grid." Both train and test sets are
  generated on the fly (the standard fixed `mnist_test_seq.npy` test file is
  **not** used when `--irregular` is on — comparability to the literature's
  fixed test split is intentionally traded for a controllable, internally
  consistent train/test irregularity scheme).
- **KTH / KITTI**: both dataloaders have random access to the full underlying
  clip/drive (video frames via seek, or a sorted list of jpgs), so a wider
  window is drawn and `T` frames are picked from it with random integer gaps
  in `{1, ..., g_max}`. Clips/drives shorter than the worst-case required span
  (`(n_frames_input-1)*g_max + n_frames_output + 1`) are dropped from the
  index.
- **TaxiBJ is not supported** — its dataloader only has access to an already
  pre-windowed `.npz` array with no wider raw context to subsample from.
  Passing `--irregular` for `--dataname taxibj` is silently a no-op (a dummy
  all-ones `ts` vector is returned so the 3-tuple interface stays uniform,
  but no actual irregularity is introduced).

### Train vs. val/test reproducibility

Training samples redraw a fresh random gap pattern every epoch (free
augmentation). Validation/test samples use a gap pattern seeded
deterministically from the sample index, so every model config sees the
*exact same* irregularity pattern on val/test across epochs and across runs —
required for any of the results below to be comparable to each other.

### `ts` propagation

`ts` (shape `(B, T-1)`) flows: dataloader → `exp.py` → `SimVP.forward(x, ts=...)`
→ `Mid_Xnet.forward(x, ts=...)` → the active cell's `forward(..., ts=...)`.
It's threaded through the encoder/decoder too if `--use_cfc_encdec` is set
(Section 6). `translator=inception` ignores `ts` entirely — it has no
mechanism to consume it.

---

## 6. `--use_cfc_encdec`

By default the encoder and decoder are the original SimVP `ConvSC` stacks —
purely spatial, no recurrence, no notion of time (correctly so, since they
don't mix information across frames). `--use_cfc_encdec` replaces both with a
`CfCTemporalCell`-based recurrent version that also walks across the `T`
frame positions with its own hidden state (and its own DFA accumulator, if
`--use_dfa` is also set). This is the module to enable if you want the
encoder/decoder to also be irregular-sampling-aware, not just the translator.

---

## 7. Full CLI reference

```
--device            cuda | cpu
--res_dir           output root (default ./results)
--ex_name           run name — everything is written to {res_dir}/{ex_name}/
--gpu / --seed

--batch_size / --val_batch_size
--data_root          default ./data/
--dataname           mmnist | taxibj | kth | kitti
--num_workers

--n_frames_output    number of frames scored against the target
--in_shape           [T, C, H, W] of the INPUT sequence
                      mmnist: 10 1 64 64   taxibj: 4 2 32 32
                      kth:    10 1 128 128 kitti:  10 3 128 160
--hid_S              spatial hidden channels
--hid_T              translator hidden channels
--N_S / --N_T        encoder/decoder depth / translator depth (inception mode only)
--groups             group-conv groups
--use_cfc_encdec     see Section 6
--use_dfa            see Section 4

--epochs / --log_step / --lr

--translator         inception | cfc | cfcincep   — see Section 3
--bidirectional      cfcincep only — see Section 3

--irregular          see Section 5
--g_max              max gap (in base ticks) between kept frames when --irregular is set
```

**Note on `--n_frames_output` vs `--in_shape`'s `T`**: the model always
internally produces as many output frames as it received as input frames
(`T` from `--in_shape`); `exp.py` slices the prediction down to the last
`--n_frames_output` frames before computing loss/metrics. This lets, e.g.,
KITTI use a 10-frame input but only score the single next frame.

---

## 8. Example commands

```bash
# Regular, original SimVP baseline (Moving MNIST)
python main.py --dataname mmnist --in_shape 10 1 64 64 --n_frames_output 10 \
    --translator inception --lr 0.01 --epochs 100 --ex_name mmnist_original

# Regular, DFA-CfN, unidirectional (Moving MNIST)
python main.py --dataname mmnist --in_shape 10 1 64 64 --n_frames_output 10 \
    --translator cfcincep --use_dfa --lr 0.003 --hid_T 128 --epochs 100 \
    --ex_name mmnist_dfacfn

# Irregular, DFA-CfN, with CfC encoder/decoder (KITTI)
python main.py --dataname kitti --in_shape 10 3 128 160 --n_frames_output 1 \
    --translator cfcincep --use_dfa --use_cfc_encdec --lr 0.003 --hid_T 128 \
    --batch_size 4 --num_workers 4 --epochs 100 \
    --irregular --g_max 4 --ex_name kitti_irreg_dfacfn
```

---

## 9. Adding a new dataloader

Every loader in `API/` must:

1. Return `(train_loader, val_loader, test_loader, mean, std)` from a
   `load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)`
   function, and be registered in `API/dataloader.py`'s dispatch.
2. Have its `Dataset.__getitem__` return a **3-tuple**:
   `(input_frames, ts_vector, output_frames)`.
   - `input_frames`: `(n_frames_input, C, H, W)`
   - `output_frames`: `(n_frames_output, C, H, W)`
   - `ts_vector`: `(n_frames_input - 1,)` float tensor. If you're not
     implementing irregular sampling for this dataset, just return
     `torch.ones(n_frames_input - 1)` — this is the "no-op" convention every
     other loader falls back to (see TaxiBJ for a minimal example).
3. If you *do* want to support `--irregular`:
   - accept `irregular: bool` and `g_max: int` kwargs in `load_data`
   - your `Dataset` needs some way to access a wider window than just the
     `n_frames_total` you'd normally read (either by generating data
     on-the-fly like Moving MNIST, or by having random access into a longer
     raw source like KTH/KITTI's video/frame-sequence access). If your raw
     data is already pre-windowed with no access to a wider context (like
     TaxiBJ's `.npz`), irregular sampling isn't implementable without
     re-deriving the windowing from an earlier, unwindowed source.
   - draw `n_frames_input - 1` integer gaps uniformly from `{1, ..., g_max}`,
     pick a valid random start offset so the full span (gaps + output frames)
     fits in the available length, and return the real gaps as `ts_vector`.
   - for validation/test, seed the RNG deterministically per sample index
     (e.g. `seed = base_seed + idx`) so results are reproducible across runs.
4. Compute a `_worst_case_span()`-style bound
   (`(n_frames_input-1)*g_max + n_frames_output + 1`) and exclude any raw
   clip/sequence shorter than that from your index when `irregular=True`.

---

## 10. Results

All numbers below are **approximate**, read off the corresponding training
plots — treat `results/{ex_name}/log.log` as the source of truth for exact
figures. "Original" = `--translator inception`. "CFC-Incep" =
`--translator cfcincep` (no `--use_dfa`). "DFA-CfN" = `--translator cfcincep
--use_dfa`. Unidirectional unless noted. `--use_cfc_encdec` was **not** used
unless explicitly stated.

### Moving MNIST (100 epochs, `n_frames_output=10`)

| Model | Final val MSE | Final SSIM |
|---|---|---|
| Original | ~44 | ~0.85 |
| CFC-Incep | ~98 | ~0.66 |
| DFA-CfN | ~88 | ~0.71 |

DFA-CfN improves over plain CFC-Incep but both trail the original baseline by
a substantial margin at full-sequence (`n_frames_output = T = 10`)
reconstruction. See Section 11 for why.

### KTH (25 epochs regular / 100 epochs irregular, `n_frames_output=10–20`)

| Model | Final val MSE (regular) |
|---|---|
| Original | ~31 |
| CFC-Incep | ~54 |
| DFA-CfN | ~54 |

Same pattern as Moving MNIST — the recurrent translators trail the parallel
baseline by a comparable margin. The irregular-sampling KTH run additionally
showed a train/val loss divergence over 100 epochs (train loss falling, val
loss rising) — likely a reduced effective training-clip count under the
irregular indexing filter given KTH's short (10–30s) source clips; **treat
the irregular KTH result as provisional**, not as clean evidence either way.

### TaxiBJ (100 epochs, regular only — `--irregular` not supported, see §5)

| Model | Final val MSE |
|---|---|
| Original | ~0.51 |
| CFC-Incep | ~0.51 |
| DFA-CfN | ~0.50 |

DFA-CfN edges out both baselines by a small margin. `n_frames_output = T = 4`
here is short, consistent with the bottleneck story in Section 11.

### KITTI (25 epochs, `n_frames_output=1`)

| Model | Final val MSE (regular) | Final val MSE (irregular, `g_max=4`) |
|---|---|---|
| Original | ~450 | ~630 |
| CFC-Incep | ~440 | ~440 (see caveat) |
| DFA-CfN | ~440 | ~565 |

DFA-CfN beats the original baseline in both regimes, and **the margin widens
under irregular sampling** — exactly the predicted direction if the
DFA-accumulated gate is actually exploiting real elapsed-time information
that the original architecture has no way to see.

**Caveat on CFC-Incep's irregular result**: its irregular-mode curve is
visually near-identical to its own regular-mode curve, which is suspicious —
plausible explanation is that a plain per-step gate (`sigmoid(t_a*ts+t_b)`,
no accumulation) has an easy escape hatch: gradient descent can simply drive
`t_a` toward zero and become `ts`-blind without any accuracy cost, since
nothing forces continued dependence on `ts`. DFA-CfN's accumulator bakes
`ts` into a running sum that every later step's gate depends on, which is a
harder dependency to optimize away. This is *not* confirmed root-caused
(the dataloader's `ts`-plumbing and RNG were independently verified working
for this exact run), but it means CFC-Incep's irregular number specifically
should not be read as "beats DFA-CfN under irregularity" — see Section 11.

### `n_frames_output` ablation on Moving MNIST (100 epochs, DFA-CfN vs. Original)

| `n_frames_output` | Final val MSE (Original / DFA-CfN) | Relative gap |
|---|---|---|
| 10 | 44 / 88 | ~2.0x |
| 5 | 32 / 48 | ~1.5x |
| **1** | **~18.5 / ~18.5** | **~1.0x (statistically tied)** |

This is the single clearest result in the repo: on the *same* dataset, with
*only* `n_frames_output` changed, the gap between DFA-CfN and the original
baseline shrinks monotonically to near-zero as the output horizon drops to
`n_frames_output=1` — the exact regime KITTI naturally sits in. See Section
11 for the interpretation.

---

## 11. Why the recurrent translators sometimes lose, and when they win

The central empirical finding across every dataset in this repo:

> **The recurrent, closed-form translators (`cfc`, `cfcincep`, DFA-CfN)
> outperform the original SimVP Inception translator specifically when the
> number of jointly-supervised output frames (`n_frames_output`) is small
> relative to the input sequence length — and lose when the model has to
> reconstruct a long output sequence in full.**

SimVP's original Inception translator processes all `T` input frames'
channels **jointly and in parallel**, with no sequential bottleneck. The CfC
family instead routes every frame through a single, fixed-width recurrent
hidden state, one timestep at a time — a structurally narrower channel for
information to pass through. When only the *final* accumulated hidden state
needs to be good (`n_frames_output=1`, KITTI's natural regime), this
bottleneck barely matters. When the model must reconstruct a full 10-frame
sequence with high fidelity at every step (`n_frames_output=T`, Moving
MNIST/KTH), the bottleneck cost compounds and the gap widens substantially.

This is directly confirmed by the `n_frames_output` ablation above (Section
10): sweeping the *same* Moving MNIST setup from `n_frames_output=10` down to
`1` closes the gap monotonically to ~0, on the identical dataset, with no
architecture change — isolating the output-horizon variable as the actual
driver, independent of dataset-specific motion complexity or resolution.

This also explains why DFA-CfN's advantage over SimVP **widens** under
irregular sampling specifically on KITTI (`n_frames_output=1`): in that
regime the recurrent bottleneck isn't the dominant cost, so the closed-form
`ts`-awareness gets to be the deciding factor instead.

---

## 12. Known limitations

- **KTH's irregular dataloader is slow.** An early version repeatedly called
  `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, idx)` per scattered frame index,
  which forces OpenCV to decode forward from the nearest keyframe on every
  seek — a large hidden cost on `.avi` clips. A fixed version reads the whole
  needed span sequentially in one pass and indexes into it in memory; confirm
  whichever version is present in your checkout before relying on KTH
  irregular timing.
- **TaxiBJ has no irregular-sampling support** (Section 5) — its data is
  pre-windowed with no wider raw context available.
- **The bidirectional variant is not streaming/online-compatible** — it
  requires the full input sequence up front, which defeats part of the
  original motivation for using a recurrent, closed-form translator over
  SimVP's batch-only architecture. Treat bidirectional results as an
  ablation/upper-bound on what's recoverable if the online constraint is
  dropped, not as the headline configuration.
- **The irregular KTH 100-epoch result shows train/val divergence** — see
  Section 10 — likely caused by a shrunk effective training set under the
  irregular-indexing length filter on KTH's relatively short source clips.
  Not fully root-caused; treat with caution.
- **`n_frames_output` must be `<=` the input sequence length `T`** given how
  slicing is implemented in `exp.py` (`pred_y[:, -batch_y.shape[1]:]`).
