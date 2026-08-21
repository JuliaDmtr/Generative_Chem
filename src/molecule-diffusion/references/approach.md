# Approach: Equivariant Diffusion for 3D Molecule Generation

This skill implements a version of **EDM** (Hoogeboom, Satorras, Vignac &
Welling, *"Equivariant Diffusion for Molecule Generation in 3D"*, ICML
2022, arXiv:2203.17003) — the model that established joint
position+atom-type diffusion with an E(3)-equivariant denoiser as the
standard recipe for 3D molecule generation. Later SOTA work (GeoLDM,
MiDi, EQGAT-diff) extends this same core recipe (latent-space diffusion,
joint 2D bond diffusion, better schedules) rather than replacing it — so
this is a solid, well-understood foundation to build on.

## Why not just "diffuse the coordinates"?

Molecules are unordered point clouds with two symmetries that a naive
model would have to learn from data (wasting capacity and data):

- **Translation**: a molecule doesn't change if you shift all atoms by
  the same vector.
- **Rotation/reflection (O(3))**: a molecule doesn't change if you
  rotate or mirror it in space.

Ignoring these means the model has to separately learn that every
rotated copy of the same molecule is equally valid — a huge, unnecessary
burden. Instead:

1. **Translation** is handled by always working in the **zero
   center-of-mass (CoM) subspace**: every ground-truth molecule, every
   noise sample, and every intermediate diffusion state has its CoM
   subtracted (`utils.remove_mean_with_mask`). This effectively drops
   one degree of freedom per molecule and means the model never even
   sees absolute position.
2. **Rotation/reflection** is handled architecturally by the **EGNN**
   backbone (`scripts/egnn.py`): every coordinate update is a learned
   scalar multiple of a relative vector `(x_i - x_j)`, and every
   feature update only ever consumes rotation-invariant scalars
   (`||x_i - x_j||^2`). This guarantees `f(Qx) = Q f(x)` for any
   orthogonal `Q`, exactly (not just approximately, and not just on
   the training distribution) — verified in `tests/test_egnn.py`.

## The diffusion process

Standard variance-preserving (DDPM-style) forward process, applied
jointly to positions `x` and a continuous relaxation of the atom-type
one-hot vectors `h`:

```
z_t = sqrt(alpha_bar(t)) * x0 + sqrt(1 - alpha_bar(t)) * eps,   eps ~ N(0, I)
```

with `alpha_bar(t)` a **cosine schedule** (Nichol & Dhariwal 2021),
`t` sampled continuously in `[0, 1)` per molecule during training
(`diffusion.py: q_sample`). The denoiser is trained with the standard
simple noise-prediction objective:

```
L = E_t [ || eps_x - eps_x_hat(z_t, t) ||^2 + w_h * || eps_h - eps_h_hat(z_t, t) ||^2 ]
```

`eps_x` is itself projected onto the zero-CoM subspace before being used
as a training target, so the model only ever has to predict noise
components that keep the sample on that subspace.

At generation time (`diffusion.py: sample`), we start from pure noise
(zero-CoM Gaussian for `x`, standard Gaussian for `h`) and iteratively
denoise using a small number of reverse steps, predicting `x0`/`h0` at
each step and re-noising toward the next (smaller) noise level — a
DDIM-style sampler, which needs far fewer steps than ancestral DDPM
sampling while remaining a valid approximate sampler for this SDE/ODE
family. At the final step, `argmax(h0_pred)` over the atom-type
dimension recovers a discrete atom type per atom.

## Data format

`dataset.py` expects, per molecule: a `(n_atoms, 3)` array of atomic
coordinates in Angstroms and an `(n_atoms,)` array of atom-type indices
into `utils.ATOM_VOCAB` (default: `H, C, N, O, F, S, Cl, Br, I, P, B, Si` —
QM9's element set plus the common heteroatoms in drug-like molecules).

To use real QM9 data:
1. Get QM9 via any local mirror (e.g. `torch_geometric.datasets.QM9`, or
   the raw `.xyz` dump from https://doi.org/10.6084/m9.figshare.978904).
2. Write a one-off script that, for each molecule, extracts positions
   and atom symbols, maps symbols through `ATOM2IDX`, and appends to two
   object arrays; save with
   `np.savez("qm9.npz", positions=positions_arr, atom_types=types_arr)`.
3. Pass `--data-npz qm9.npz` to `train.py`.

For larger/more diverse chemistry (e.g. drug-like molecules), swap in
GEOM-Drugs with the same format and extend `ATOM_VOCAB` in `utils.py`
(then retrain from scratch — the vocabulary is baked into the model's
input dimension).

In this project, real training data is ChEMBL + a curated PARP-inhibitor
set rather than QM9/GEOM-Drugs: `scripts/prepare_chembl.py` embeds a
single ETKDG conformer per ChEMBL SMILES (`chembl_3d.npz`, used as the
`negative`/`null` condition pool), and `scripts/prepare_parp_conformers.py`
embeds several ETKDG conformers per curated PARP-inhibitor SMILES
(`parp_3d.npz`, used as the `positive` condition pool) for the
conditional model trained by `scripts/train_conditional.py`.

## Known simplifications vs. the full EDM paper (extension points)

- **No explicit formal charge channel.** The paper diffuses charges
  alongside atom type; we only diffuse atom type. Straightforward to
  add: extend `h` with a charge channel and update `NUM_ATOM_TYPES`
  usage accordingly.
- **Property conditioning: implemented, but as explicit 3-way class
  conditioning rather than the paper's classifier-free-guidance
  dropout.** `EGNN`/`EquivariantMoleculeDiffusion` accept an optional
  `cond_dim`/`cond` (default 0/None, fully backward compatible),
  concatenated into `EGNN.forward`'s node input exactly like the time
  embedding. `scripts/train_conditional.py` trains a PARP-inhibitor
  conditional model with three condition classes (negative/positive/null
  — see `utils.COND_*`) instead of single-vector CFG dropout, and
  `generate.py --condition {negative,positive,null} --guidance-scale`
  samples from it (with optional CFG-style extrapolation against the
  null class). See `scripts/prepare_parp_conformers.py` for building the
  conditioning dataset.
- **DDIM-style sampler, not the paper's exact ancestral sampler.** Faster
  and works well in practice for this SDE family; swap in full ancestral
  DDPM sampling in `diffusion.py: sample` if you need to match the paper
  exactly.
- **No explicit bond-order graph.** Bonds aren't modeled directly; this
  follows the original EDM (bonds are inferred post-hoc, e.g. from
  interatomic distances via `rdkit.Chem.rdDetermineBonds`, or with
  OpenBabel). Graph-diffusion models like **DiGress** diffuse bond
  orders directly if that's what you need instead of/alongside 3D
  coordinates.
- **Validity evaluation: implemented; novelty/uniqueness are not.**
  `scripts/evaluate.py` runs `write_xyz` output through RDKit's
  `DetermineBonds` + `SanitizeMol` and reports the % that parse as valid
  molecules (`xyz_to_mol`/`convert_dir`), plus an in-memory variant
  (`mol_from_coords`, no file I/O) used for fast periodic validity
  diagnostics during training (see `train_conditional.py`). Novelty
  (not in the training set) and uniqueness (not duplicated among
  generated samples) — both reported in the original paper — still
  aren't implemented; add a canonical-SMILES set comparison against the
  training data if you need them.

## Managing long training runs

`train_conditional.py` runs are long enough to be interrupted and
resumed across sessions, which surfaced two real bugs worth documenting
so they aren't reintroduced:

- **Checkpoint history**: `save_checkpoint()` always overwrites `--out`
  (so `--resume`/`generate.py` always have a "latest" pointer), but when
  called with `also_tag=True` (every `--ckpt-every` epochs, and at the
  final epoch) it *also* writes a distinct, never-overwritten
  `{out}_epoch_{N}{ext}` file. Without this, every periodic save
  clobbers the same path and no intermediate history survives — which
  is exactly what happened to this project's first ~150 training
  epochs before the fix, leaving only whatever checkpoint happened to be
  on disk at inspection time. `scripts/eval_checkpoint_trend.py` depends
  on these distinct per-epoch files to build a real epoch-vs-validity
  trend (fixed seed + fixed sampled sizes across all checkpoints, so
  differences reflect the model, not the sample).
- **Resume semantics**: `--resume ckpt.pt --epochs N` continues from
  `ckpt['epoch'] + 1` for `N` more epochs, but the cosine LR schedule is
  *not* state-restored — each invocation gets its own fresh anneal from
  `--lr` to `--lr-min` over just its own `--epochs`, independent of how
  far the original schedule had progressed.
- **Unbuffered logging**: always launch backgrounded runs with
  `python -u` (or `PYTHONUNBUFFERED=1`) when redirecting stdout to a log
  file. Piping to a file makes Python fully buffer stdout (rather than
  line-buffer, as it does for an interactive TTY), so without `-u` a log
  file can stay empty for hours of real progress — the internal buffer
  simply hasn't filled yet, not because nothing happened.

## Compute expectations

This is a real (if compact) equivariant diffusion model — training to
QM9-quality results in the original paper takes on the order of a few
GPU-days. The synthetic dataset + small hidden dims used in the test
suite and default `train.py` args are for pipeline validation, not for
producing chemically meaningful molecules; scale `--hidden-dim`,
`--n-layers`, `--epochs`, and use real data for that.
