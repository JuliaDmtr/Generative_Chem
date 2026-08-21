---
name: molecule-diffusion
description: Build, train, and sample from an E(3)-equivariant diffusion model (EDM-style) that generates new 3D molecules — atom positions and atom types jointly. Use this whenever the user asks to build/create a diffusion model for molecule/drug generation, generative chemistry, 3D molecular structure generation, or a SOTA molecule-generation model, even if they don't name "EDM" or "equivariant" specifically. Includes the equivariant network, training loop, generation script, and a test suite (equivariance checks, diffusion-math checks, end-to-end smoke test).
---

# Equivariant Molecule Diffusion

A complete, from-scratch implementation of an **E(3)-equivariant
diffusion model** for generating new 3D molecules, following the EDM
approach (Hoogeboom et al. 2022, ICML) that later SOTA methods (GeoLDM,
MiDi) build on. Read `references/approach.md` for the method, equations,
and why equivariance matters before making changes to the model.

## What's included

```
scripts/
  egnn.py        E(3)-equivariant graph network (the denoiser backbone)
  diffusion.py   Forward noising process, training loss, reverse sampler
  dataset.py     QM9-format loader + a synthetic dataset (no download needed)
  utils.py       Atom vocab, CoM removal, noise schedule, EMA
  train.py       CLI: python train.py --synthetic --epochs 50 --out ckpt.pt
  generate.py    CLI: python generate.py --ckpt ckpt.pt --n-samples 20 --out-dir samples/
                       --condition {negative,positive,null} --guidance-scale
                       selects/extrapolates the class-conditional model's
                       output when the checkpoint was trained with cond_dim>0.
  evaluate.py    CLI: python evaluate.py --xyz-dir samples/ --out-dir renders/
                       Converts generated .xyz -> SMILES (RDKit bond
                       perception) + 2D depiction PNGs. Needs `pip install rdkit`.
  prepare_chembl.py   CLI: SMILES -> 3D .npz via RDKit (ETKDG embed + MMFF
                       relax), wired to chembl_processor.ChemblProcessor.
                       Needs `pip install rdkit`.
  chembl_processor.py User-supplied SMILES loader/tokenizer (ChEMBL cache
                       loading + char-vocab prep for a separate LSTM model).
                       prepare_chembl.py only uses its make_samples() method.
  prepare_parp_conformers.py  CLI: curated PARP-inhibitor SMILES (CSV) ->
                       parp_3d.npz, several ETKDG conformers per molecule
                       (reuses prepare_chembl.smiles_to_3d). Needs rdkit.
  train_conditional.py  CLI: trains the 3-way class-conditional model
                       (COND_POSITIVE = real PARP inhibitor, COND_NEGATIVE =
                       size-matched ChEMBL, COND_NULL = random ChEMBL) on
                       parp_3d.npz + chembl_3d.npz jointly. Supports --resume
                       (continues from a checkpoint's epoch + 1 with a fresh
                       cosine LR anneal over the new --epochs) and periodic
                       in-memory validity diagnostics (--diag-every). See
                       "Managing long training runs" below before launching
                       one in the background.
  eval_checkpoint_trend.py  CLI: builds a real epoch-vs-validity trend across
                       several distinct checkpoint files (fixed seed/sizes so
                       results are comparable), plots it alongside a training
                       loss CSV. Only useful if checkpoints were saved with
                       distinct filenames per epoch (see "Managing long
                       training runs").
chembl_3d.npz, parp_3d.npz  Prebuilt conditioning datasets (ChEMBL background
                       pool and curated PARP-inhibitor conformers) used by
                       train_conditional.py by default.
tests/
  test_egnn.py               shape + rotation/translation equivariance checks
  test_diffusion.py          noise schedule, loss finiteness, sampling checks
  test_end_to_end.py         tiny full train->generate->xyz smoke test
  test_math_numpy_sanity.py  dependency-free check of the core diffusion math
  test_conditional.py        cond_dim=0 backward-compat, conditioned output
                              shape, and equivariance-under-conditioning checks
references/
  approach.md    the method explained: equivariance, schedule, data format,
                 known simplifications and how to extend them (charges,
                 property conditioning, bond graphs, evaluation metrics)
requirements.txt
```

## Setting this up (do this first, in the user's environment)

```bash
pip install -r requirements.txt
python tests/test_math_numpy_sanity.py   # sanity check, no torch needed
pytest tests/                             # full suite, needs torch
```

If you (Claude) are running in a sandbox with no network access, you
won't be able to `pip install torch` — say so plainly rather than
pretending the full test suite ran. You *can* still run
`test_math_numpy_sanity.py` directly, since it only needs numpy.

## Quick start

```bash
# 1. Train on the built-in synthetic dataset — no data download needed,
#    just validates the whole pipeline runs (not chemically meaningful).
python scripts/train.py --synthetic --n-molecules 512 --epochs 50 --out ckpt.pt

# 2. Generate molecules from the trained checkpoint
python scripts/generate.py --ckpt ckpt.pt --n-samples 20 --out-dir samples/
# -> writes samples/mol_0000.xyz, mol_0001.xyz, ... (standard .xyz format,
#    openable in VMD/PyMOL/Avogadro, or via rdkit.Chem.rdmolfiles.MolFromXYZFile)

# 3. Convert generated 3D coordinates back to SMILES + render images
pip install rdkit
python scripts/evaluate.py --xyz-dir samples/ --out-dir renders/
# -> writes renders/smiles.txt (one "filename<TAB>SMILES" per molecule that
#    converted successfully) and renders/mol_0000.png, ... 2D depictions.
#    Reports a validity rate: the % of generated .xyz files where RDKit's
#    distance-based bond perception (DetermineBonds) + SanitizeMol succeed.
#    A meaningful fraction failing is normal for a diffusion model and
#    reflects real generation quality, not a bug — geometries need to be
#    fairly accurate for RDKit to infer a chemically valid bond graph.
```

For real results, train on real data instead of `--synthetic`:
see "Data format" in `references/approach.md` for how to point
`--data-npz` at a preprocessed QM9 (or GEOM-Drugs) file, and scale up
`--hidden-dim`/`--n-layers`/`--epochs` — the synthetic-data defaults are
sized for a fast pipeline check, not for training a good model.

## Using ChEMBL SMILES instead of QM9

If your real data starts as SMILES (e.g. from ChEMBL) rather than
ready-made 3D coordinates, `scripts/prepare_chembl.py` converts it to
the same `.npz` format `QM9Dataset` expects, using RDKit (parse ->
add explicit H's -> ETKDG conformer embedding -> MMFF relax):

```bash
pip install rdkit
python scripts/prepare_chembl.py \
    --data-path-prefix ../../Datasets/ \
    --num-samples 20000 --max-len 100 \
    --out chembl_3d.npz
python scripts/train.py --data-npz chembl_3d.npz --epochs 50 --out ckpt.pt
```

It uses `ChemblProcessor.make_samples()` for raw SMILES — not
`prepare_data_for_lstm()`, whose `$`/`E`-wrapped output is for a
separate char-LSTM's vocabulary, not for RDKit parsing. ChEMBL covers
far more elements than QM9's default H/C/N/O/F `ATOM_VOCAB`
(S, Cl, Br, P, ...); the script skips and counts out-of-vocab
molecules rather than erroring, and warns if that fraction is large —
extend `ATOM_VOCAB` in `utils.py` and retrain if you're losing too
much data that way. ETKDG embedding also fails outright for some
SMILES (large/flexible/macrocyclic molecules especially); those are
skipped and counted too.

## Conditional PARP-inhibitor training

`train_conditional.py` trains a 3-way class-conditional model
(`COND_POSITIVE`/`COND_NEGATIVE`/`COND_NULL` — see "Property
conditioning" in `references/approach.md`) on top of the same
EGNN/diffusion core:

```bash
# 1. Build the PARP-inhibitor conditioning set (several ETKDG conformers
#    per curated SMILES, since there are only a few hundred molecules)
pip install rdkit
python scripts/prepare_parp_conformers.py \
    --csv ../../Datasets/PARP_inhibitors_July26/merged_parp_inhibitors.csv \
    --n-conformers 5 --out parp_3d.npz

# 2. Train (needs chembl_3d.npz too, for the negative/null classes)
python scripts/train_conditional.py \
    --parp-npz parp_3d.npz --chembl-npz chembl_3d.npz \
    --epochs 300 --hidden-dim 192 --n-layers 6 --out checkpoints/conditional_checkpoint.pt

# 3. Sample under a specific condition
python scripts/generate.py --ckpt checkpoints/conditional_checkpoint.pt \
    --n-samples 20 --condition positive --out-dir samples/
```

## Managing long training runs

Training runs of this size take long enough to be backgrounded and
resumed across sessions — two easy-to-miss pitfalls to avoid:

- **Always launch with `python -u`** (or `PYTHONUNBUFFERED=1`) when
  redirecting stdout to a log file for a backgrounded run
  (`python -u scripts/train_conditional.py ... > train_log.txt 2>&1 &`).
  Without it, Python fully buffers stdout when piped to a non-TTY, so a
  log file can sit at 0 bytes for hours even though training is
  progressing — the loss/diagnostic prints are real, just stuck in an
  unflushed buffer that may never fill before the run ends.
- **`--ckpt-every` also writes a distinct `{out}_epoch_{N}{ext}` file**
  alongside the overwritten "latest" `--out` path (see
  `save_checkpoint(..., also_tag=True)` in `train_conditional.py`), so
  intermediate history survives instead of every periodic save
  clobbering the same file. This is what `eval_checkpoint_trend.py`
  needs to build a real epoch-vs-validity trend — without distinct
  filenames, only the final/most-recent checkpoint would ever be
  inspectable.
- **`--resume checkpoint.pt --epochs N`** continues from
  `checkpoint['epoch'] + 1` and runs `N` more epochs, with a *fresh*
  cosine LR schedule annealing `--lr -> --lr-min` over just those `N`
  epochs (the schedule is not state-restored across resumes) — pick
  `--lr`/`--lr-min` for a resume with that in mind, not as if it were a
  continuation of the original schedule.
- **`eval_checkpoint_trend.py`** evaluates a fixed list of checkpoint
  paths (edit the `CHECKPOINTS` list for a new run) with a shared fixed
  seed/sizes list, so results across checkpoints differ only because of
  the model, not because of different sampled molecule sizes. It plots
  training loss (from a CSV with `epoch,loss,loss_x,loss_h` columns)
  alongside generation validity — a real evaluation metric, not a
  held-out validation loss (there is no held-out denoising loss computed
  anywhere in this project).

## When modifying the model

- Keep coordinate updates as scalar multiples of relative vectors
  `(x_i - x_j)` and keep feature updates fed only by invariant scalars
  (distances). Any change that lets the network see raw absolute
  coordinates breaks equivariance — `tests/test_egnn.py`'s rotation test
  will catch this, so run it after any `egnn.py` change.
- Keep `remove_mean_with_mask` calls wherever `x` is produced or
  consumed (ground truth, noise, model output) — dropping one breaks the
  translation-invariance trick and the CoM tests will fail.
- If you extend the atom vocabulary or feature set (charges, aromaticity
  flags, etc.), update `utils.ATOM_VOCAB`/`NUM_ATOM_TYPES` and retrain —
  the vocabulary size is baked into the model's input dimension, so old
  checkpoints won't load against a changed vocab.

## Extending

See "Known simplifications vs. the full EDM paper" in
`references/approach.md` for concrete extension points: formal-charge
diffusion, property-conditioned generation (classifier-free guidance),
explicit bond-order diffusion (DiGress-style), and an RDKit-based
validity/novelty/uniqueness evaluation script.
