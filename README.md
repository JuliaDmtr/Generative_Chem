# PARP Conditional Molecular Generation

A research prototype for generating PARP-inhibitor-like molecules in 3D: an
**E(3)-equivariant diffusion model** (EDM-style, Hoogeboom et al. 2022) that
jointly generates atom positions and atom types, class-conditioned on whether a
molecule should resemble a known PARP inhibitor or a generic ChEMBL molecule.

The model is trained on **heavy atoms only** — no explicit hydrogens — which is
the central design decision and is explained below.

## Results

Latest run (`v11`, 499 epochs). 256 samples at 500 denoising steps, compared
against 256 real PARP inhibitors pushed through the identical reconstruction
pipeline, so measurement error affects both columns equally:

| | generated | real PARP |
|---|---|---|
| usable single molecules | **68.0%** | 99.2% |
| aromatic rings / molecule | 1.60 | 2.77 |
| aliphatic rings / molecule | 2.36 | 1.46 |
| 3-membered rings | 7.4% | 0.7% |
| 6-membered rings | 62.1% | 81.1% |
| stereocentres | 1.58 | 0.51 |
| fraction sp3 C | 0.36 | 0.32 |
| molecular weight | 374 | 401 |
| logP | 2.54 | 2.82 |
| rotatable bonds | 4.20 | 4.24 |

Size, lipophilicity, heteroatom content and flexibility are indistinguishable
from real inhibitors. What remains clearly wrong is **aromaticity** and
**stereochemistry**: the model still prefers saturated rings to flat aromatic
ones, and generates roughly three times too many stereocentres.

Training improved monotonically and had not plateaued when the run ended:

| | v8 (ep 99) | v9 (ep 199) | v10 (ep 299) | v11 (ep 499) |
|---|---|---|---|---|
| loss_x | 1.3783 | 1.3516 | 1.3032 | **1.2608** |

### Generated candidates

**[parp_gallery/index.html](src/molecule-diffusion/parp_gallery/index.html)** —
the 24 candidates most similar to real inhibitors, each drawn beside the
inhibitor it resembles.

From 1024 samples, 388 molecules pass the chemistry filters; all 388 are novel
against the training set. Similarity to the nearest of 6,563 real PARP
inhibitors (Tanimoto, Morgan radius 2): median 0.27, maximum 0.83. The best
candidate is a benzimidazole-4-carboxamide — veliparib's scaffold — differing
from a known inhibitor by a single atom.

That last point cuts both ways and is worth stating plainly: **the most
inhibitor-like outputs are near-analogues, not novel chemical matter.** 92% of
candidates fall below 0.40 similarity. Read the gallery as evidence that the
model learned real PARP chemistry, not as a list of drug leads. Nothing in this
repository predicts that any molecule inhibits PARP.

## How it works

1. RDKit conformer generation builds 3D training data from SMILES: 11,552 PARP
   conformers (5,776 distinct inhibitors, median 30 heavy atoms) and 19,882
   ChEMBL background molecules.
2. An E(3)-equivariant graph network (EGNN) denoises atom positions and types
   jointly, so the model never has to learn that a rotated copy of a molecule is
   the same molecule.
3. Training is noise prediction with 3-way class conditioning (PARP / generic
   ChEMBL / null), so sampling can be steered toward the PARP class.
4. Generated coordinates are turned back into molecules: the bond graph is
   perceived from interatomic distances, then bond orders are assigned from
   geometry.

### Why no hydrogens

Dropping explicit hydrogens cuts the median molecule from 42 atoms to 30, and
cost grows with N². That made training feasible on a laptop GPU (MPS, ~345 s per
epoch) — but it breaks the usual evaluation path, because RDKit's
`DetermineBonds` infers bond orders by balancing valences against explicit
hydrogens and reports ~0% validity without them.

So the measurement stack was rebuilt and calibrated against real molecules:

- **Bond graph** (`heavy_atom_eval.py`) — perceived from distances. On real
  stripped conformers it recovers the exact bond set for 100/100 molecules.
- **Bond orders** (`bond_orders.py`) — geometry says how many π bonds each atom
  needs; placing the double bonds is a maximum-weight matching. On 500 real
  conformers: 100% produce a valid molecule, **85.2% reproduce the original
  molecule exactly**. So roughly one structure in seven carries a wrong bond
  order or tautomer, and that bounds how far any drawn structure can be trusted.

## Repository structure

- [src/molecule-diffusion/CASE_STUDY.md](src/molecule-diffusion/CASE_STUDY.md) —
  debugging case study: why generated molecules had unrealistic 3-membered
  rings, the fixes that worked, and one that didn't.
- [src/molecule-diffusion/references/approach.md](src/molecule-diffusion/references/approach.md) —
  the method, and a line-by-line comparison against the reference EDM
  implementation with every difference marked intentional or open.
- [src/molecule-diffusion/SKILL.md](src/molecule-diffusion/SKILL.md) — the
  skill definition the project was scaffolded from, kept for provenance. It
  describes the original with-hydrogen prototype and is deliberately not kept
  in sync; this README is the current walkthrough.
- [CLAUDE.md](CLAUDE.md) — the rules this project works under, written in
  response to the bugs in the case study.

`src/molecule-diffusion/scripts/`

| | |
|---|---|
| `egnn.py`, `diffusion.py`, `utils.py` | denoiser, forward/reverse process, shared constants |
| `dataset.py`, `prepare_chembl.py`, `prepare_parp_conformers.py`, `strip_hydrogens.py` | data preparation, SMILES → 3D `.npz` |
| `train_conditional.py` | conditional training (`train.py` is a pipeline smoke test) |
| `heavy_atom_eval.py`, `bond_orders.py` | heavy-atom reconstruction |
| `calibrate_heavy_eval.py`, `calibrate_bond_orders.py` | calibration of the above against real molecules |
| `evaluate_checkpoint.py`, `compare_chemistry.py` | checkpoint evaluation and chemistry comparison |
| `generate.py`, `generate_candidates.py`, `match_reference_inhibitors.py`, `make_gallery.py` | sampling, filtering, and reporting |
| `diagnose_scale.py`, `diagnose_denoising.py`, `diagnose_aromaticity.py` | one-off investigations, each backing a claim made above |

Earlier exploratory approaches (a graph VAE, a SMILES-string VAE, and a
SMILES-token conditional denoiser) are no longer tracked — they didn't produce
chemically reliable results and were superseded by the 3D diffusion model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r src/molecule-diffusion/requirements.txt
```

## Reproduce

```bash
cd src/molecule-diffusion

# train (the v11 settings; ~345 s/epoch on an M-series GPU)
python scripts/train_conditional.py \
    --parp-npz parp_noH.npz --chembl-npz chembl_3d_noH.npz --no-hydrogens \
    --epochs 500 --hidden-dim 256 --n-layers 6 --lr 1e-4 --lr-min 1e-4 \
    --angle-loss-weight 0 --ckpt-every 10 --out checkpoints/v11_noH.pt

# evaluate a checkpoint against real molecules
python scripts/compare_chemistry.py checkpoints/v11_noH_epoch_499.pt \
    --n-samples 256 --n-steps 500

# generate candidates and build the gallery
python scripts/generate_candidates.py checkpoints/v11_noH_epoch_499.pt \
    --n-samples 1024 --out-dir candidates_v11
python scripts/make_gallery.py --candidates candidates_v11/candidates.csv \
    --checkpoint v11_noH_epoch_499
```

Checkpoint selection is deliberately *not* left to the in-training diagnostic:
it samples 64 molecules, so its 95% interval is about ±8 points while real
differences between late epochs are 1–2 points. It is used to reject a
collapsed epoch; picking a winner means running `compare_chemistry.py` on the
last few tagged checkpoints.

## Tests

```bash
cd src/molecule-diffusion && pytest tests -q      # 127 tests, about 1.5 s
```

The suite includes invariance tests run on **randomly initialised** networks —
padding amount, padded-slot contents, permutation, rotation, translation, and
zero-centre-of-mass output. That detail matters: at initialisation the
coordinate MLP maps 0 to ≈0, which hid a real padding leak from tests that used
an untrained network.

Enable the pre-commit hook, which runs the suite whenever a commit touches the
model, with `git config core.hooksPath .githooks`.

## Limitations

- Aromaticity and stereochemistry are still wrong (see the results table).
  Generated 5/6-rings have a median out-of-plane thickness of 0.067 Å against
  0.006 Å for real rings, so many rings that should read as aromatic are
  reconstructed as saturated.
- Bond orders are inferred post hoc and are 85.2% exact on real molecules.
  Explicit bond-order diffusion (DiGress-style) would remove this step.
- 32% of samples still fail to produce a usable molecule.
- No binding, docking or activity model of any kind. Similarity to known
  inhibitors is structural resemblance only.

## Licence and data

Code is MIT licensed — see [LICENSE](LICENSE).

No chemical data is redistributed here. The training sets are built locally from
[ChEMBL](https://www.ebi.ac.uk/chembl/), which is released under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); the scripts in
`scripts/` fetch and prepare it, and the resulting `.npz` files are gitignored.
Anyone redistributing data derived from ChEMBL should attribute it accordingly.

## Citation

Follows the EDM approach: Hoogeboom, Satorras, Vignac & Welling, *"Equivariant
Diffusion for Molecule Generation in 3D"*, ICML 2022,
[arXiv:2203.17003](https://arxiv.org/abs/2203.17003). This repository is a
prototype for learning and iteration, not a finalised generative chemistry
model.
