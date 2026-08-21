# PARP Conditional Molecular Generation

# PARP Conditional Molecular Generation

This repository contains a research prototype for conditional generation of PARP-inhibitor-like molecules: an **E(3)-equivariant diffusion model** (EDM-style, Hoogeboom et al. 2022, ICML) that jointly generates 3D atom positions and atom types, conditioned on whether a molecule should resemble a known PARP inhibitor.

## Overview

The model diffuses molecules directly in 3D space rather than as SMILES token sequences:

- Atom positions and atom types are jointly noised and denoised, following the standard EDM recipe (later SOTA methods like GeoLDM and MiDi build on the same core idea).
- An E(3)-equivariant graph network (EGNN) is the denoiser backbone, so the model never has to separately learn that every rotated/translated copy of a molecule is equally valid.
- The model is trained with 3-way class conditioning — a real PARP inhibitor, a size-matched generic ChEMBL molecule, or a representative random ChEMBL molecule — so generation can be steered toward the PARP-inhibitor class at sampling time.
- Generated 3D coordinates are converted back to molecules/SMILES via RDKit bond perception, giving a concrete chemical-validity metric for generated samples.

The goal is to explore conditional 3D molecular generation for a small, focused chemistry task rather than to claim state-of-the-art performance.

## What the model does

The pipeline consists of:

1. RDKit-based conformer generation to build 3D training data from SMILES (ChEMBL background pool + a curated PARP-inhibitor set).
2. An E(3)-equivariant denoiser (EGNN) operating jointly on atom positions and atom types.
3. A diffusion training objective (noise prediction) with 3-way class conditioning.
4. A reverse-diffusion sampler that generates new 3D molecules under a chosen condition, followed by RDKit-based validity evaluation.

## Repository structure

- [src/molecule-diffusion/SKILL.md](src/molecule-diffusion/SKILL.md) — full walkthrough: setup, quick start, conditional PARP training, and guidance on managing long training runs.
- [src/molecule-diffusion/references/approach.md](src/molecule-diffusion/references/approach.md) — the method explained: equivariance, diffusion schedule, data format, and known simplifications vs. the original EDM paper.
- `src/molecule-diffusion/scripts/` — `egnn.py` (denoiser backbone), `diffusion.py` (forward/reverse process), `dataset.py`, `train.py`/`train_conditional.py`, `generate.py`, `evaluate.py`, `prepare_chembl.py`/`prepare_parp_conformers.py` (SMILES → 3D `.npz` conversion), `eval_checkpoint_trend.py` (epoch-vs-validity trend across checkpoints).
- `src/molecule-diffusion/tests/` — equivariance checks, diffusion-math sanity checks, conditioning tests, and an end-to-end smoke test.

Earlier exploratory approaches (a graph VAE, a SMILES-string VAE, and a SMILES-token conditional denoiser) are no longer tracked in this repository — they didn't produce chemically reliable results and have been superseded by the 3D diffusion approach above.

## Setup

This project uses Python and PyTorch. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r src/molecule-diffusion/requirements.txt
```

Dependencies are listed in [pyproject.toml](pyproject.toml) and [src/molecule-diffusion/requirements.txt](src/molecule-diffusion/requirements.txt).

## Run training

```bash
cd src/molecule-diffusion
# Unconditional, on the built-in synthetic dataset (pipeline check only)
python scripts/train.py --synthetic --epochs 50 --out ckpt.pt

# PARP-conditioned, on real data
python scripts/train_conditional.py --parp-npz parp_3d.npz --chembl-npz chembl_3d.npz \
    --epochs 300 --hidden-dim 192 --n-layers 6 --out checkpoints/conditional_checkpoint.pt
```

See [src/molecule-diffusion/SKILL.md](src/molecule-diffusion/SKILL.md) for the full quick start, sampling/evaluation commands, and notes on resuming/backgrounding long runs.

## Run tests

```bash
cd src/molecule-diffusion
pytest tests/
```

## Current status

This is an early-stage research prototype. It can train, generate candidate 3D molecules under PARP-inhibitor conditioning, and evaluate them for chemical validity via RDKit, but it is not yet a production-grade or state-of-the-art molecular generation system.

## Future directions

Potential next steps include:

- stronger chemistry-aware generation constraints,
- explicit bond-order diffusion (DiGress-style) instead of post-hoc bond inference,
- larger or more expressive equivariant architectures or more training data,
- novelty/uniqueness evaluation (beyond the validity metric already implemented).

## Citation / note

This repository is intended as a practical experimentation codebase for conditional 3D molecular generation, following the EDM approach (Hoogeboom, Satorras, Vignac & Welling, *"Equivariant Diffusion for Molecule Generation in 3D"*, ICML 2022, arXiv:2203.17003). It should be viewed as a prototype for learning and iteration rather than a finalized generative chemistry model.
