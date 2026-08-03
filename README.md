# PARP Conditional Molecular Generation

This repository contains a research prototype for conditional generation of PARP-like molecules using SMILES token sequences. The project explores whether a transformer-based denoising model can learn to reconstruct chemically structured strings and generate new candidates under a simple condition.

## Overview

The current implementation is not a classic diffusion model in the strict sense. Instead, it uses a conditional transformer trained in a denoising-style setup:

- SMILES strings are tokenized into chemically meaningful fragments.
- A vocabulary is built from the available molecular corpus.
- The model receives a corrupted SMILES sequence and learns to reconstruct the original sequence.
- Generation is performed autoregressively with syntax-aware filtering to avoid obviously invalid prefixes.

The goal is to explore conditional molecular generation for a small, focused chemistry task rather than to claim state-of-the-art performance.

## What the model does

The pipeline consists of:

1. A regex-based SMILES tokenizer and vocabulary builder.
2. A conditional transformer model over token IDs.
3. A denoising training objective that learns to recover clean SMILES from corrupted inputs.
4. A generation loop that samples candidate SMILES and filters them with simple structural heuristics.

## Related modeling approaches in this project

This repository explores three related directions for molecular generation, with the current work in the SMILES diffusion track as the primary focus:

- Graph VAE: a variational autoencoder operating on graph representations of molecules. It learns a latent space for molecular graphs and reconstructs graph structures from that space. This approach is more structure-aware than string-based generation, but it is also more complex to implement and train.
- SMILES VAE: a variational autoencoder over SMILES strings. It maps molecules into a continuous latent space and decodes samples back into SMILES. This approach is simpler and faster to experiment with, but it often struggles with chemical validity and syntax consistency.
- Conditional SMILES denoiser: the main direction of the current project. Rather than relying only on latent-space reconstruction, this model learns to recover a clean SMILES sequence from a corrupted version. It uses a transformer architecture and a simple conditioning signal to guide generation toward the desired molecular property or class.

In short, the Graph VAE focuses on graph structure, the SMILES VAE focuses on latent string generation, and the current SMILES diffusion work focuses on denoising-based sequence generation for chemically meaningful SMILES output.

## Repository structure

- [src/smiles_diffusion/conditional_smiles_denoiser.py](src/smiles_diffusion/conditional_smiles_denoiser.py) — tokenizer, vocabulary, corruption helpers, syntax checks, and model implementation.
- [src/smiles_diffusion/train_conditional_denoiser.py](src/smiles_diffusion/train_conditional_denoiser.py) — training loop, data splitting, evaluation, and generation script.
- [src/tests/test_conditional_smiles_denoiser.py](src/tests/test_conditional_smiles_denoiser.py) — regression tests for tokenization, corruption, syntax filtering, and model output shapes.
- [src/smiles_vae/chembl_processor.py](src/smiles_vae/chembl_processor.py) — helper code for loading ChEMBL-based background data.

## Setup

This project uses Python and PyTorch. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies are listed in [pyproject.toml](pyproject.toml).

## Run training

```bash
python src/smiles_diffusion/train_conditional_denoiser.py
```

## Run tests

```bash
pytest -q src/tests/test_conditional_smiles_denoiser.py
```

## Current status

This is an early-stage research prototype. It can train, generate candidate SMILES strings, and apply basic validity heuristics, but it is not yet a production-grade or state-of-the-art molecular generation system.

## Future directions

Potential next steps include:

- stronger chemistry-aware generation constraints,
- RDKit-based validity and novelty filtering,
- larger or more expressive transformer architectures,
- graph-based or diffusion-based alternatives for molecular generation.

## Citation / note

This repository is intended as a practical experimentation codebase for conditional molecular generation with SMILES. It should be viewed as a prototype for learning and iteration rather than a finalized generative chemistry model.
