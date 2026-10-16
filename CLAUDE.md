# PARP_Dif_mod

3D molecule generation for PARP-inhibitor candidates.

- `src/molecule-diffusion/`: EDM-style E(3)-equivariant diffusion model (EGNN denoiser), class-conditioned on PARP vs. ChEMBL.
- `src/protein-context-molecule-diffusion/`: protein-conditioned prototype (not yet audited with the invariance tests).

Python environment: `.venv/bin/python`. Tests: `cd src/molecule-diffusion && ../../.venv/bin/python -m pytest tests -q` (under a second).

## Rules for changing model code

These exist because the first version of this model passed its tests while padded atoms leaked into the coordinate update, EDM's coordinate normalization was missing, and the H_SCALE trick was a no-op. See the correction in `src/molecule-diffusion/CASE_STUDY.md`.

1. **Write tests first, and make them fail.** Every model change comes with a test that fails on the old code. Invariance tests (padding amount, padded-slot content, permutation, rotation, translation, zero-CoM output) must run on a network with **randomized parameters**: at initialization `coord_mlp(0) ≈ 0` hides leaks. See `tests/test_invariances.py`.
2. **Tests must actually run.** Never commit a test that wasn't executed, and never mark one "not runnable here".
3. **Compare against the reference implementation.** EDM: `ehoogeboom/e3_diffusion_for_molecules` (`egnn/egnn_new.py`, `equivariant_diffusion/en_diffusion.py`, defaults in `main_qm9.py`). Every difference goes into the comparison table in `references/approach.md`, marked intentional or open.
4. **Explain the mechanism, then check that it takes effect.** "The paper does X" isn't enough: say why X works and add a check that it changes the computation. A constant scale before a linear layer, for example, does nothing.
5. **Measure distributions, not averages.** Geometry diagnostics (nearest-neighbour distance, Rg, angles, validity) are reported per molecule-size bucket and must not depend on padding.
6. **Every claim in a write-up needs a rerunnable script.** No "works well" or "fixed" in docs or commit messages without a script or test that shows it.
7. **Bump `ARCH_VERSION` in `utils.py`** whenever a change makes old checkpoints compute something different. Loaders refuse mismatched checkpoints via `check_arch_version`.

## Before long training runs

Run a small overfit sanity run first (about 64 molecules, `angle_loss_weight=0`). Check per-size geometry against real molecules, padding independence, and Rg stability through sampling. Only then start a multi-hour or multi-day run.
