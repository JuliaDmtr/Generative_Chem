"""
Drop explicit hydrogens from a conformer .npz, keeping heavy atoms only.

Why: the model's cost grows with the number of atoms squared, and a molecule
counts as valid only if *every* atom is right. Dropping H takes PARP molecules
from ~42 atoms (median) to ~24-30 heavy atoms, i.e. roughly 0.3x the pairwise
cost per step and far fewer atoms that can go wrong. Most 3D molecule
generators work on heavy atoms and let RDKit add hydrogens back afterwards.

The heavy-atom coordinates are NOT recomputed: they come from conformers that
were embedded *with* explicit hydrogens (prepare_chembl.py), which is what
makes the geometry realistic. We only delete the H rows.

Usage (from src/molecule-diffusion):
    python scripts/strip_hydrogens.py chembl_3d.npz chembl_3d_noH.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from utils import ATOM2IDX

H_INDEX = ATOM2IDX["H"]


def strip_hydrogens(positions, atom_types, min_heavy_atoms: int = 3):
    """Return (positions, atom_types, n_dropped) with every H row removed.

    Molecules left with fewer than min_heavy_atoms heavy atoms are dropped
    entirely (nothing useful to learn from, and they break the geometry
    diagnostics).
    """
    out_pos, out_types, n_dropped = [], [], 0
    for pos, types in zip(positions, atom_types):
        keep = np.asarray(types) != H_INDEX
        if keep.sum() < min_heavy_atoms:
            n_dropped += 1
            continue
        out_pos.append(np.asarray(pos, dtype=np.float32)[keep])
        out_types.append(np.asarray(types, dtype=np.int64)[keep])
    return out_pos, out_types, n_dropped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("in_npz")
    p.add_argument("out_npz")
    p.add_argument("--min-heavy-atoms", type=int, default=3)
    args = p.parse_args()

    data = np.load(args.in_npz, allow_pickle=True)
    pos, types, n_dropped = strip_hydrogens(
        data["positions"], data["atom_types"], min_heavy_atoms=args.min_heavy_atoms)
    if not pos:
        print("No molecules left after stripping hydrogens.", file=sys.stderr)
        sys.exit(1)

    before = np.array([len(t) for t in data["atom_types"]])
    after = np.array([len(t) for t in types])
    np.savez(args.out_npz,
             positions=np.array(pos, dtype=object),
             atom_types=np.array(types, dtype=object))
    print(f"{args.in_npz}: {len(before):,} molecules, {np.median(before):.0f} atoms (median), max {before.max()}")
    print(f"{args.out_npz}: {len(after):,} molecules, {np.median(after):.0f} heavy atoms (median), max {after.max()}"
          f"{f' — dropped {n_dropped:,} with < {args.min_heavy_atoms} heavy atoms' if n_dropped else ''}")
    print(f"   pairwise-cost ratio vs input: ~{np.mean(after ** 2) / np.mean(before ** 2):.2f}x")


if __name__ == "__main__":
    main()
