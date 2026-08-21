"""
Build a 3D-conformer dataset of PARP inhibitors, in the exact .npz format
QM9Dataset expects (see dataset.py):

    positions:  object array of (n_i, 3) float32   [Angstroms]
    atom_types: object array of (n_i,)  int64       [indices into ATOM_VOCAB]

Unlike prepare_chembl.py (one conformer per SMILES, because ChEMBL is
large), this generates several ETKDG conformers per PARP-inhibitor SMILES
(different random seeds), since the curated PARP set only has a few
hundred distinct molecules. This gives the conditional model more than a
single 3D snapshot per molecule to learn from, without duplicating exact
copies (each conformer is geometrically distinct after MMFF relaxation).

Usage:
    python prepare_parp_conformers.py \
        --csv ../../Datasets/PARP_inhibitors_July26/merged_parp_inhibitors.csv \
        --n-conformers 5 --out parp_3d.npz

Needs rdkit (pip install rdkit) in addition to this skill's requirements.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from prepare_chembl import smiles_to_3d
from utils import ATOM2IDX


def convert(smiles_list: list[str], n_conformers: int = 5, seed: int = 0,
            max_atoms: int | None = None):
    """Embed n_conformers ETKDG conformers per SMILES (skipping SMILES that
    fail to parse, and individual conformers that fail to embed, are
    out-of-vocab, or exceed max_atoms). Returns (positions, atom_types,
    n_molecules_with_at_least_one_conformer).
    """
    positions, atom_types = [], []
    n_molecules_kept = 0
    n_embed_fail = 0
    n_vocab_fail = 0
    n_too_big = 0

    for i, smi in enumerate(smiles_list):
        kept_any = False
        for c in range(n_conformers):
            result = smiles_to_3d(smi, seed=seed + i * 1000 + c)
            if result is None:
                n_embed_fail += 1
                continue
            symbols, coords = result
            if max_atoms is not None and len(symbols) > max_atoms:
                n_too_big += 1
                continue
            try:
                idx = np.array([ATOM2IDX[s] for s in symbols], dtype=np.int64)
            except KeyError:
                n_vocab_fail += 1
                continue

            positions.append(coords.astype(np.float32))
            atom_types.append(idx)
            kept_any = True

        if kept_any:
            n_molecules_kept += 1

        if (i + 1) % 50 == 0:
            print(f"   ... processed {i + 1:,}/{len(smiles_list):,} SMILES "
                  f"(kept {len(positions):,} conformers from {n_molecules_kept:,} molecules)")

    print(f"✅ Converted {len(positions):,} conformers from "
          f"{n_molecules_kept:,}/{len(smiles_list):,} molecules.")
    print(f"   Skipped: {n_embed_fail:,} embed failures, {n_too_big:,} over max-atoms, "
          f"{n_vocab_fail:,} out-of-vocab elements.")

    return positions, atom_types, n_molecules_kept


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="../../../Datasets/PARP_inhibitors_July26/merged_parp_inhibitors.csv",
                    help="CSV with a 'smiles' column (as produced for the curated PARP inhibitor set).")
    p.add_argument("--n-conformers", type=int, default=5,
                    help="Number of ETKDG conformers to embed per SMILES.")
    p.add_argument("--max-atoms", type=int, default=None,
                    help="Skip conformers with more atoms (incl. H) than this.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="parp_3d.npz")
    args = p.parse_args()

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        smiles_list = [row["smiles"] for row in reader if row.get("smiles")]

    print(f"Loaded {len(smiles_list):,} SMILES from {args.csv}")
    positions, atom_types, n_kept = convert(
        smiles_list, n_conformers=args.n_conformers, seed=args.seed, max_atoms=args.max_atoms,
    )

    if not positions:
        print("❌ No conformers converted successfully — nothing to save.", file=sys.stderr)
        sys.exit(1)

    np.savez(
        args.out,
        positions=np.array(positions, dtype=object),
        atom_types=np.array(atom_types, dtype=object),
    )
    print(f"💾 Saved {len(positions):,} conformers ({n_kept:,} distinct molecules) to {args.out}")
    print(f"   Train with: python train_conditional.py --parp-npz {args.out} ...")


if __name__ == "__main__":
    main()
