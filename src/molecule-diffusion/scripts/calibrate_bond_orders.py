"""
Calibration for bond_orders.py: can it rebuild a molecule it has never seen,
given only its heavy-atom coordinates?

Each real PARP conformer is reconstructed from its heavy atoms alone and
compared with the SMILES obtained from the SAME conformer WITH hydrogens via
RDKit's DetermineBonds (which is reliable when hydrogens are present). Two
numbers are reported:

    built  — a chemically valid molecule came out at all
    exact  — it is the original molecule (identical canonical SMILES)

"exact" is the number that matters: it bounds how much any bond order in a
generated molecule can be trusted. The skeleton underneath is exact (100%,
see calibrate_heavy_eval.py), so whatever this misses is a bond-order or
tautomer mistake, not a wrong connectivity.

Usage (from src/molecule-diffusion):
    python scripts/calibrate_bond_orders.py --n 500
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

from bond_orders import smiles_from_heavy_coords
from heavy_atom_eval import heavy_mol_from_coords
from utils import IDX2ATOM

RDLogger.DisableLog("rdApp.*")


def reference_smiles(symbols, coords):
    """Ground truth: the with-hydrogen pipeline on the same conformer."""
    mol = heavy_mol_from_coords(symbols, coords)
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noh-npz", default="parp_noH.npz")
    p.add_argument("--with-h-npz", default="parp_chembl_3d.npz",
                   help="the SAME conformers with hydrogens, in the same order")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--show-mismatches", type=int, default=3)
    p.add_argument("--flatness-tol", type=float, default=None,
                   help="override bond_orders.RING_FLATNESS_TOL (the RMS out-of-plane "
                        "spread below which a 5/6-ring counts as aromatic)")
    args = p.parse_args()
    if args.flatness_tol is not None:
        import bond_orders
        bond_orders.RING_FLATNESS_TOL = args.flatness_tol

    noh = np.load(args.noh_npz, allow_pickle=True)
    wh = np.load(args.with_h_npz, allow_pickle=True)
    if len(noh["positions"]) != len(wh["positions"]):
        raise SystemExit("the two files must hold the same conformers in the same order")

    n = min(args.n, len(noh["positions"]))
    stats, mismatches = Counter(), []
    for i in range(n):
        symbols = [IDX2ATOM[int(t)] for t in noh["atom_types"][i]]
        got = smiles_from_heavy_coords(symbols, noh["positions"][i].astype(np.float64))
        stats["built"] += got is not None
        truth = reference_smiles([IDX2ATOM[int(t)] for t in wh["atom_types"][i]], wh["positions"][i])
        if truth is None:
            continue
        stats["comparable"] += 1
        if got == truth:
            stats["exact"] += 1
        elif got is not None and len(mismatches) < args.show_mismatches:
            mismatches.append((truth, got))

    c = max(stats["comparable"], 1)
    import bond_orders
    print(f"{n} real PARP conformers ({stats['comparable']} with a usable reference), "
          f"ring flatness tolerance {bond_orders.RING_FLATNESS_TOL:.3f} A")
    print(f"  built a valid molecule : {stats['built'] / n:6.1%}")
    print(f"  EXACT original molecule: {stats['exact'] / c:6.1%}")
    for truth, got in mismatches:
        print(f"\n  truth: {truth}\n  got  : {got}")


if __name__ == "__main__":
    main()
