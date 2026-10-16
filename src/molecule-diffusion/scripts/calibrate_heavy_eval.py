"""
Calibration for the heavy-atom checks (heavy_atom_eval.py): run them on REAL
conformers with the hydrogens stripped. Real molecules must pass at a high
rate, otherwise the measurement is broken rather than the model.

Also verifies, where a with-H version of the same conformer is available, that
heavy-only perception recovers exactly the heavy-heavy bond set the with-H
molecule has.

Usage (from src/molecule-diffusion):
    python scripts/calibrate_heavy_eval.py parp_3d_old_noH.npz --with-h-npz parp_3d.npz
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from heavy_atom_eval import heavy_atom_checks, heavy_bond_set, heavy_mol_from_coords
from utils import ATOM2IDX, IDX2ATOM


def reference_bond_set(symbols, coords):
    """Heavy-heavy bonds of the with-H molecule, as perceived by the (trusted)
    with-H pipeline. Returns None if that pipeline itself fails."""
    mol = heavy_mol_from_coords(symbols, coords)  # builds + perceives connectivity
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    remap = {old: new for new, old in enumerate(keep)}
    return {frozenset((remap[b.GetBeginAtomIdx()], remap[b.GetEndAtomIdx()]))
            for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in remap and b.GetEndAtomIdx() in remap}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("noh_npz")
    p.add_argument("--with-h-npz", default=None,
                   help="Same conformers WITH hydrogens, in the same order (bond-set cross-check).")
    p.add_argument("--n", type=int, default=500)
    args = p.parse_args()

    d = np.load(args.noh_npz, allow_pickle=True)
    n = min(args.n, len(d["positions"]))
    res = [heavy_atom_checks([IDX2ATOM[int(t)] for t in d["atom_types"][i]], d["positions"][i])
           for i in range(n)]
    rings, bad = Counter(), Counter()
    for r in res:
        rings.update(r["ring_sizes"])
        bad.update(r["bad_valence_atoms"])
    total_rings = sum(rings.values())
    print(f"=== {args.noh_npz}: {n} real conformers, hydrogens stripped ===")
    for key in ("perceived", "connected", "valence_ok", "plausible"):
        print(f"  {key:11s}: {sum(r[key] for r in res) / n:6.1%}")
    print(f"  ring sizes : " + (", ".join(f"{s}-ring {c / total_rings:.0%}" for s, c in sorted(rings.items()))
                                 or "no rings"))
    if bad:
        print(f"  over-coordinated atoms seen: {dict(bad)}")

    if args.with_h_npz:
        dh = np.load(args.with_h_npz, allow_pickle=True)
        assert len(dh["positions"]) == len(d["positions"]), "the two files must hold the same conformers in order"
        same = comparable = 0
        for i in range(n):
            ref = reference_bond_set([IDX2ATOM[int(t)] for t in dh["atom_types"][i]], dh["positions"][i])
            if ref is None:
                continue
            comparable += 1
            mol = heavy_mol_from_coords([IDX2ATOM[int(t)] for t in d["atom_types"][i]], d["positions"][i])
            same += mol is not None and heavy_bond_set(mol) == ref
        print(f"  bond graph identical to the with-H reference: {same}/{comparable}"
              f" ({same / max(comparable, 1):.1%})")


if __name__ == "__main__":
    main()
