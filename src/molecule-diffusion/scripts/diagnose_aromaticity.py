"""
Why do reconstructed molecules have too many aliphatic rings?

Generated molecules come out with 2.60 aliphatic rings each against 1.46 for
real PARP inhibitors (scripts/compare_chemistry.py). Roughly half of that gap
is the model building ring sizes that cannot be aromatic at all (3-, 4-,
7-membered). The rest is 5- and 6-rings that end up saturated: only 40% of
generated 5/6-rings are perceived aromatic, against 68% of real ones.

That second part has two possible causes, and they call for opposite fixes:
  (a) the model really makes puckered, saturated rings, or
  (b) the rings are right but their geometry is slightly off, and
      bond_orders.py only calls a ring aromatic when it is flat to 0.05 A RMS.

The script separates them with one number: RMS out-of-plane thickness of each
5/6-ring, which needs no bond orders and so is free of reconstruction error.
Real conformers are also perturbed with Gaussian noise to show how fast
aromatic perception decays with geometric error, which calibrates how much
flatness error the pipeline tolerates.

Metrics match compare_chemistry.py exactly (RDKit CalcNumAromaticRings /
CalcNumAliphaticRings over single-fragment reconstructions, same random
sample), so the "real, sigma 0" row reproduces its real-PARP column.

Usage (from src/molecule-diffusion):
    python scripts/diagnose_aromaticity.py --n 256
    python scripts/diagnose_aromaticity.py --n 256 --ckpt checkpoints/v10_noH_epoch_299.pt
"""

from __future__ import annotations

import argparse

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

from bond_orders import assign_bond_orders
from heavy_atom_eval import heavy_mol_from_coords
from utils import IDX2ATOM

from bond_orders import RING_FLATNESS_TOL, ring_thickness_of

# Real 5/6-rings are sharply bimodal: 70% sit below 0.02 A (aromatic) and 25%
# above 0.12 A (genuinely saturated), with only 2% in between. So real data
# cannot distinguish a 0.03 tolerance from a 0.12 one -- measured, accuracy is
# flat across that range (scripts/calibrate_bond_orders.py --flatness-tol).
# Generated rings land in exactly that empty band, which means the tolerance
# partly manufactures their aromatic-ring count. The share of rings inside the
# band is therefore reported: it says how much of the number is a judgement
# call rather than a measurement.
DECISIVE_BAND = (0.05, 0.12)


def flatness(symbols, coords):
    """Thickness of every 5/6-ring in the perceived skeleton (no bond orders)."""
    skeleton = heavy_mol_from_coords(symbols, coords)
    if skeleton is None:
        return []
    return [ring_thickness_of(coords, r) for r in Chem.GetSymmSSSR(skeleton) if 5 <= len(r) <= 6]


def ring_counts(symbols, coords):
    """Aromatic / aliphatic / 5-6-ring counts, exactly as compare_chemistry counts them."""
    mol = assign_bond_orders(symbols, coords)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return {"arom": rdMolDescriptors.CalcNumAromaticRings(mol),
            "ali": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "r56": sum(1 for s in sizes if 5 <= s <= 6)}


def row(label, mols, sigma, seed):
    rng = np.random.default_rng(seed)
    built, arom, ali, r56, thick = 0, 0, 0, 0, []
    for symbols, coords in mols:
        xyz = coords + rng.normal(0.0, sigma, coords.shape) if sigma > 0 else coords
        thick += flatness(symbols, xyz)
        res = ring_counts(symbols, xyz)
        if res is None:
            continue
        built += 1
        arom += res["arom"]
        ali += res["ali"]
        r56 += res["r56"]
    thick = np.array(thick) if thick else np.array([np.nan])
    lo, hi = DECISIVE_BAND
    print(f"{label:>22s} {built / max(len(mols),1):6.1%} {arom / max(built,1):9.2f} "
          f"{ali / max(built,1):8.2f} {arom / max(r56,1):14.0%} "
          f"{np.nanmedian(thick):16.3f} {np.mean(thick < RING_FLATNESS_TOL):11.0%} "
          f"{np.mean((thick >= lo) & (thick < hi)):10.0%}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="parp_noH.npz")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.08, 0.10])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", help="also sample this model and measure the same columns")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--sampler", default="ddim")
    p.add_argument("--ddim-noise", type=float, default=0.2)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(data["positions"]), min(args.n, len(data["positions"])), replace=False)
    real = [([IDX2ATOM[int(t)] for t in data["atom_types"][i]],
             data["positions"][i].astype(np.float64)) for i in idx]

    print(f"{len(real)} real PARP conformers (same sample as compare_chemistry.py)")
    print(f"bond_orders.py calls a 5/6-ring aromatic only below {RING_FLATNESS_TOL} A RMS thickness\n")
    print(f"{'':>22s} {'built':>6s} {'arom/mol':>9s} {'ali/mol':>8s} "
          f"{'5-6 arom':>14s} {'med thickness':>16s} {'flat enough':>11s} "
          f"{'in 0.05-0.12':>10s}")
    print("-" * 104)
    for sigma in args.sigmas:
        row(f"real, noise {sigma:.2f}", real, sigma, args.seed)

    if args.ckpt:
        import torch
        from dataset import QM9Dataset, sample_molecule_sizes
        from evaluate_checkpoint import generate, load_model
        from utils import COND_POSITIVE
        device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        model, epoch = load_model(args.ckpt, device)
        sizes = sample_molecule_sizes(QM9Dataset(args.npz), args.n, seed=args.seed)
        gen = [(s, np.asarray(c, dtype=np.float64))
               for s, c in generate(model, sizes, COND_POSITIVE, args, device)]
        print("-" * 104)
        row(f"generated (ep {epoch})", gen, 0.0, args.seed)

    print("\nThe generated row's median thickness places the model on this noise ladder:")
    print("that is how much geometric error its rings carry, and therefore how much of the")
    print("missing aromaticity is reconstruction loss rather than wrong chemistry.")
    print("A large share in the 0.05-0.12 A band means the aromatic-ring count for that row")
    print("depends on the tolerance rather than on the chemistry, and should be quoted with it.")


if __name__ == "__main__":
    main()
