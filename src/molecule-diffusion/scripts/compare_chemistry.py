"""
Compare the *chemistry* of generated molecules against real ones, not just
whether RDKit accepts them.

Validity says a molecule is well-formed; it does not say it looks like a drug.
Real PARP inhibitors are flat and aromatic with few stereocentres, while an
undertrained model produces saturated, strained, highly stereogenic cages that
are perfectly valid. These properties make that difference measurable:

  aromatic rings per molecule   real PARP inhibitors are aromatic-heavy
  fraction of sp3 carbons       "Fsp3": low for flat aromatics, high for cages
  stereocentres per molecule    real inhibitors have few
  ring sizes                    3- and 4-rings are rare in real chemistry
  heteroatom fraction, MW, logP  coarse drug-likeness

Both sides go through the identical pipeline (heavy-atom coordinates ->
bond_orders.assign_bond_orders), so reconstruction errors affect both equally
and the comparison stays fair.

Usage (from src/molecule-diffusion):
    python scripts/compare_chemistry.py checkpoints/v8_noH_epoch_99.pt \
        --parp-npz parp_noH.npz --n-samples 256 --n-steps 500
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

from bond_orders import assign_bond_orders
from dataset import QM9Dataset, sample_molecule_sizes
from evaluate_checkpoint import generate, load_model
from utils import COND_POSITIVE, IDX2ATOM

PROPERTIES = [
    ("aromatic rings", lambda m: rdMolDescriptors.CalcNumAromaticRings(m)),
    ("aliphatic rings", lambda m: rdMolDescriptors.CalcNumAliphaticRings(m)),
    ("fraction sp3 C", lambda m: rdMolDescriptors.CalcFractionCSP3(m)),
    ("stereocentres", lambda m: len(Chem.FindMolChiralCenters(m, useLegacyImplementation=False,
                                                              includeUnassigned=True))),
    ("heteroatom fraction", lambda m: (sum(1 for a in m.GetAtoms() if a.GetSymbol() not in ("C", "H"))
                                       / max(m.GetNumHeavyAtoms(), 1))),
    ("rotatable bonds", lambda m: rdMolDescriptors.CalcNumRotatableBonds(m)),
    ("molecular weight", lambda m: Descriptors.MolWt(m)),
    ("logP", lambda m: Crippen.MolLogP(m)),
]


def molecules_to_mols(molecules):
    """(symbols, coords) pairs -> single-fragment RDKit Mols (others dropped)."""
    out = []
    for symbols, coords in molecules:
        mol = assign_bond_orders(symbols, np.asarray(coords, dtype=np.float64))
        if mol is not None and len(Chem.GetMolFrags(mol)) == 1:
            out.append(mol)
    return out


def summarise(mols):
    stats = {name: np.array([fn(m) for m in mols], dtype=float) for name, fn in PROPERTIES}
    rings = Counter()
    for m in mols:
        rings.update(len(r) for r in m.GetRingInfo().AtomRings())
    return stats, rings


def report(name_a, mols_a, name_b, mols_b):
    sa, ra = summarise(mols_a)
    sb, rb = summarise(mols_b)
    print(f"\n{'property':22s} {name_a:>16s} {name_b:>16s}")
    print("-" * 58)
    for name, _ in PROPERTIES:
        a, b = sa[name], sb[name]
        print(f"{name:22s} {a.mean():10.2f}±{a.std():4.2f} {b.mean():10.2f}±{b.std():4.2f}")
    ta, tb = max(sum(ra.values()), 1), max(sum(rb.values()), 1)
    sizes = sorted(set(ra) | set(rb))
    print(f"\n{'ring size':22s} {name_a:>16s} {name_b:>16s}")
    print("-" * 58)
    for s in sizes:
        if ra[s] / ta < 0.005 and rb[s] / tb < 0.005:
            continue
        print(f"{str(s) + '-membered':22s} {ra[s] / ta:15.1%} {rb[s] / tb:15.1%}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt")
    p.add_argument("--parp-npz", default="parp_noH.npz")
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--sampler", default="ddim")
    p.add_argument("--ddim-noise", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-hydrogens", action="store_true", default=True)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--show", type=int, default=10, help="print this many generated SMILES")
    args = p.parse_args()
    device = torch.device(args.device)

    parp = QM9Dataset(args.parp_npz)
    rng = np.random.default_rng(args.seed)
    real_idx = rng.choice(len(parp), min(args.n_samples, len(parp)), replace=False)
    real = molecules_to_mols([([IDX2ATOM[int(t)] for t in parp[int(i)]["types"]],
                               parp[int(i)]["pos"]) for i in real_idx])

    model, epoch = load_model(args.ckpt, device)
    sizes = sample_molecule_sizes(parp, args.n_samples, seed=args.seed)
    gen = molecules_to_mols(generate(model, sizes, COND_POSITIVE, args, device))

    print(f"{args.ckpt} (epoch {epoch}), {args.n_samples} samples at {args.n_steps} steps")
    print(f"usable single molecules: generated {len(gen)}/{args.n_samples} "
          f"({len(gen) / args.n_samples:.1%}), real {len(real)}/{len(real_idx)} "
          f"({len(real) / len(real_idx):.1%})")
    if not gen:
        print("No generated molecule could be reconstructed — nothing to compare.")
        return
    report("generated", gen, "real PARP", real)
    if args.show:
        print(f"\ngenerated examples:")
        for smi in sorted({Chem.MolToSmiles(m) for m in gen})[:args.show]:
            print(f"  {smi}")
        print(f"\nreal examples:")
        for smi in sorted({Chem.MolToSmiles(m) for m in real})[:3]:
            print(f"  {smi}")


if __name__ == "__main__":
    main()
