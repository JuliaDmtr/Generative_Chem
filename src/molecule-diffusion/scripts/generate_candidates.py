"""
Generate PARP-inhibitor candidates from a heavy-atom (no-hydrogen) model.

The full path from model output to a molecule you can look at:
  sample heavy-atom coordinates + types
    -> perceive the bond graph            (heavy_atom_eval.py, exact on real data)
    -> assign bond orders from geometry   (bond_orders.py, 70% exact on real data)
    -> filter for chemistry worth keeping
    -> report novelty and similarity to known PARP inhibitors

The filters reject what the model is known to get wrong rather than what is
merely unusual: fragments, impossible valences, 3- and 4-membered rings (1% and
1.2% of real PARP rings, but 25% of generated ones), and molecules with no
aromatic ring at all (real inhibitors average 2.65). Strain energy at the
generated geometry is reported, not filtered on, since the as-generated
conformer is not relaxed.

Nothing here says a molecule inhibits PARP. It says the molecule is
well-formed, chemically reasonable, novel against the training set, and
similar to known inhibitors — a shortlist to look at, not a prediction.

Usage (from src/molecule-diffusion):
    python scripts/generate_candidates.py checkpoints/v9_noH_epoch_199.pt \
        --n-samples 1024 --out-dir candidates_v9
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw, Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

from bond_orders import assign_bond_orders
from dataset import QM9Dataset, sample_molecule_sizes
from evaluate_checkpoint import load_model
from generate import molecules_from_output
from utils import COND_NULL, COND_POSITIVE, IDX2ATOM, make_condition_batch


def strain_per_heavy_atom(mol):
    """MMFF energy at the generated geometry (no relaxation), per heavy atom.
    Reported for triage: real relaxed conformers sit near 1-5 kcal/mol/atom."""
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        return ff.CalcEnergy() / max(mol.GetNumHeavyAtoms(), 1) if ff else None
    except Exception:
        return None


def reference_fingerprints(parp_npz):
    """Morgan fingerprints + canonical SMILES of the training inhibitors."""
    ds = QM9Dataset(parp_npz)
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps, smiles = [], set()
    for i in range(len(ds)):
        mol = assign_bond_orders([IDX2ATOM[int(t)] for t in ds[i]["types"]],
                                 ds[i]["pos"].astype(np.float64))
        if mol is None:
            continue
        smiles.add(Chem.MolToSmiles(mol))
        fps.append(gen.GetFingerprint(mol))
    return fps, smiles, gen


# Motifs RDKit accepts but a chemist would reject on sight. Each one was seen
# in the v9 candidate list, so these are observed failures, not hypothetical.
# Aromatic heteroatom chains are deliberately NOT flagged: tetrazoles and
# triazoles are ordinary medicinal chemistry.
STRUCTURAL_ALERTS = {
    "N-halogen bond": "[#7]-[F,Cl,Br,I]",
    "allene": "[CX2](=[#6])=[#6]",
    "peroxide": "[OX2]-[OX2]",
    "3+ aliphatic heteroatoms in a row": "[!#6;!#1;!a]~[!#6;!#1;!a]~[!#6;!#1;!a]",
}
_ALERTS = {name: Chem.MolFromSmarts(sma) for name, sma in STRUCTURAL_ALERTS.items()}
MIN_RING_FOR_TRIPLE_BOND = 8          # cyclooctyne is the smallest isolable cycloalkyne;
                                      # a triple bond in a 7-ring or smaller cannot exist


def structural_alerts(mol):
    """Names of chemically implausible motifs present in `mol` (empty = clean)."""
    found = [name for name, patt in _ALERTS.items() if patt is not None and mol.HasSubstructMatch(patt)]
    ring_info = mol.GetRingInfo()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.TRIPLE:
            continue
        sizes = [len(r) for r in ring_info.AtomRings()
                 if bond.GetBeginAtomIdx() in r and bond.GetEndAtomIdx() in r]
        if sizes and min(sizes) < MIN_RING_FOR_TRIPLE_BOND:
            found.append("triple bond in a small ring")
            break
    return found


def passes_filters(mol, args):
    """Returns (ok, reason). Reasons are counted so the yield is explainable."""
    if len(Chem.GetMolFrags(mol)) != 1:
        return False, "fragmented"
    rings = [len(r) for r in mol.GetRingInfo().AtomRings()]
    if any(r < args.min_ring_size for r in rings):
        return False, f"ring smaller than {args.min_ring_size}"
    if any(r > args.max_ring_size for r in rings):
        return False, f"ring larger than {args.max_ring_size}"
    if rdMolDescriptors.CalcNumAromaticRings(mol) < args.min_aromatic_rings:
        return False, "too few aromatic rings"
    mw = Descriptors.MolWt(mol)
    if not args.min_mw <= mw <= args.max_mw:
        return False, "molecular weight out of range"
    alerts = structural_alerts(mol)
    if alerts:
        return False, f"implausible motif: {alerts[0]}"
    return True, "accepted"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt")
    p.add_argument("--parp-npz", default="parp_noH.npz")
    p.add_argument("--n-samples", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--guidance-scale", type=float, default=1.0,
                   help=">1 extrapolates away from the null class toward the PARP class")
    p.add_argument("--min-ring-size", type=int, default=5)
    p.add_argument("--max-ring-size", type=int, default=7)
    p.add_argument("--min-aromatic-rings", type=int, default=1)
    p.add_argument("--min-mw", type=float, default=250.0)
    p.add_argument("--max-mw", type=float, default=550.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out-dir", default="candidates")
    args = p.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    model, epoch = load_model(args.ckpt, device)
    parp = QM9Dataset(args.parp_npz)
    sizes = sample_molecule_sizes(parp, args.n_samples, seed=args.seed)
    torch.manual_seed(args.seed)

    print(f"{args.ckpt} (epoch {epoch}): sampling {args.n_samples} molecules, "
          f"{args.n_steps} steps, guidance {args.guidance_scale}", flush=True)
    raw = []
    for start in range(0, args.n_samples, args.batch):
        chunk = sizes[start:start + args.batch]
        nm = torch.zeros(len(chunk), max(chunk), 1, device=device)
        for b, n in enumerate(chunk):
            nm[b, :n] = 1.0
        cond = make_condition_batch(COND_POSITIVE, len(chunk), device=device)
        null = make_condition_batch(COND_NULL, len(chunk), device=device) if args.guidance_scale != 1.0 else None
        with torch.no_grad():
            x, h = model.sample(nm, n_steps=args.n_steps, device=device, cond=cond,
                                null_cond=null, guidance_scale=args.guidance_scale)
        raw += molecules_from_output(x.cpu(), h.cpu(), nm.cpu())
        print(f"  sampled {len(raw)}/{args.n_samples}", flush=True)

    print("reconstructing molecules and filtering...", flush=True)
    ref_fps, ref_smiles, fpgen = reference_fingerprints(args.parp_npz)
    reasons, accepted, seen = Counter(), [], set()
    for symbols, coords in raw:
        mol = assign_bond_orders(symbols, np.asarray(coords, dtype=np.float64))
        if mol is None:
            reasons["could not build a molecule"] += 1
            continue
        ok, why = passes_filters(mol, args)
        reasons[why] += 1
        if not ok:
            continue
        smi = Chem.MolToSmiles(mol)
        if smi in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(smi)
        fp = fpgen.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps) if ref_fps else [0.0]
        accepted.append({
            "smiles": smi,
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
            "mol_weight": round(Descriptors.MolWt(mol), 1),
            "logp": round(Crippen.MolLogP(mol), 2),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "stereocentres": len(Chem.FindMolChiralCenters(mol, useLegacyImplementation=False,
                                                           includeUnassigned=True)),
            "max_similarity_to_known": round(float(max(sims)), 3),
            "novel": smi not in ref_smiles,
            "strain_per_heavy_atom": (lambda e: round(e, 1) if e is not None else "")(strain_per_heavy_atom(mol)),
        })

    accepted.sort(key=lambda r: -r["max_similarity_to_known"])
    csv_path = os.path.join(args.out_dir, "candidates.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(accepted[0].keys()) if accepted else ["smiles"])
        w.writeheader()
        w.writerows(accepted)

    print(f"\nfrom {args.n_samples} samples:")
    for why, k in reasons.most_common():
        print(f"  {k:5d}  {why}")
    print(f"\n{len(accepted)} candidates ({len(accepted) / args.n_samples:.1%} of samples) -> {csv_path}")
    if accepted:
        novel = sum(r["novel"] for r in accepted)
        print(f"  novel (not in the training set): {novel}/{len(accepted)}")
        sim = np.array([r["max_similarity_to_known"] for r in accepted])
        print(f"  similarity to nearest known inhibitor: median {np.median(sim):.2f}, max {sim.max():.2f}")
        mols = [Chem.MolFromSmiles(r["smiles"]) for r in accepted[:24]]
        img = os.path.join(args.out_dir, "top_candidates.png")
        Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(320, 260),
                             legends=[f"{r['smiles'][:28]}... sim {r['max_similarity_to_known']}"
                                      for r in accepted[:24]]).save(img)
        print(f"  pictures of the 24 most similar: {img}")


if __name__ == "__main__":
    main()
