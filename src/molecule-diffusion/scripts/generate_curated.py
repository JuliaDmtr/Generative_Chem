"""
Generate a curated set of molecules from a diffusion checkpoint by
rejection sampling with strict chemistry filters.

Unlike generate.py (which writes every raw sample), this script keeps only
molecules that pass in-memory RDKit checks and simple sanity gates so the
saved set is intentionally higher quality.

Usage:
    python generate_curated.py \
        --ckpt ../checkpoints/conditional_checkpoint_v2_epoch_351.pt \
        --condition positive --size-dataset-npz ../parp_3d.npz \
        --target-valid 100 --batch-size 64 --n-steps 300 \
        --guidance-scale 1.0 --out-dir ../samples_curated_351
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from dataset import QM9Dataset, sample_molecule_sizes
from diffusion import EquivariantMoleculeDiffusion
from evaluate import mol_from_coords, n_fragments
from generate import CONDITION_NAME_TO_LABEL, molecules_from_output, write_xyz
from utils import COND_NULL, make_condition_batch


def ring_sizes(mol):
    out = []
    for r in mol.GetRingInfo().AtomRings():
        out.append(len(r))
    return out


def energy_per_heavy_atom(mol):
    """MMFF (falling back to UFF for atom types MMFF can't parametrize)
    force-field energy of `mol` AT ITS GENERATED CONFORMER — no geometry
    optimization is performed, this scores the as-generated coordinates
    as-is — normalized by heavy-atom count so molecules of different
    sizes are comparable. Returns None if neither force field can
    parametrize the molecule (treated as a reject by the caller: an
    unparametrizable geometry is untrusted, not passed through).

    Calibrated against real, MMFF-relaxed parp_3d.npz conformers scored
    the same way: ground truth sits at median ~1.3, 99th-percentile ~5.2
    kcal/mol per heavy atom, while this model's generated molecules
    (which are dominated by strained 3-membered rings — see
    diagnose_scale.py's angle diagnostic) score in the tens to hundreds,
    with zero overlap observed between the two distributions in a
    ~300-molecule / ~14-molecule calibration sample.
    """
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props is not None else None
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(mol)
        energy = ff.CalcEnergy()
    except Exception:
        return None
    n_heavy = mol.GetNumHeavyAtoms()
    return energy / max(n_heavy, 1)


def pass_filters(mol, *, require_single: bool, min_heavy_atoms: int,
                  min_ring_size: int, max_ring_size: int, max_energy_per_heavy: float | None):
    if require_single and n_fragments(mol) != 1:
        return False, "multi_fragment"

    if mol.GetNumHeavyAtoms() < min_heavy_atoms:
        return False, "too_few_heavy_atoms"

    sizes = ring_sizes(mol)
    if sizes and min(sizes) < min_ring_size:
        return False, "undersized_ring"
    if sizes and max(sizes) > max_ring_size:
        return False, "oversized_ring"

    if max_energy_per_heavy is not None:
        epa = energy_per_heavy_atom(mol)
        if epa is None:
            return False, "unparametrizable_energy"
        if epa > max_energy_per_heavy:
            return False, "high_strain_energy"

    return True, "accepted"


def pick_sizes(args, n, seed_offset):
    if args.sizes:
        base = [int(s) for s in args.sizes.split(",") if s.strip()]
        return (base * (n // len(base) + 1))[:n]

    if args.size_dataset_npz:
        return sample_molecule_sizes(
            QM9Dataset(args.size_dataset_npz),
            n,
            seed=args.seed + seed_offset,
        )

    rng = np.random.default_rng(args.seed + seed_offset)
    return rng.integers(args.min_atoms, args.max_atoms + 1, size=n).tolist()


def build_model(ckpt_path: str, device: torch.device, use_ema: bool):
    ckpt = torch.load(ckpt_path, map_location=device)
    margs = ckpt["args"]
    cond_dim = margs.get("cond_dim", 0)
    model = EquivariantMoleculeDiffusion(
        hidden_dim=margs["hidden_dim"],
        n_layers=margs["n_layers"],
        timesteps=margs["timesteps"],
        cond_dim=cond_dim,
    ).to(device)
    state = ckpt["ema_state"] if use_ema else ckpt["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model, cond_dim


def sample_batch(model, cond_dim, args, device, n_batch, seed_offset):
    sizes = pick_sizes(args, n_batch, seed_offset)
    n_max = max(sizes)
    node_mask = torch.zeros(n_batch, n_max, 1, device=device)
    for i, n in enumerate(sizes):
        node_mask[i, :n] = 1.0

    if cond_dim > 0:
        cond = make_condition_batch(CONDITION_NAME_TO_LABEL[args.condition], n_batch, device=device)
        null_cond = None
        if args.guidance_scale != 1.0:
            null_cond = make_condition_batch(COND_NULL, n_batch, device=device)
        x_t, h_t = model.sample(
            node_mask,
            n_steps=args.n_steps,
            device=device,
            cond=cond,
            null_cond=null_cond,
            guidance_scale=args.guidance_scale,
        )
    else:
        x_t, h_t = model.sample(node_mask, n_steps=args.n_steps, device=device)

    return molecules_from_output(x_t, h_t, node_mask)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--target-valid", type=int, default=100,
                   help="How many accepted molecules to save.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=200,
                   help="Safety cap on rejection-sampling attempts.")
    p.add_argument("--n-steps", type=int, default=300)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--out-dir", type=str, default="samples_curated")
    p.add_argument("--manifest", type=str, default="manifest.csv")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")

    p.add_argument("--condition", type=str, default=None, choices=list(CONDITION_NAME_TO_LABEL))
    p.add_argument("--guidance-scale", type=float, default=1.0)

    p.add_argument("--sizes", type=str, default=None)
    p.add_argument("--min-atoms", type=int, default=4)
    p.add_argument("--max-atoms", type=int, default=19)
    p.add_argument("--size-dataset-npz", type=str, default=None)

    p.add_argument("--require-single", action="store_true", default=True)
    p.add_argument("--min-heavy-atoms", type=int, default=6)
    p.add_argument("--min-ring-size", type=int, default=5,
                   help="Reject any ring below this size — real rings are essentially "
                        "never 3- or 4-membered; this directly targets the strained "
                        "cyclopropane-like triangles this model tends to generate "
                        "(see diagnose_scale.py's angle diagnostic).")
    p.add_argument("--max-ring-size", type=int, default=8)
    p.add_argument("--max-energy-per-heavy", type=float, default=10.0,
                   help="Reject molecules whose MMFF/UFF force-field energy at their "
                        "generated conformer, normalized by heavy-atom count, exceeds "
                        "this (kcal/mol). Calibrated well above real MMFF-relaxed "
                        "parp_3d.npz conformers (99th pct. ~5.2) and well below this "
                        "model's typical strained output (observed min ~15.8) — see "
                        "energy_per_heavy_atom() docstring. Pass a negative value (e.g. "
                        "-1) to disable this filter.")

    args = p.parse_args()
    max_energy = args.max_energy_per_heavy if args.max_energy_per_heavy >= 0 else None

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    model, cond_dim = build_model(args.ckpt, device, args.use_ema)

    if args.condition is not None and cond_dim == 0:
        raise ValueError("--condition was given but checkpoint is unconditional (cond_dim=0).")
    if args.condition is None and cond_dim > 0:
        raise ValueError("Checkpoint is conditional; pass --condition.")
    if args.guidance_scale != 1.0 and args.condition == "null":
        raise ValueError("--guidance-scale only makes sense for positive/negative condition.")

    accepted = []
    stats = {
        "total_sampled": 0,
        "rdkit_invalid": 0,
        "multi_fragment": 0,
        "too_few_heavy_atoms": 0,
        "undersized_ring": 0,
        "oversized_ring": 0,
        "unparametrizable_energy": 0,
        "high_strain_energy": 0,
    }

    batch_idx = 0
    while len(accepted) < args.target_valid and batch_idx < args.max_batches:
        batch_idx += 1
        molecules = sample_batch(model, cond_dim, args, device, args.batch_size, seed_offset=batch_idx)

        for symbols, coords in molecules:
            stats["total_sampled"] += 1
            mol = mol_from_coords(symbols, coords)
            if mol is None:
                stats["rdkit_invalid"] += 1
                continue

            ok, reason = pass_filters(
                mol,
                require_single=args.require_single,
                min_heavy_atoms=args.min_heavy_atoms,
                min_ring_size=args.min_ring_size,
                max_ring_size=args.max_ring_size,
                max_energy_per_heavy=max_energy,
            )
            if not ok:
                stats[reason] += 1
                continue

            smiles = Chem.MolToSmiles(mol)
            accepted.append((symbols, coords, smiles, n_fragments(mol), mol.GetNumHeavyAtoms(), ring_sizes(mol)))
            if len(accepted) >= args.target_valid:
                break

        if batch_idx % 5 == 0 or len(accepted) >= args.target_valid:
            rate = len(accepted) / max(stats["total_sampled"], 1)
            print(
                f"batch {batch_idx:3d} | accepted {len(accepted):4d}/{args.target_valid} "
                f"| sampled {stats['total_sampled']:5d} | accept rate {rate:.1%}"
            )

    for i, (symbols, coords, smiles, n_frags, n_heavy, rings) in enumerate(accepted):
        write_xyz(
            os.path.join(args.out_dir, f"mol_{i:04d}.xyz"),
            symbols,
            coords,
            comment=f"accepted; smiles={smiles}; frags={n_frags}; heavy={n_heavy}; rings={rings}",
        )

    manifest_path = os.path.join(args.out_dir, args.manifest)
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "smiles", "n_fragments", "n_heavy_atoms", "ring_sizes"])
        for i, (_, _, smiles, n_frags, n_heavy, rings) in enumerate(accepted):
            w.writerow([f"mol_{i:04d}", smiles, n_frags, n_heavy, " ".join(str(x) for x in rings)])

    print("\nDone.")
    print(f"Saved {len(accepted)} accepted molecules to: {args.out_dir}")
    print(f"Manifest: {manifest_path}")

    sampled = max(stats["total_sampled"], 1)
    print("Sampling stats:")
    print(f"  total sampled:      {stats['total_sampled']}")
    print(f"  rdkit invalid:      {stats['rdkit_invalid']} ({stats['rdkit_invalid']/sampled:.1%})")
    print(f"  multi-fragment:     {stats['multi_fragment']} ({stats['multi_fragment']/sampled:.1%})")
    print(f"  too few heavy atom: {stats['too_few_heavy_atoms']} ({stats['too_few_heavy_atoms']/sampled:.1%})")
    print(f"  undersized ring:    {stats['undersized_ring']} ({stats['undersized_ring']/sampled:.1%})")
    print(f"  oversized ring:     {stats['oversized_ring']} ({stats['oversized_ring']/sampled:.1%})")
    print(f"  unparametrizable:   {stats['unparametrizable_energy']} ({stats['unparametrizable_energy']/sampled:.1%})")
    print(f"  high strain energy: {stats['high_strain_energy']} ({stats['high_strain_energy']/sampled:.1%})")
    print(f"  accepted:           {len(accepted)} ({len(accepted)/sampled:.1%})")

    if len(accepted) < args.target_valid:
        print("Warning: target not reached. Increase --max-batches or relax filters.")


if __name__ == "__main__":
    main()
