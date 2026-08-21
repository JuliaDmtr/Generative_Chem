"""
Convert generated .xyz molecules (from generate.py) back to SMILES via
RDKit bond perception, and render 2D depiction images.

Pipeline per .xyz file: read coords+symbols -> build an RDKit RWMol with
those 3D positions -> DetermineBonds (distance-based bond perception,
using covalent radii) -> SanitizeMol -> MolToSmiles. Molecules that fail
bond perception or sanitization (common for diffusion-generated
coordinates — this is expected, not a bug) are counted and skipped
rather than crashing the run.

Usage:
    python evaluate.py --xyz-dir samples/ --out-dir renders/

Needs rdkit (pip install rdkit) in addition to this skill's requirements.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.rdDetermineBonds import DetermineBonds
    RDLogger.DisableLog("rdApp.*")  # silence RDKit's routine sanitize warnings
except ImportError:
    print("This script needs rdkit: pip install rdkit", file=sys.stderr)
    raise


def mol_from_coords(symbols, coords, charge: int = 0):
    """Build a sanitized RDKit Mol directly from atom symbols + (N, 3)
    coordinates (Angstroms), with no file I/O. Returns None if bond
    perception / sanitization fails. This is the in-memory counterpart of
    xyz_to_mol(), used for fast validity checks during training
    diagnostics (see train_conditional.py) where writing a .xyz file per
    molecule would be wasteful.
    """
    import numpy as np
    from rdkit.Geometry import Point3D

    coords = np.asarray(coords, dtype=float)
    raw = Chem.RWMol()
    conf = Chem.Conformer(len(symbols))
    for i, sym in enumerate(symbols):
        raw.AddAtom(Chem.Atom(sym))
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    raw.AddConformer(conf)
    mol = raw.GetMol()

    try:
        DetermineBonds(mol, charge=charge)
    except Exception:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return mol


def n_fragments(mol) -> int:
    """Number of disconnected fragments in a sanitized Mol. 1 means a
    single connected molecule; >1 means bond perception produced two or
    more spatially disjoint pieces from one requested atom cluster —
    e.g. SMILES containing '.', like a salt. RDKit's SanitizeMol accepts
    this (it only checks valence/aromaticity within existing bonds), so
    these otherwise count as "valid" even though they aren't one coherent
    generated molecule. Use this to report a stricter single-molecule
    rate alongside the plain RDKit-validity rate.
    """
    return len(Chem.GetMolFrags(mol, sanitizeFrags=False))


def xyz_to_mol(xyz_path: str, charge: int = 0):
    """Read one .xyz file and return a sanitized RDKit Mol, or None if
    bond perception / sanitization fails.
    """
    raw = Chem.MolFromXYZFile(xyz_path)
    if raw is None:
        return None

    mol = Chem.Mol(raw)
    try:
        DetermineBonds(mol, charge=charge)
    except Exception:
        return None  # distance-based bond perception failed — e.g. atoms
                      # too close/far for any plausible bond graph

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None  # valence/aromaticity rules violated — the geometry
                      # doesn't correspond to a chemically valid molecule

    return mol


def convert_dir(xyz_dir: str, out_dir: str, charge: int = 0, image_size: int = 400):
    """Convert every .xyz in xyz_dir to SMILES + a 2D depiction PNG.
    Returns (results, n_failed) where results is a list of
    (filename, smiles) for molecules that converted successfully.
    """
    os.makedirs(out_dir, exist_ok=True)
    xyz_files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))
    if not xyz_files:
        print(f"⚠️  No .xyz files found in {xyz_dir}", file=sys.stderr)

    results = []
    n_failed = 0
    n_multi_fragment = 0

    for path in xyz_files:
        name = os.path.splitext(os.path.basename(path))[0]
        mol = xyz_to_mol(path, charge=charge)
        if mol is None:
            n_failed += 1
            continue

        smiles = Chem.MolToSmiles(mol)
        n_frags = n_fragments(mol)
        if n_frags > 1:
            n_multi_fragment += 1
        results.append((name, smiles, n_frags))

        # 2D depiction: recompute 2D coords for a clean layout rather than
        # projecting the noisy generated 3D geometry.
        mol_2d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_2d)
        Draw.MolToFile(mol_2d, os.path.join(out_dir, f"{name}.png"),
                        size=(image_size, image_size))

    n_total = len(xyz_files)
    validity = len(results) / n_total if n_total else 0.0
    n_single = len(results) - n_multi_fragment
    single_molecule_rate = n_single / n_total if n_total else 0.0
    print(f"✅ Converted {len(results):,}/{n_total:,} molecules to SMILES "
          f"(RDKit-validity rate: {validity:.1%}).")
    print(f"   Of those, {n_single:,}/{n_total:,} are a single connected molecule "
          f"(single-molecule rate: {single_molecule_rate:.1%}); "
          f"{n_multi_fragment:,} are 2+ disconnected fragments (e.g. SMILES with '.') "
          f"— the atoms didn't form one coherent bonded structure, even though "
          f"they passed sanitization.")
    if n_failed:
        print(f"   Skipped {n_failed:,} that failed bond perception/sanitization "
              f"— this is expected for a diffusion model and reflects real model "
              f"quality, not a bug in this script.")

    # Write a manifest alongside the images
    manifest_path = os.path.join(out_dir, "smiles.txt")
    with open(manifest_path, "w") as f:
        for name, smi, n_frags in results:
            f.write(f"{name}\t{smi}\t{n_frags}\n")
    print(f"💾 Wrote {len(results):,} SMILES to {manifest_path} (name, smiles, n_fragments)")
    print(f"💾 Wrote {len(results):,} PNG depictions to {out_dir}/")

    return results, n_failed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz-dir", default="samples",
                    help="Directory of .xyz files written by generate.py")
    p.add_argument("--out-dir", default="renders",
                    help="Where to write smiles.txt and per-molecule PNGs")
    p.add_argument("--charge", type=int, default=0,
                    help="Net molecular charge assumed for bond perception "
                         "(DetermineBonds needs this; 0 is right for neutral "
                         "molecules, which is what this model was trained on "
                         "— it doesn't diffuse formal charges, see approach.md)")
    p.add_argument("--image-size", type=int, default=400)
    args = p.parse_args()

    convert_dir(args.xyz_dir, args.out_dir, charge=args.charge, image_size=args.image_size)


if __name__ == "__main__":
    main()
