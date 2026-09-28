"""
Diagnostic: check whether generated .xyz molecules have realistic
interatomic distances, or are systematically compressed (the leading
hypothesis for the "mostly 3-membered rings" / ~10% validity symptom —
see diffusion.py's q_sample/sample: x0 is diffused with unit-variance
noise and no coordinate rescaling, unlike the reference EDM
implementation's normalize_factors).

For each .xyz file, computes the nearest-neighbor distance for every
atom (the distance to its closest other atom in the same molecule), then
reports the distribution across all generated molecules. Deliberately
does NOT use RDKit bond perception (evaluate.py's xyz_to_mol) — this
checks raw geometry directly, so it still gives a signal even on
molecules that fail bond perception entirely, which is exactly the
regime we're trying to diagnose.

Reference point: typical single-bond lengths are ~0.9-1.6 Angstrom
(H-H shortest at 0.74, C-C 1.54, C-Cl 1.77, etc. — see the EDM paper's
Appendix C bond-distance tables). A real molecule's nearest-neighbor
distance per atom should mostly fall in roughly 0.9-1.8 Angstrom. If the
generated distribution sits well below that (e.g. clustered under 0.5-0.8
Angstrom), that's direct evidence of the compression hypothesis, not a
training-quantity problem.

Usage:
    python diagnose_scale.py --xyz-dir samples/
    python diagnose_scale.py --xyz-dir samples/ --compare-npz parp_3d.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def read_xyz(path: str):
    """Minimal .xyz reader: returns (symbols, coords) as (list[str], (N,3)
    float array). Does not require RDKit — just parses the standard XYZ
    format (atom count line, comment line, then one 'symbol x y z' line
    per atom) directly, so this works even on geometries RDKit's
    MolFromXYZFile/DetermineBonds would reject outright.
    """
    with open(path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    symbols = []
    coords = np.zeros((n_atoms, 3), dtype=float)
    for i in range(n_atoms):
        parts = lines[2 + i].split()
        symbols.append(parts[0])
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return symbols, coords


def nearest_neighbor_distances(coords: np.ndarray) -> np.ndarray:
    """For each atom, the distance to its closest other atom in the same
    molecule. Returns an (N,) array, one value per atom. Molecules with
    fewer than 2 atoms are skipped by the caller (nothing to measure).
    """
    n = coords.shape[0]
    if n < 2:
        return np.array([])
    diffs = coords[:, None, :] - coords[None, :, :]        # (N, N, 3)
    dists = np.sqrt((diffs ** 2).sum(-1))                    # (N, N)
    np.fill_diagonal(dists, np.inf)                          # exclude self
    return dists.min(axis=1)                                 # (N,)


def summarize(all_nn_dists: np.ndarray, label: str):
    if len(all_nn_dists) == 0:
        print(f"{label}: no atoms measured.")
        return
    pctiles = np.percentile(all_nn_dists, [5, 25, 50, 75, 95])
    frac_below_0_8 = (all_nn_dists < 0.8).mean()
    frac_below_1_0 = (all_nn_dists < 1.0).mean()
    print(f"{label} — n_atoms={len(all_nn_dists):,}")
    print(f"   mean={all_nn_dists.mean():.3f} Å   std={all_nn_dists.std():.3f} Å")
    print(f"   percentiles (5/25/50/75/95): "
          f"{pctiles[0]:.3f} / {pctiles[1]:.3f} / {pctiles[2]:.3f} / {pctiles[3]:.3f} / {pctiles[4]:.3f} Å")
    print(f"   fraction < 0.8 Å: {frac_below_0_8:.1%}   fraction < 1.0 Å: {frac_below_1_0:.1%}")
    print(f"   (for reference: real single bonds are ~0.9-1.8 Å; "
          f"H-H is shortest at ~0.74 Å, so >20-30% of atoms under 0.8 Å "
          f"across a whole batch is a red flag, not just a few short H bonds)")


def bond_angles(coords: np.ndarray, cutoff: float = 1.8) -> np.ndarray:
    """For each atom, the angle (in degrees) at that atom between every
    pair of its neighbors within `cutoff` Angstroms of it. Returns a flat
    array of all such angles across the molecule (one molecule can
    contribute many angles, e.g. an atom with 3 close neighbors
    contributes 3 angles).

    Deliberately uses a fixed geometric distance cutoff rather than
    RDKit's DetermineBonds/covFactor — a covFactor sweep already showed
    the "3-ring" pattern is invariant to the bond-perception threshold
    (present even at covFactor=1.0, the tightest cutoff that still yields
    any valid molecules), so this diagnostic stays independent of that
    axis and looks directly at the raw geometry.

    A real cyclopropane-style 3-membered ring implies a ~60° angle at
    each of its 3 atoms; correct sp3/sp2 local geometry implies angles
    clustered around ~109.5°/~120°. If generated geometries have a
    spike in the 50-70° band that ground-truth conformers lack, that's
    direct evidence the model's local *angular* structure — not just
    pairwise bond length — is wrong.
    """
    n = coords.shape[0]
    if n < 3:
        return np.array([])
    diffs = coords[:, None, :] - coords[None, :, :]          # (N, N, 3)
    dists = np.sqrt((diffs ** 2).sum(-1))                      # (N, N)
    np.fill_diagonal(dists, np.inf)

    angles = []
    for i in range(n):
        neighbors = np.where(dists[i] < cutoff)[0]
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                j, k = neighbors[a], neighbors[b]
                v1 = coords[j] - coords[i]
                v2 = coords[k] - coords[i]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                angles.append(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    return np.array(angles)


def summarize_angles(all_angles: np.ndarray, label: str, band=(50.0, 70.0)):
    if len(all_angles) == 0:
        print(f"{label}: no angles measured.")
        return
    pctiles = np.percentile(all_angles, [5, 25, 50, 75, 95])
    frac_in_band = ((all_angles >= band[0]) & (all_angles <= band[1])).mean()
    print(f"{label} — n_angles={len(all_angles):,}")
    print(f"   mean={all_angles.mean():.1f}°   std={all_angles.std():.1f}°")
    print(f"   percentiles (5/25/50/75/95): "
          f"{pctiles[0]:.1f} / {pctiles[1]:.1f} / {pctiles[2]:.1f} / {pctiles[3]:.1f} / {pctiles[4]:.1f} °")
    print(f"   fraction in {band[0]:.0f}-{band[1]:.0f}° band (equilateral-triangle-like): {frac_in_band:.1%}")
    print(f"   (for reference: real sp3/sp2 bond angles cluster around ~109.5°/~120°; "
          f"a 3-membered ring forces ~60° at each vertex, so a spike in the "
          f"{band[0]:.0f}-{band[1]:.0f}° band beyond what ground truth shows is direct "
          f"evidence of wrong local angular structure, not just wrong bond length)")


def run_on_xyz_dir(xyz_dir: str, angle_cutoff: float = 1.8):
    xyz_files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))
    if not xyz_files:
        print(f"⚠️  No .xyz files found in {xyz_dir}", file=sys.stderr)
        return None, None

    all_dists = []
    all_angles = []
    per_molecule_means = []
    for path in xyz_files:
        symbols, coords = read_xyz(path)
        nn = nearest_neighbor_distances(coords)
        if len(nn) == 0:
            continue
        all_dists.append(nn)
        per_molecule_means.append(nn.mean())
        ang = bond_angles(coords, cutoff=angle_cutoff)
        if len(ang) > 0:
            all_angles.append(ang)

    if not all_dists:
        print("⚠️  No usable molecules (all had < 2 atoms).", file=sys.stderr)
        return None, None

    all_dists = np.concatenate(all_dists)
    all_angles = np.concatenate(all_angles) if all_angles else np.array([])
    print(f"\n=== Generated molecules ({len(xyz_files)} files) ===")
    summarize(all_dists, "generated")
    print(f"   per-molecule mean nearest-neighbor distance — "
          f"min={min(per_molecule_means):.3f}, max={max(per_molecule_means):.3f} Å "
          f"(large spread here suggests inconsistent scale across samples, "
          f"not just a uniform shift)")
    summarize_angles(all_angles, "generated (angle_cutoff=%.1f Å)" % angle_cutoff)
    return all_dists, all_angles


def run_on_npz(npz_path: str, n_sample: int = 200, seed: int = 0, angle_cutoff: float = 1.8):
    """Same measurement on real training-set conformers (e.g. parp_3d.npz
    or chembl_3d.npz), as a ground-truth comparison point measured with
    the exact same code path — avoids any ambiguity about what "normal"
    looks like for this dataset/units.
    """
    data = np.load(npz_path, allow_pickle=True)
    positions = data["positions"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(positions), size=min(n_sample, len(positions)), replace=False)

    all_dists = []
    all_angles = []
    for i in idx:
        coords = np.asarray(positions[i], dtype=float)
        nn = nearest_neighbor_distances(coords)
        if len(nn) > 0:
            all_dists.append(nn)
        ang = bond_angles(coords, cutoff=angle_cutoff)
        if len(ang) > 0:
            all_angles.append(ang)

    all_dists = np.concatenate(all_dists)
    all_angles = np.concatenate(all_angles) if all_angles else np.array([])
    print(f"\n=== Ground-truth comparison ({npz_path}, {len(idx)} molecules sampled) ===")
    summarize(all_dists, "ground truth")
    summarize_angles(all_angles, "ground truth (angle_cutoff=%.1f Å)" % angle_cutoff)
    return all_dists, all_angles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz-dir", default="samples",
                    help="Directory of .xyz files written by generate.py / generate_curated.py")
    p.add_argument("--compare-npz", default=None,
                    help="Optional: a training-set .npz (e.g. parp_3d.npz) to measure the same "
                         "statistic on, as a ground-truth reference point.")
    p.add_argument("--compare-n-sample", type=int, default=200)
    p.add_argument("--angle-cutoff", type=float, default=1.8,
                    help="Fixed distance (Å) within which two atoms count as 'neighbors' of "
                         "a third for the bond-angle diagnostic. Deliberately independent of "
                         "RDKit's DetermineBonds/covFactor — see bond_angles() docstring.")
    p.add_argument("--angle-band", type=str, default="50,70",
                    help="Comma-separated low,high degrees defining the "
                         "'equilateral-triangle-like' band to report a fraction for.")
    args = p.parse_args()

    band = tuple(float(v) for v in args.angle_band.split(","))
    generated, generated_angles = run_on_xyz_dir(args.xyz_dir, angle_cutoff=args.angle_cutoff)

    if args.compare_npz:
        ground_truth, ground_truth_angles = run_on_npz(
            args.compare_npz, n_sample=args.compare_n_sample, angle_cutoff=args.angle_cutoff)

        if generated is not None and ground_truth is not None:
            ratio = generated.mean() / ground_truth.mean()
            print(f"\n=== Verdict ===")
            print(f"generated/ground-truth mean nearest-neighbor distance ratio: {ratio:.2f}x")
            if ratio < 0.7:
                print("   -> Generated geometry is substantially COMPRESSED relative to real "
                      "molecules. This supports the coordinate-scale-mismatch hypothesis: "
                      "diffusion.py's q_sample/sample use unit-variance noise with no "
                      "coordinate rescaling, while real molecular coordinate variance "
                      "(especially for drug-sized PARP inhibitors) is likely well above 1. "
                      "Priority: add x_scale/h_scale normalization (and its inverse in "
                      "sample()) before investing in more training epochs.")
            elif ratio > 1.3:
                print("   -> Generated geometry is substantially EXPANDED relative to real "
                      "molecules — less common, but check the same normalization code path; "
                      "an inverted or double-applied scale factor could cause this.")
            else:
                print("   -> Generated and ground-truth scales are roughly comparable. "
                      "The compression hypothesis is likely NOT the main driver of low "
                      "validity here — look at undertraining, model capacity, or the "
                      "atom-type (h) channel instead.")

        if len(generated_angles) and len(ground_truth_angles):
            gen_frac = ((generated_angles >= band[0]) & (generated_angles <= band[1])).mean()
            gt_frac = ((ground_truth_angles >= band[0]) & (ground_truth_angles <= band[1])).mean()
            print(f"\n=== Angle verdict ===")
            print(f"fraction of angles in {band[0]:.0f}-{band[1]:.0f}° band: "
                  f"generated={gen_frac:.1%}  ground-truth={gt_frac:.1%}  "
                  f"(ratio {gen_frac / max(gt_frac, 1e-9):.1f}x)")
            if gen_frac > gt_frac * 2 and gen_frac > 0.05:
                print("   -> Generated geometry has a clear excess of near-60° angles "
                      "relative to ground truth. This is direct evidence of a genuine "
                      "angular/3-body geometry defect (not just a bond-perception "
                      "artifact) — the model places atoms in mutually close, "
                      "near-equilateral-triangle arrangements it doesn't produce for "
                      "real training-set conformers. This justifies investing in "
                      "angle-aware training/architecture changes.")
            else:
                print("   -> Generated angle distribution is not dramatically more "
                      "triangle-like than ground truth. The 3-ring pattern seen after "
                      "bond perception may be driven more by which atoms happen to be "
                      "close (density/packing) than by a systematic ~60° angle defect — "
                      "re-check with a tighter/looser --angle-cutoff before concluding.")


if __name__ == "__main__":
    main()