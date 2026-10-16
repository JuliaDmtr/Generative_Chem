"""
Recover bond orders from heavy-atom coordinates (no explicit hydrogens).

Needed because the model generates heavy atoms only, while anything downstream
— SMILES, pictures, strain filters, the curated candidate pipeline — needs a
real chemical structure. RDKit's DetermineBonds cannot do this: it assigns
orders by balancing valences against explicit hydrogens (measured here: 15%
produce anything, 0% correct). EDM's published distance lookup table also
fails at drug size (0% exact on real PARP conformers), because aromatic C-C
(~1.40 A) falls on the single-bond side of its threshold; the paper itself
reports this degradation for GEOM-Drugs.

What works instead: geometry says how many pi bonds each atom needs.
  - 4 neighbours, or 3 arranged pyramidally -> sp3, no pi bond
  - 3 arranged planar, or 2 at ~120 deg     -> sp2, exactly one pi bond
  - 2 at ~180 deg                           -> sp, two pi bonds
  - a flat 5- or 6-ring is aromatic, so its atoms need a pi bond (their
    ~108 deg internal angles would otherwise read as sp3)
Placing the double bonds is then a matching problem — every sp2 atom paired
with exactly one sp2 neighbour — solved exactly by networkx (blossom),
weighted by how much shorter each bond is than a typical single bond. RDKit
then perceives aromaticity, fills hydrogens and reads stereochemistry off the
3D coordinates.

Calibration on 500 real PARP conformers with hydrogens stripped
(scripts/calibrate_bond_orders.py):
    81.6% produce a valid molecule
    70.0% reproduce the original molecule exactly (canonical SMILES)
The connectivity underneath is exact (100%, see heavy_atom_eval.py), so the
~30% that differ are tautomer/resonance mistakes in fused aromatic systems,
not wrong skeletons. Treat the bond orders as a best-effort annotation and
the skeleton as the reliable part.

A note for future tuning: adding a bonus for ring-internal double bonds (to
discourage quinoid forms) was tried and collapsed accuracy to 2%, because it
overrides the bond-length signal that is doing the real work.
"""

import numpy as np
import networkx as nx
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")
from heavy_atom_eval import heavy_mol_from_coords

# Typical single-bond lengths (A), used only to score how "short" a bond is.
SINGLE_REF = {("C", "C"): 1.54, ("C", "N"): 1.47, ("C", "O"): 1.43, ("C", "S"): 1.82,
              ("N", "N"): 1.45, ("N", "O"): 1.40, ("O", "O"): 1.48, ("N", "S"): 1.69,
              ("O", "S"): 1.57, ("O", "P"): 1.63, ("C", "P"): 1.84, ("S", "S"): 2.04}
TRIPLE_MAX = {("C", "N"): 1.20, ("C", "C"): 1.24}


def pair(a, b):
    return tuple(sorted((a, b)))


# How far a 5- or 6-ring may deviate from flat (RMS out-of-plane spread, A) and
# still be treated as aromatic. Real PARP conformers have aromatic rings at
# ~0.006 A and genuinely saturated ones well above 0.05; generated geometry is
# less precise, so this value decides how much of the model's aromaticity
# survives reconstruction. Calibrated with
# scripts/calibrate_bond_orders.py --flatness-tol.
RING_FLATNESS_TOL = 0.05


def ring_thickness_of(coords, ring):
    """RMS out-of-plane spread of a ring, in angstrom (0 = perfectly flat)."""
    pts = np.asarray(coords, dtype=float)[list(ring)]
    centred = pts - pts.mean(axis=0)
    return float(np.linalg.svd(centred, compute_uv=False)[-1] / np.sqrt(len(list(ring))))


# Maximum number of bonds each element forms in neutral organic molecules.
# S and P are given their expanded valences (sulfone, phosphate); being
# permissive there is safer than being wrong about the common elements.
MAX_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 6, "P": 5, "F": 1, "Cl": 1,
               "Br": 1, "I": 1, "B": 3, "Si": 4}


def pi_capacity_limit(symbol: str, degree: int) -> int:
    """How many further bonds this atom can take, given how many it already has.

    Geometry says an atom *looks* sp2; this says whether it can actually
    accept the extra bond. An amide nitrogen is planar with three neighbours
    but is already at nitrogen's limit of three bonds, so it takes no double
    bond — and an ether oxygen likewise. Ignoring this produced "Explicit
    valence greater than permitted" on about half of all plausible generated
    skeletons (tests/test_bond_order_valence.py).
    """
    return max(MAX_VALENCE.get(symbol, 4) - degree, 0)


def geometry_pi_capacity(mol, coords):
    """How many pi bonds each atom's 3D shape demands: 0, 1 or 2."""
    cap = {}
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        d = len(nbrs)
        headroom = pi_capacity_limit(atom.GetSymbol(), d)
        if headroom == 0:          # no room for a pi bond, whatever the shape
            cap[i] = 0
            continue
        if d >= 4:
            cap[i] = 0
        elif d == 3:
            # planar (angles sum to ~360) means sp2; pyramidal (~328) means sp3
            angs = []
            for a in range(3):
                for b in range(a + 1, 3):
                    v1 = coords[nbrs[a]] - coords[i]
                    v2 = coords[nbrs[b]] - coords[i]
                    angs.append(np.degrees(np.arccos(np.clip(
                        v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))
            cap[i] = 1 if sum(angs) > 350.0 else 0
        elif d == 2:
            v1 = coords[nbrs[0]] - coords[i]
            v2 = coords[nbrs[1]] - coords[i]
            ang = np.degrees(np.arccos(np.clip(
                v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))
            cap[i] = 2 if ang > 155 else (1 if ang > 115 else 0)
        else:
            # terminal atom: short bond relative to a single bond means pi
            j = nbrs[0] if nbrs else None
            if j is None:
                cap[i] = 0
            else:
                ref = SINGLE_REF.get(pair(atom.GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()))
                dist = np.linalg.norm(coords[i] - coords[j])
                cap[i] = 1 if (ref and dist < ref - 0.08) else 0
        cap[i] = min(cap[i], headroom)
    return cap


def assign_bond_orders(symbols, coords):
    """Heavy-atom coords -> sanitized RDKit Mol (implicit H), or None."""
    coords = np.asarray(coords, dtype=float)
    if len(symbols) == 0:
        return None          # "no atoms" is not a molecule; say so explicitly
    mol = heavy_mol_from_coords(symbols, coords)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    cap = geometry_pi_capacity(mol, coords)

    # A 5-ring's internal angles are ~108 deg, which the angle rule above
    # reads as sp3 -- so aromatic 5-rings (imidazole, furan, ...) would come
    # out saturated. Ring planarity is the reliable signal instead: a flat
    # 5- or 6-ring is aromatic, a puckered one is not.
    for ring in Chem.GetSymmSSSR(mol):
        ring = list(ring)
        if not 5 <= len(ring) <= 6:
            continue
        # smallest singular value = RMS thickness of the ring
        if ring_thickness_of(coords, ring) < RING_FLATNESS_TOL:
            for i in ring:
                atom = mol.GetAtomWithIdx(i)
                if pi_capacity_limit(atom.GetSymbol(), atom.GetDegree()) >= 1:
                    cap[i] = max(cap.get(i, 0), 1)

    # 1. unambiguous triple bonds, by distance; their atoms leave the matching
    used = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        p = pair(symbols[i], symbols[j])
        if p in TRIPLE_MAX and np.linalg.norm(coords[i] - coords[j]) < TRIPLE_MAX[p]:
            rw.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.TRIPLE)
            used.update((i, j))

    # 2. one double bond per sp2 atom, chosen by maximum-weight matching
    g = nx.Graph()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in used or j in used or cap.get(i, 0) < 1 or cap.get(j, 0) < 1:
            continue
        ref = SINGLE_REF.get(pair(symbols[i], symbols[j]), 1.5)
        shortening = ref - np.linalg.norm(coords[i] - coords[j])
        g.add_edge(i, j, weight=max(shortening, 0.001))
    for i, j in nx.max_weight_matching(g, maxcardinality=True):
        rw.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.DOUBLE)

    # let RDKit fill every remaining valence with implicit hydrogens: the
    # connectivity perception marks atoms "no implicit H", which would leave
    # them as radicals ([c], [C]) instead of ordinary atoms.
    for atom in rw.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetFormalCharge(0)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        # stereocentres and double-bond geometry are in the 3D coordinates;
        # without this they are simply dropped
        Chem.AssignStereochemistryFrom3D(out)
    except Exception:
        return None
    return out


def smiles_from_heavy_coords(symbols, coords):
    """Canonical SMILES for a heavy-atom molecule, or None if reconstruction fails."""
    mol = assign_bond_orders(symbols, coords)
    return None if mol is None else Chem.MolToSmiles(mol)
