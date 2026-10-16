"""
Quality checks for heavy-atom-only molecules (no explicit hydrogens).

The with-hydrogens pipeline (evaluate.mol_from_coords -> RDKit DetermineBonds)
cannot be used here: DetermineBonds assigns bond orders by balancing valences
against the explicit hydrogens, which no longer exist. Instead we perceive the
heavy-atom *bond graph* from interatomic distances and check it.

Calibration (scripts/calibrate_heavy_eval.py): on real PARP conformers with
the hydrogens stripped, DetermineConnectivity recovers the exact heavy-heavy
bond set of the with-H reference for 100/100 molecules tested, so the
perception step is sound and a failed check means a bad molecule, not a bad
measurement.

What is checked, in increasing strictness:
  connected     — the bond graph is one connected component
  valence_ok    — no heavy atom has more neighbours than its element allows
  plausible     — connected AND valence_ok (the headline number for noH runs)

Bond *orders* are deliberately not assigned: from heavy atoms alone a single,
double or aromatic C-C differ only by ~0.1-0.2 Å, so inferred orders would
measure the inference, not the model. Ring sizes and bond angles are reported
separately by the caller (see evaluate_checkpoint.py).
"""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

# Maximum number of neighbours a heavy atom of each element can have. A heavy
# atom's degree exceeding this is impossible in a neutral organic molecule,
# whatever the bond orders are (hydrogens only add to the count).
MAX_DEGREE = {"C": 4, "N": 4, "O": 2, "F": 1, "S": 6, "Cl": 1, "Br": 1, "I": 1,
              "P": 5, "B": 3, "Si": 4, "H": 1}


def heavy_mol_from_coords(symbols, coords):
    """Build an RDKit Mol with a perceived heavy-atom bond graph (no bond
    orders, no sanitization). Returns None if perception fails."""
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(symbols))
    for i, sym in enumerate(symbols):
        rw.AddAtom(Chem.Atom(sym))
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol = rw.GetMol()
    mol.AddConformer(conf)
    try:
        rdDetermineBonds.DetermineConnectivity(mol, charge=0)
    except Exception:
        return None
    return mol


def heavy_atom_checks(symbols, coords) -> dict:
    """Run the heavy-atom checks on one molecule. Keys: perceived, connected,
    valence_ok, plausible, n_fragments, bad_valence_atoms, ring_sizes."""
    out = {"perceived": False, "connected": False, "valence_ok": False,
           "plausible": False, "n_fragments": 0, "bad_valence_atoms": [], "ring_sizes": []}
    mol = heavy_mol_from_coords(symbols, coords)
    if mol is None:
        return out
    out["perceived"] = True
    out["n_fragments"] = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
    out["connected"] = out["n_fragments"] == 1
    bad = [a.GetSymbol() for a in mol.GetAtoms()
           if a.GetDegree() > MAX_DEGREE.get(a.GetSymbol(), 4)]
    out["bad_valence_atoms"] = bad
    out["valence_ok"] = not bad
    out["plausible"] = out["connected"] and out["valence_ok"]
    try:
        out["ring_sizes"] = [len(r) for r in Chem.GetSymmSSSR(mol)]
    except Exception:
        pass
    return out


def heavy_bond_set(mol) -> set[frozenset[int]]:
    """The perceived bond graph as a set of index pairs (for calibration)."""
    return {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}


def skeleton_smiles(symbols, coords):
    """Canonical SMILES of the perceived heavy-atom skeleton, with every bond
    forced to single and all aromaticity cleared. Returns None if perception
    or sanitization fails.

    This is how two heavy-atom molecules are compared for being "the same
    molecule": from heavy atoms alone, bond orders are not reliably
    recoverable (single/double/aromatic C-C differ by only ~0.1-0.2 A), so
    the comparison is over the bond graph, not the bond orders. Both sides of
    any comparison must go through this same function.
    """
    mol = heavy_mol_from_coords(symbols, coords)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)      # heavy atoms only: no implicit H's invented
        atom.SetNumExplicitHs(0)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return Chem.MolToSmiles(out)
