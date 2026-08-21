"""
Convert ChEMBL SMILES (via ChemblProcessor) into 3D conformers, in the
exact .npz format QM9Dataset expects (see dataset.py):

    positions:  object array of (n_i, 3) float32   [Angstroms]
    atom_types: object array of (n_i,)  int64       [indices into ATOM_VOCAB]

Pipeline per SMILES: parse -> add explicit H's -> ETKDG embed a 3D
conformer -> MMFF relax -> record symbols + coords. Molecules that fail
to parse, fail to embed, or contain an element outside ATOM_VOCAB are
skipped and counted.

Usage:
    python prepare_chembl.py --num-samples 20000 --max-len 100 \
        --data-path-prefix ../../Datasets/ --out chembl_3d.npz

Needs rdkit (pip install rdkit) in addition to this skill's requirements.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from utils import ATOM2IDX

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")  # silence RDKit's routine parse warnings
except ImportError:
    print("This script needs rdkit: pip install rdkit", file=sys.stderr)
    raise


def smiles_to_3d(smiles: str, seed: int = 0):
    """SMILES -> (atom_symbols, xyz_coords) via ETKDG embedding + MMFF relax.
    Returns None if parsing or embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)  # 3D embedding needs explicit H's to be realistic

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None  # embedding failed (common for large/strained/flexible SMILES)

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # relaxation failing isn't fatal — keep the raw embedded conformer

    conf = mol.GetConformer()
    coords = conf.GetPositions()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return symbols, coords


def convert(smiles_list: list[str], seed: int = 0, verbose_every: int = 1000):
    """Convert a list of raw SMILES into positions/atom_types arrays,
    skipping anything that fails to parse/embed or uses an out-of-vocab
    element. Prints a running tally so failures aren't silent.
    """
    positions, atom_types = [], []
    n_embed_fail = 0
    n_vocab_fail = 0

    for i, smi in enumerate(smiles_list):
        result = smiles_to_3d(smi, seed=seed)
        if result is None:
            n_embed_fail += 1
            continue
        symbols, coords = result
        try:
            idx = np.array([ATOM2IDX[s] for s in symbols], dtype=np.int64)
        except KeyError:
            n_vocab_fail += 1
            continue

        positions.append(coords.astype(np.float32))
        atom_types.append(idx)

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"   ... processed {i + 1:,}/{len(smiles_list):,} "
                  f"(kept {len(positions):,}, embed-fail {n_embed_fail:,}, "
                  f"vocab-fail {n_vocab_fail:,})")

    print(f"✅ Converted {len(positions):,}/{len(smiles_list):,} molecules to 3D.")
    print(f"   Skipped: {n_embed_fail:,} parse/embed failures, "
          f"{n_vocab_fail:,} out-of-vocab elements "
          f"(ATOM_VOCAB={sorted(ATOM2IDX.keys())}).")
    if n_vocab_fail > 0.05 * len(smiles_list):
        print("   ⚠️  A large fraction failed on vocabulary — consider extending "
              "ATOM_VOCAB in utils.py to cover ChEMBL's element set (S, Cl, Br, "
              "P, ...) and retraining, rather than silently dropping this much data.")

    return positions, atom_types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path-prefix", default="../../Datasets/",
                    help="Passed through to ChemblProcessor (needs "
                         "chembl_canonical_cache.txt at this prefix).")
    p.add_argument("--num-samples", type=int, default=20000)
    p.add_argument("--max-len", type=int, default=100,
                    help="Max SMILES string length passed to make_samples.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="chembl_3d.npz")
    args = p.parse_args()

    # Imported here, not at module top, so this script still runs (and its
    # --help works) even if chembl_processor.py isn't on the path.
    from chembl_processor import ChemblProcessor

    proc = ChemblProcessor(data_path_prefix=args.data_path_prefix)
    print(f"Loading up to {args.num_samples:,} SMILES (max_len={args.max_len})...")
    # NOTE: use make_samples' raw SMILES output directly — NOT
    # prepare_data_for_lstm's output, which wraps strings with the
    # '$'/'E' start/end tokens meant for the char-LSTM vocabulary, not
    # for RDKit parsing.
    samples = proc.make_samples(args.num_samples, args.max_len)

    positions, atom_types = convert(samples, seed=args.seed)

    if not positions:
        print("❌ No molecules converted successfully — nothing to save.", file=sys.stderr)
        sys.exit(1)

    np.savez(
        args.out,
        positions=np.array(positions, dtype=object),
        atom_types=np.array(atom_types, dtype=object),
    )
    print(f"💾 Saved {len(positions):,} molecules to {args.out}")
    print(f"   Train with: python train.py --data-npz {args.out} ...")


if __name__ == "__main__":
    main()
