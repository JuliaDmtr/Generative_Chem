"""
For each known PARP inhibitor, find the most similar generated candidate.

Produces a slide-ready table: reference drug, best-matching candidate, and
their Tanimoto similarity, with both structures drawn side by side.

Similarity is Tanimoto over Morgan fingerprints (radius 2, 2048 bits), the
usual default for this comparison. As a rule of thumb for interpretation:
~0.3 means a shared scaffold or a few shared motifs, ~0.5 a clear structural
relationship, and >0.7 a close analogue. The same candidate may be the best
match for more than one reference; nothing is reserved.

This measures structural similarity only. It does not predict that any
candidate inhibits PARP.

Usage (from src/molecule-diffusion):
    python scripts/match_reference_inhibitors.py \
        --references ../../Datasets/PARP_inhibitors_July26/curated_parp_inhibitors.csv \
        --candidates candidates_v10/candidates.csv \
        --out-dir candidates_v10/reference_matches
"""

from __future__ import annotations

import argparse
import csv
import os

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")


def load_molecules(path, smiles_column, name_column=None):
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            smiles = row.get(smiles_column)
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  ! could not parse {smiles[:50]} — skipped")
                continue
            out.append({"name": row.get(name_column, "") if name_column else "",
                        "smiles": Chem.MolToSmiles(mol), "mol": mol, "row": row})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", default="../../Datasets/PARP_inhibitors_July26/curated_parp_inhibitors.csv")
    p.add_argument("--reference-smiles-column", default="smiles")
    p.add_argument("--reference-name-column", default="name")
    p.add_argument("--candidates", default="candidates_v10/candidates.csv")
    p.add_argument("--candidate-smiles-column", default="smiles")
    p.add_argument("--out-dir", default="candidates_v10/reference_matches")
    p.add_argument("--per-image", type=int, default=5,
                   help="references per image file; the table is split into chunks of this size "
                        "so each one stays readable on a slide (0 = one single image)")
    p.add_argument("--cell-size", type=int, default=700, help="pixels per structure cell")
    p.add_argument("--legend-font-size", type=int, default=34)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--fp-size", type=int, default=2048)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    refs = load_molecules(args.references, args.reference_smiles_column, args.reference_name_column)
    cands = load_molecules(args.candidates, args.candidate_smiles_column)
    if not refs or not cands:
        raise SystemExit("Need at least one reference and one candidate.")
    print(f"{len(refs)} reference inhibitors, {len(cands)} candidates")

    gen = AllChem.GetMorganGenerator(radius=args.radius, fpSize=args.fp_size)
    cand_fps = [gen.GetFingerprint(c["mol"]) for c in cands]

    rows = []
    for ref in refs:
        sims = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(ref["mol"]), cand_fps)
        best = max(range(len(sims)), key=lambda i: sims[i])
        rows.append({
            "reference": ref["name"] or ref["smiles"][:30],
            "reference_targets": ref["row"].get("parps", ""),
            "reference_smiles": ref["smiles"],
            "best_candidate_smiles": cands[best]["smiles"],
            "tanimoto_similarity": round(float(sims[best]), 3),
            "_ref_mol": ref["mol"],
            "_cand_mol": cands[best]["mol"],
        })
    rows.sort(key=lambda r: -r["tanimoto_similarity"])

    csv_path = os.path.join(args.out_dir, "reference_matches.csv")
    with open(csv_path, "w", newline="") as f:
        fields = [k for k in rows[0] if not k.startswith("_")]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in rows])

    def draw(subset, path):
        """One row per reference: the drug on the left, its best candidate on the right."""
        mols, legends = [], []
        for r in subset:
            mols += [r["_ref_mol"], r["_cand_mol"]]
            legends += [f"{r['reference']} (reference)",
                        f"best candidate - Tanimoto {r['tanimoto_similarity']}"]
        opts = Draw.rdMolDraw2D.MolDrawOptions()
        opts.legendFontSize = args.legend_font_size
        opts.legendFraction = 0.12
        Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(args.cell_size, int(args.cell_size * 0.8)),
                             legends=legends, drawOptions=opts, useSVG=False).save(path)
        return path

    written = []
    chunk = args.per_image if args.per_image > 0 else len(rows)
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        suffix = "" if chunk >= len(rows) else f"_part{start // chunk + 1}"
        written.append(draw(part, os.path.join(args.out_dir, f"reference_matches{suffix}.png")))

    # and one image per pair, for putting a single comparison on its own slide
    pair_dir = os.path.join(args.out_dir, "pairs")
    os.makedirs(pair_dir, exist_ok=True)
    for r in rows:
        name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in r["reference"])
        draw([r], os.path.join(pair_dir, f"{name}.png"))

    print(f"\n{'reference':14s} {'targets':22s} {'Tanimoto':>9s}")
    print("-" * 48)
    for r in rows:
        print(f"{r['reference']:14s} {r['reference_targets'][:22]:22s} {r['tanimoto_similarity']:9.3f}")
    print(f"\ntable: {csv_path}")
    for path in written:
        print(f"image: {path}")
    print(f"per-pair images: {pair_dir}/")


if __name__ == "__main__":
    main()
