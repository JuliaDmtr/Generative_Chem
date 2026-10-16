"""
Build parp_gallery/index.html: the generated candidates that most resemble real
PARP inhibitors, each shown next to the inhibitor it resembles.

Two similarity numbers are computed per candidate, both Tanimoto over Morgan
fingerprints (radius 2, 2048 bits):

  nearest inhibitor  the closest of the ~5,800 real PARP inhibitors the model
                     was trained on (ChEMBL). Its structure is drawn alongside.
  nearest named drug the closest of the ten curated reference drugs
                     (olaparib, niraparib, talazoparib, ...), named on the card.

Candidates are ranked by the first and the top --top are shown. Synthetic
accessibility (Ertl & Schuffenhauer, 1 = easy, 10 = hard) comes from RDKit's
SA_Score contrib; strain is the unrelaxed MMFF energy per heavy atom carried
over from the candidate table.

The page is committed on purpose: .gitignore keeps *.png out of the repo except
under parp_gallery/, so this renders for anyone who clones.

Nothing here predicts activity. A high Tanimoto means the molecule looks like a
known inhibitor, not that it inhibits anything.

Usage (from src/molecule-diffusion):
    python scripts/make_gallery.py --candidates candidates_v11/candidates.csv \
        --checkpoint v11_noH_epoch_499 --top 24
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import sys

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, RDConfig, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

from bond_orders import assign_bond_orders
from dataset import QM9Dataset
from utils import IDX2ATOM

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

FP = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
CARBOXAMIDE = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")


def training_inhibitors(parp_npz):
    """(smiles, fingerprint) for every real PARP inhibitor in the training set."""
    ds = QM9Dataset(parp_npz)
    seen, out = set(), []
    for i in range(len(ds)):
        mol = assign_bond_orders([IDX2ATOM[int(t)] for t in ds[i]["types"]],
                                 ds[i]["pos"].astype(np.float64))
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if smi in seen:
            continue
        seen.add(smi)
        out.append((smi, FP.GetFingerprint(mol)))
    return out


def named_drugs(path):
    """(name, smiles, fingerprint) for the curated reference inhibitors."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is not None:
                out.append((row["name"], row.get("parps", ""),
                            Chem.MolToSmiles(mol), FP.GetFingerprint(mol)))
    return out


def nearest(fp, pool, smiles_at):
    """Index and similarity of the pool entry closest to fp."""
    sims = DataStructs.BulkTanimotoSimilarity(fp, [p[smiles_at] for p in pool])
    best = max(range(len(sims)), key=lambda i: sims[i])
    return best, float(sims[best])


def draw(mol, path, size=440):
    """Write a clean 2D depiction, generating coordinates from scratch."""
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    AllChem.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    d.drawOptions().clearBackground = False
    d.drawOptions().bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    with open(path, "wb") as f:
        f.write(d.GetDrawingText())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", default="candidates_v11/candidates.csv")
    p.add_argument("--parp-npz", default="parp_noH.npz")
    p.add_argument("--references",
                   default="../../Datasets/PARP_inhibitors_July26/curated_parp_inhibitors.csv")
    p.add_argument("--out-dir", default="parp_gallery")
    p.add_argument("--top", type=int, default=24)
    p.add_argument("--checkpoint", default="v11_noH_epoch_499",
                   help="recorded on the page so the figures stay traceable")
    args = p.parse_args()

    images = os.path.join(args.out_dir, "images")
    os.makedirs(images, exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.candidates, newline="")) if r.get("smiles")]
    print(f"{len(rows)} candidates from {args.candidates}", flush=True)
    print("fingerprinting the training inhibitors...", flush=True)
    pool = training_inhibitors(args.parp_npz)
    drugs = named_drugs(args.references)
    print(f"  {len(pool)} distinct real inhibitors, {len(drugs)} named drugs", flush=True)

    cards = []
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        fp = FP.GetFingerprint(mol)
        pi, psim = nearest(fp, pool, 1)
        di, dsim = nearest(fp, drugs, 3)
        cards.append({
            "mol": mol, "smiles": Chem.MolToSmiles(mol),
            "sim": psim, "ref_smiles": pool[pi][0],
            "drug": drugs[di][0], "drug_targets": drugs[di][1], "drug_sim": dsim,
            "mw": float(row.get("mol_weight") or Descriptors.MolWt(mol)),
            "logp": float(row.get("logp") or Crippen.MolLogP(mol)),
            "aromatic": int(row.get("aromatic_rings") or rdMolDescriptors.CalcNumAromaticRings(mol)),
            "stereo": int(row.get("stereocentres") or 0),
            "strain": row.get("strain_per_heavy_atom", ""),
            "sa": sascorer.calculateScore(mol),
            "amide": mol.HasSubstructMatch(CARBOXAMIDE),
            "n_heavy": mol.GetNumHeavyAtoms(),
        })

    cards.sort(key=lambda c: -c["sim"])
    shown = cards[:args.top]
    print(f"drawing {len(shown)} pairs...", flush=True)
    for i, c in enumerate(shown):
        c["img"] = f"images/candidate_{i:02d}.png"
        c["ref_img"] = f"images/nearest_{i:02d}.png"
        draw(c["mol"], os.path.join(args.out_dir, c["img"]))
        draw(Chem.MolFromSmiles(c["ref_smiles"]), os.path.join(args.out_dir, c["ref_img"]), size=300)

    all_sims = np.array([c["sim"] for c in cards])
    all_sa = np.array([c["sa"] for c in cards])
    stats = [
        (f"{len(cards)}", "candidates"),
        (f"{sum(c['amide'] for c in cards) / len(cards):.0%}", "carry a carboxamide"),
        (f"{np.median(all_sims):.2f}", "median similarity"),
        (f"{all_sims.max():.2f}", "best similarity"),
        (f"{np.median(all_sa):.1f}", "median SA score"),
        (f"{np.mean(all_sa <= 4.0):.0%}", "SA score under 4"),
    ]
    write_page(args, shown, stats, len(cards), len(pool),
               float(np.median(all_sims)), float(np.mean(all_sims < 0.40)))
    print(f"\ngallery: {os.path.join(args.out_dir, 'index.html')}")
    print(f"images:  {images}/ ({2 * len(shown)} files)")


def write_page(args, cards, stats, n_total, n_pool, median_sim, frac_below):
    e = html.escape
    css = CSS
    stat_html = "\n".join(
        f'      <div class="stat"><span class="n">{e(n)}</span><span class="l">{e(l)}</span></div>'
        for n, l in stats)

    card_html = []
    for i, c in enumerate(cards):
        classes = "card"
        if c["sim"] >= 0.5:
            classes += " t-close"
        if c["sa"] <= 4.0:
            classes += " t-easy"
        if c["amide"]:
            classes += " t-amide"
        badge = (f'<span class="badge ok">{c["sim"]:.2f}</span>' if c["sim"] >= 0.5
                 else f'<span class="badge">{c["sim"]:.2f}</span>')
        strain = f'{float(c["strain"]):.1f}' if c["strain"] not in ("", None) else "n/a"
        card_html.append(f'''    <article class="{classes}">
      <div class="diagram"><img src="{c['img']}" alt="generated candidate {i + 1}" loading="lazy"></div>
      <div class="card-body">
        <div class="card-head">
          <span class="rank">#{i + 1}</span>
          {badge}
        </div>
        <dl class="props">
          <div><dt>MW</dt><dd>{c['mw']:.0f}</dd></div>
          <div><dt>logP</dt><dd>{c['logp']:.1f}</dd></div>
          <div><dt>arom. rings</dt><dd>{c['aromatic']}</dd></div>
          <div><dt>stereocentres</dt><dd>{c['stereo']}</dd></div>
          <div><dt>SA score</dt><dd>{c['sa']:.1f}</dd></div>
          <div><dt>strain/atom</dt><dd>{strain}</dd></div>
        </dl>
        <p class="smi">{e(c['smiles'])}</p>
        <div class="nearest">
          <img src="{c['ref_img']}" alt="nearest known inhibitor" loading="lazy">
          <div>
            <span class="l">Nearest real inhibitor</span>
            <strong>Tanimoto {c['sim']:.2f}</strong>
            <span class="l">Nearest named drug</span>
            <strong>{e(c['drug'])} &middot; {c['drug_sim']:.2f}</strong>
          </div>
        </div>
      </div>
    </article>''')

    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PARP Candidate Gallery</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;650&display=swap" rel="stylesheet">
<style>
{css}
</style>
</head>
<body>
<div class="wrap">
  <header class="page">
    <p class="eyebrow">Generated &middot; checkpoint {e(args.checkpoint)}</p>
    <h1>Candidates that look like PARP inhibitors</h1>
    <p class="lede">
      The {len(cards)} generated molecules closest to a real PARP inhibitor, out of
      {n_total} that passed the chemistry filters. Each card shows the generated
      structure, and underneath it the nearest of the {n_pool} real inhibitors the
      model was trained on &mdash; Tanimoto over Morgan fingerprints
      (radius&nbsp;2, 2048&nbsp;bits).
    </p>
    <p class="lede">
      <strong>This is the similar tail, not a typical sample.</strong> These
      {len(cards)} span {cards[-1]['sim']:.2f}&ndash;{cards[0]['sim']:.2f}, while the median
      candidate sits at {median_sim:.2f} and {frac_below:.0%} fall below 0.40. The closest
      ones are near-analogues: the top candidate differs from a known inhibitor
      by a single atom. Read them as evidence that the model has learned real
      PARP chemistry, not as novel chemical matter.
      <strong>Similarity is structural only.</strong> Nothing here predicts that
      a molecule inhibits PARP, and bond orders are reconstructed from 3D
      coordinates with 85.2% exact accuracy on real molecules, so roughly one
      structure in seven carries a wrong bond order or tautomer.
    </p>
    <div class="stats">
{stat_html}
    </div>
    <div class="filters">
      <button class="active" data-f="all">All</button>
      <button data-f="t-close">Tanimoto &ge; 0.50</button>
      <button data-f="t-easy">SA score &le; 4</button>
      <button data-f="t-amide">Carboxamide</button>
    </div>
  </header>
  <main class="grid">
{chr(10).join(card_html)}
  </main>
  <footer>
    <p>
      Regenerate with <code>python scripts/make_gallery.py --candidates
      {e(args.candidates)} --checkpoint {e(args.checkpoint)}</code>.
      SA score: Ertl &amp; Schuffenhauer (1&nbsp;easy &ndash; 10&nbsp;hard).
      Strain is unrelaxed MMFF energy per heavy atom at the generated geometry;
      real relaxed conformers sit near 1&ndash;5&nbsp;kcal/mol/atom.
    </p>
  </footer>
</div>
<script>
  const buttons = document.querySelectorAll('.filters button');
  const cards = document.querySelectorAll('.card');
  buttons.forEach(b => b.addEventListener('click', () => {{
    buttons.forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    const f = b.dataset.f;
    cards.forEach(c => {{
      c.style.display = (f === 'all' || c.classList.contains(f)) ? '' : 'none';
    }});
  }}));
</script>
</body>
</html>
'''
    with open(os.path.join(args.out_dir, "index.html"), "w") as f:
        f.write(page)


CSS = '''  :root {
    --bg: #EEF1F2; --surface: #FFFFFF; --surface-2: #F7F8F9;
    --text: #17212A; --text-dim: #55636E; --border: rgba(23, 33, 42, 0.12);
    --accent: #B65E23; --accent-ink: #7A3E14; --accent-soft: rgba(182, 94, 35, 0.12);
    --ok: #2F7A6D; --ok-soft: rgba(47, 122, 109, 0.12);
    --diagram-bg: #FDFCFA; --diagram-border: rgba(23, 33, 42, 0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #12181D; --surface: #1B2429; --surface-2: #212C32;
      --text: #E7EDF0; --text-dim: #93A2AB; --border: rgba(231, 237, 240, 0.12);
      --accent: #E08A4D; --accent-ink: #F4B37F; --accent-soft: rgba(224, 138, 77, 0.16);
      --ok: #5FB3A4; --ok-soft: rgba(95, 179, 164, 0.16);
      --diagram-bg: #F4F1EC; --diagram-border: rgba(0, 0, 0, 0.08);
    }
  }
  :root[data-theme="dark"] {
    --bg: #12181D; --surface: #1B2429; --surface-2: #212C32;
    --text: #E7EDF0; --text-dim: #93A2AB; --border: rgba(231, 237, 240, 0.12);
    --accent: #E08A4D; --accent-ink: #F4B37F; --accent-soft: rgba(224, 138, 77, 0.16);
    --ok: #5FB3A4; --ok-soft: rgba(95, 179, 164, 0.16);
    --diagram-bg: #F4F1EC; --diagram-border: rgba(0, 0, 0, 0.08);
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text); margin: 0;
    font-family: "IBM Plex Sans", -apple-system, "Segoe UI", sans-serif;
    padding-inline: 16px; padding-block: 32px 64px;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  header.page { margin-bottom: 28px; }
  .eyebrow {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--accent-ink); margin: 0 0 10px;
  }
  h1 {
    font-weight: 650; font-size: clamp(27px, 4vw, 38px); line-height: 1.1;
    letter-spacing: -0.01em; margin: 0 0 10px; text-wrap: balance;
  }
  .lede {
    color: var(--text-dim); font-size: 15.5px; line-height: 1.55;
    max-width: 68ch; margin: 0 0 22px;
  }
  .lede strong { color: var(--text); }
  code {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.92em;
    background: var(--surface-2); border: 1px solid var(--border);
    padding: 0.05em 0.4em; border-radius: 4px;
  }
  .stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1px; background: var(--border); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
  }
  .stat { background: var(--surface); padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; }
  .stat .n {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums; font-size: 21px; font-weight: 600;
  }
  .stat .l { font-size: 11.5px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
  .filters { display: flex; gap: 8px; flex-wrap: wrap; margin: 22px 0 18px; }
  .filters button {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12.5px;
    background: var(--surface); color: var(--text-dim); border: 1px solid var(--border);
    border-radius: 999px; padding: 6px 13px; cursor: pointer;
  }
  .filters button:hover { color: var(--text); border-color: var(--accent); }
  .filters button.active { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-ink); }
  .filters button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(272px, 1fr)); gap: 16px; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; display: flex; flex-direction: column;
  }
  .diagram {
    background: var(--diagram-bg); border-bottom: 1px solid var(--diagram-border);
    aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center; padding: 10px;
  }
  .diagram img { max-width: 100%; max-height: 100%; display: block; }
  .card-body { padding: 12px 14px 14px; display: flex; flex-direction: column; gap: 10px; }
  .card-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .rank {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12.5px; color: var(--text-dim);
  }
  .badge {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12.5px;
    font-weight: 600; padding: 2px 9px; border-radius: 999px;
    background: var(--accent-soft); color: var(--accent-ink);
  }
  .badge.ok { background: var(--ok-soft); color: var(--ok); }
  .props { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; margin: 0; }
  .props div { display: flex; justify-content: space-between; gap: 6px; font-size: 12.5px; }
  .props dt { color: var(--text-dim); margin: 0; }
  .props dd {
    margin: 0; font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
  }
  .smi {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 10.5px;
    line-height: 1.45; color: var(--text-dim); word-break: break-all;
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 8px; margin: 0; max-height: 5.2em; overflow: auto;
  }
  .nearest {
    display: flex; gap: 10px; align-items: center;
    border-top: 1px solid var(--border); padding-top: 10px;
  }
  .nearest img {
    width: 84px; height: 84px; flex: none; border-radius: 6px;
    background: var(--diagram-bg); border: 1px solid var(--diagram-border);
  }
  .nearest > div { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
  .nearest .l { font-size: 10.5px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
  .nearest strong {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12.5px;
    font-weight: 600; margin-bottom: 5px; overflow-wrap: anywhere;
  }
  footer { margin-top: 32px; border-top: 1px solid var(--border); padding-top: 16px; }
  footer p { color: var(--text-dim); font-size: 12.5px; line-height: 1.6; max-width: 76ch; margin: 0; }
'''

if __name__ == "__main__":
    main()
