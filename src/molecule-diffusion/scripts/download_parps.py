#!/usr/bin/env python3
"""
Download and curate PARP-family inhibitors from ChEMBL for generative model fine-tuning.

Pipeline
--------
1. Resolve each gene symbol to SINGLE PROTEIN / Homo sapiens ChEMBL target(s),
   matching on curated GENE_SYMBOL synonyms rather than on search ranking.
2. Pull IC50 / Ki / Kd activities, filtered server-side to:
      pchembl_value is not null   (ChEMBL computed it -> '=' relation, sane units)
      standard_relation == '='    (no censored ">10 uM" inactives)
      data_validity_comment is null   (no ChEMBL-flagged outliers / unit errors)
3. Join assay metadata and keep only high-confidence biochemical assays
   (confidence_score >= 8, assay_type == 'B', single-protein BAO format).
4. Standardise structures with RDKit (largest fragment, neutralise, canonicalise,
   InChIKey) if RDKit is available.
5. Aggregate to one row per (molecule, target): median pChEMBL, drop molecules
   whose replicate measurements disagree by more than MAX_LOG_SPREAD.
6. Write a curated activity table and a ready-to-train SMILES set.

Outputs
-------
    parp_activities_curated.csv   one row per surviving measurement
    parp_molecules_curated.csv    one row per (molecule, target), median pChEMBL
    parp_finetune_<TARGET>.csv    SMILES + pChEMBL above the potency cut

Usage
-----
    python download_parps.py                      # all PARP family genes
    python download_parps.py --genes PARP1 PARP2  # just the ones you want
    python download_parps.py --min-pchembl 7.0 --finetune-targets PARP1 PARP2
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Iterable

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_LIMIT = 1000          # ChEMBL hard maximum
ASSAY_BATCH = 50           # assay ids per /assay call (keeps URL length sane)
SLEEP = 0.2                # be polite to EBI
TIMEOUT = 60

PARP_GENES = [
    "PARP1", "PARP2", "PARP3", "PARP4",
    "TNKS", "TNKS2", "PARP6", "TIPARP",
    "PARP8", "PARP9", "PARP10", "PARP11",
    "PARP12", "PARP14", "PARP15", "PARP16",
]

# ChEMBL stores the curated symbol, which is not always the name people use.
GENE_ALIASES = {
    "TIPARP": ["TIPARP", "PARP7"],
    "TNKS":   ["TNKS", "PARP5A", "TNKS1"],
    "TNKS2":  ["TNKS2", "PARP5B"],
    "PARP4":  ["PARP4", "VPARP"],
}

ACTIVITY_TYPES = ["IC50", "Ki", "Kd"]

MIN_CONFIDENCE = 8         # 9 = direct single protein target, 8 = homologous
MAX_LOG_SPREAD = 1.0       # discard molecules whose replicates span >1 log unit
DEFAULT_MIN_PCHEMBL = 7.0  # 100 nM

ACTIVITY_FIELDS = [
    "molecule_chembl_id", "canonical_smiles", "target_chembl_id",
    "assay_chembl_id", "document_chembl_id", "standard_type",
    "standard_relation", "standard_value", "standard_units",
    "pchembl_value", "activity_comment", "data_validity_comment",
    "potential_duplicate", "assay_description", "src_id",
]


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def get_json(url: str, params: dict | None = None, retries: int = 4) -> dict:
    """GET with simple exponential backoff. Raises on persistent failure."""
    delay = 1.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as exc:                      # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Request failed after {retries} attempts: {url} ({last})")


def paginate(endpoint: str, params: dict, collection: str) -> list[dict]:
    """Follow ChEMBL's page_meta.next until exhausted."""
    out: list[dict] = []
    url = f"{BASE}/{endpoint}.json"
    params = {**params, "limit": PAGE_LIMIT, "offset": 0}
    while True:
        data = get_json(url, params)
        out.extend(data.get(collection, []))
        nxt = data.get("page_meta", {}).get("next")
        if not nxt:
            break
        # 'next' is a fully-formed relative URL with its own query string
        url = f"https://www.ebi.ac.uk{nxt}"
        params = None
        time.sleep(SLEEP)
    return out


# --------------------------------------------------------------------------- #
# 1. Target resolution
# --------------------------------------------------------------------------- #
def gene_symbols(target: dict) -> set[str]:
    """Curated GENE_SYMBOL synonyms attached to a target's protein components."""
    syms: set[str] = set()
    for comp in target.get("target_components", []) or []:
        for syn in comp.get("target_component_synonyms", []) or []:
            if (syn.get("syn_type") or "").upper() == "GENE_SYMBOL":
                syms.add((syn.get("component_synonym") or "").upper())
    return syms


def resolve_targets(gene: str) -> list[dict]:
    """
    Find human SINGLE PROTEIN targets whose curated gene symbol matches `gene`.

    Unlike /target/search, this does not depend on relevance ranking, so it
    cannot silently return a protein complex, a family node or a cell line.
    """
    aliases = [a.upper() for a in GENE_ALIASES.get(gene, [gene])]
    found: dict[str, dict] = {}

    for alias in aliases:
        params = {
            "target_components__target_component_synonyms__component_synonym__iexact": alias,
            "target_type": "SINGLE PROTEIN",
            "organism": "Homo sapiens",
        }
        for tgt in paginate("target", params, "targets"):
            # Confirm the hit really carries the symbol as a GENE_SYMBOL,
            # not just as a description substring.
            if gene_symbols(tgt) & set(aliases):
                found[tgt["target_chembl_id"]] = tgt
        time.sleep(SLEEP)

    return list(found.values())


# --------------------------------------------------------------------------- #
# 2. Activities
# --------------------------------------------------------------------------- #
def fetch_activities(target_chembl_id: str) -> list[dict]:
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": ",".join(ACTIVITY_TYPES),
        "pchembl_value__isnull": "false",
        "standard_relation": "=",
        "data_validity_comment__isnull": "true",
    }
    return paginate("activity", params, "activities")


# --------------------------------------------------------------------------- #
# 3. Assay metadata
# --------------------------------------------------------------------------- #
def fetch_assays(assay_ids: Iterable[str]) -> pd.DataFrame:
    ids = sorted({a for a in assay_ids if a})
    rows: list[dict] = []
    for i in range(0, len(ids), ASSAY_BATCH):
        batch = ids[i:i + ASSAY_BATCH]
        data = get_json(
            f"{BASE}/assay.json",
            {"assay_chembl_id__in": ",".join(batch), "limit": PAGE_LIMIT},
        )
        rows.extend(data.get("assays", []))
        time.sleep(SLEEP)
        print(f"   assays {min(i + ASSAY_BATCH, len(ids))}/{len(ids)}", end="\r")
    print()
    if not rows:
        return pd.DataFrame(
            columns=["assay_chembl_id", "assay_type", "confidence_score",
                     "bao_label", "assay_organism"]
        )
    df = pd.DataFrame(rows)
    keep = ["assay_chembl_id", "assay_type", "confidence_score",
            "bao_label", "assay_organism"]
    return df[[c for c in keep if c in df.columns]]


# --------------------------------------------------------------------------- #
# 4. Structure standardisation
# --------------------------------------------------------------------------- #
def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Largest fragment -> neutralise -> canonical SMILES + InChIKey.

    Without RDKit the script still runs; it just falls back to ChEMBL's own
    canonical SMILES and de-duplicates on molecule_chembl_id instead.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        print("!  RDKit not found - skipping standardisation "
              "(install with: conda install -c conda-forge rdkit)")
        df["std_smiles"] = df["canonical_smiles"]
        df["inchikey"] = pd.NA
        return df

    RDLogger.DisableLog("rdApp.*")
    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    cache: dict[str, tuple[Any, Any]] = {}

    def one(smi: str) -> tuple[Any, Any]:
        if not isinstance(smi, str) or not smi:
            return (pd.NA, pd.NA)
        if smi in cache:
            return cache[smi]
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                res = (pd.NA, pd.NA)
            else:
                mol = rdMolStandardize.Cleanup(mol)
                mol = chooser.choose(mol)       # strip salts / counterions
                mol = uncharger.uncharge(mol)   # neutralise where possible
                res = (Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol))
        except Exception:                        # noqa: BLE001
            res = (pd.NA, pd.NA)
        cache[smi] = res
        return res

    std = df["canonical_smiles"].map(one)
    df["std_smiles"] = [s for s, _ in std]
    df["inchikey"] = [k for _, k in std]
    return df


# --------------------------------------------------------------------------- #
# 5. Aggregation
# --------------------------------------------------------------------------- #
def aggregate(df: pd.DataFrame, max_spread: float) -> pd.DataFrame:
    """
    One row per (molecule, target). Median pChEMBL in log space; molecules whose
    independent measurements disagree by more than `max_spread` logs are dropped
    rather than averaged, because a 2-log disagreement is not noise you can mean
    away - it means the two experiments were not measuring the same thing.
    """
    key = ["target_gene", "inchikey_or_id"]
    g = df.groupby(key, dropna=False)

    agg = g.agg(
        n_measurements=("pchembl_value", "size"),
        n_documents=("document_chembl_id", "nunique"),
        pchembl_median=("pchembl_value", "median"),
        pchembl_min=("pchembl_value", "min"),
        pchembl_max=("pchembl_value", "max"),
        molecule_chembl_id=("molecule_chembl_id", "first"),
        std_smiles=("std_smiles", "first"),
        activity_types=("standard_type", lambda s: "|".join(sorted(set(s)))),
    ).reset_index()

    agg["log_spread"] = agg["pchembl_max"] - agg["pchembl_min"]
    inconsistent = (agg["n_measurements"] > 1) & (agg["log_spread"] > max_spread)
    print(f"   dropped {int(inconsistent.sum())} molecule-target pairs "
          f"with >{max_spread} log disagreement")
    agg = agg[~inconsistent].copy()

    agg["ic50_nM_equiv"] = 10 ** (9 - agg["pchembl_median"])
    return agg.sort_values(["target_gene", "pchembl_median"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--genes", nargs="+", default=PARP_GENES)
    p.add_argument("--min-pchembl", type=float, default=DEFAULT_MIN_PCHEMBL,
                   help="potency cut for the fine-tuning set (7.0 = 100 nM)")
    p.add_argument("--min-confidence", type=int, default=MIN_CONFIDENCE)
    p.add_argument("--max-log-spread", type=float, default=MAX_LOG_SPREAD)
    p.add_argument("--finetune-targets", nargs="+", default=["PARP1", "PARP2"],
                   help="write a separate SMILES file per target listed here")
    p.add_argument("--keep-functional", action="store_true",
                   help="keep assay_type 'F' (cell-based/functional) as well as 'B'")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    frames: list[pd.DataFrame] = []

    for gene in args.genes:
        print(f"\n== {gene}")
        try:
            targets = resolve_targets(gene)
        except RuntimeError as exc:
            print(f"   target lookup failed: {exc}")
            continue

        if not targets:
            print("   no human SINGLE PROTEIN target with that gene symbol")
            continue

        for tgt in targets:
            tid = tgt["target_chembl_id"]
            print(f"   {tid}  {tgt.get('pref_name')}  "
                  f"[{','.join(sorted(gene_symbols(tgt))) or '-'}]")
            try:
                acts = fetch_activities(tid)
            except RuntimeError as exc:
                print(f"      activity fetch failed: {exc}")
                continue
            if not acts:
                print("      no qualifying activities")
                continue
            df = pd.DataFrame(acts)
            for col in ACTIVITY_FIELDS:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[ACTIVITY_FIELDS].copy()
            df["target_gene"] = gene
            df["target_pref_name"] = tgt.get("pref_name")
            frames.append(df)
            print(f"      {len(df)} activities")

    if not frames:
        print("\nNo data retrieved.")
        return 1

    raw = pd.concat(frames, ignore_index=True)
    print(f"\nRaw qualifying activities: {len(raw)}")

    # --- assay-level quality gate -----------------------------------------
    print("Fetching assay metadata...")
    assays = fetch_assays(raw["assay_chembl_id"])
    df = raw.merge(assays, on="assay_chembl_id", how="left")

    before = len(df)
    allowed_types = ["B", "F"] if args.keep_functional else ["B"]
    df = df[df["assay_type"].isin(allowed_types)]
    print(f"  assay_type in {allowed_types}: {len(df)}/{before}")

    df = df[pd.to_numeric(df["confidence_score"], errors="coerce")
            >= args.min_confidence]
    print(f"  confidence_score >= {args.min_confidence}: {len(df)}")

    if not args.keep_functional:
        # BAO format gives an explicit, non-regex way to exclude cell-based
        # readouts. Missing labels are kept - assay_type B already screened them.
        bao = df["bao_label"].fillna("")
        cell_based = bao.str.contains("cell-based|cell membrane|organism-based",
                                      case=False, regex=True)
        df = df[~cell_based]
        print(f"  non-cell-based BAO format: {len(df)}")

    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])
    df = df[df["potential_duplicate"].fillna(0).astype(float) == 0]
    print(f"  valid pChEMBL, SMILES, non-duplicate: {len(df)}")

    # --- structures --------------------------------------------------------
    print("Standardising structures...")
    df = standardise(df)
    df = df.dropna(subset=["std_smiles"])
    df["inchikey_or_id"] = df["inchikey"].fillna(df["molecule_chembl_id"])
    print(f"  parsed OK: {len(df)} rows, "
          f"{df['inchikey_or_id'].nunique()} unique structures")

    # --- aggregate ---------------------------------------------------------
    print("Aggregating per molecule...")
    agg = aggregate(df, args.max_log_spread)

    act_path = f"{args.outdir}/parp_activities_curated.csv"
    mol_path = f"{args.outdir}/parp_molecules_curated.csv"
    df.to_csv(act_path, index=False)
    agg.to_csv(mol_path, index=False)
    print(f"\nWrote {act_path}  ({len(df)} rows)")
    print(f"Wrote {mol_path}  ({len(agg)} molecule-target pairs)")

    # --- fine-tuning sets --------------------------------------------------
    print(f"\nFine-tuning sets at pChEMBL >= {args.min_pchembl} "
          f"(IC50 <= {10 ** (9 - args.min_pchembl):.0f} nM):")
    for gene in args.finetune_targets:
        sub = agg[(agg["target_gene"] == gene)
                  & (agg["pchembl_median"] >= args.min_pchembl)]
        if sub.empty:
            print(f"  {gene}: nothing above the cut")
            continue
        sub = sub.drop_duplicates("std_smiles")
        out = f"{args.outdir}/parp_finetune_{gene}.csv"
        sub[["std_smiles", "pchembl_median", "ic50_nM_equiv",
             "molecule_chembl_id", "n_measurements",
             "activity_types"]].to_csv(out, index=False)
        print(f"  {gene}: {len(sub)} molecules -> {out}")

    print("\nPer-target summary (all curated molecules):")
    summary = agg.groupby("target_gene").agg(
        molecules=("std_smiles", "nunique"),
        median_pchembl=("pchembl_median", "median"),
        n_ge7=("pchembl_median", lambda s: int((s >= 7).sum())),
        n_ge8=("pchembl_median", lambda s: int((s >= 8).sum())),
    ).sort_values("molecules", ascending=False)
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
