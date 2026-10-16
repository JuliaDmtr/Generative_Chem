"""
Full evaluation of one or more trained checkpoints against real PARP
conformers, with the same measurement pipeline applied to both.

Reports, per checkpoint:
  - geometry per size bucket: nearest-neighbor distance and radius of
    gyration (Rg), as ratios to real molecules of the same size;
  - RDKit validity, single-connected-molecule rate and fragment counts,
    with 95% Wilson intervals (the in-training diagnostics use only 64
    samples, too few to rank checkpoints);
  - ring-size distribution among valid molecules;
  - fraction of near-60° bond angles (the old "3-membered ring" symptom).

The real molecules go through the identical pipeline first as a
calibration: if they don't come out ~100% valid, the measurement is
broken, not the model.

Usage (from src/molecule-diffusion):
    python scripts/evaluate_checkpoint.py checkpoints/v7_fixed_epoch_299.pt [more.pt ...] \
        --n-samples 256 --n-steps 500 [--sampler ancestral|ddim --ddim-noise 0.2]
    Size sweep (per exact atom count, vs real molecules of that size):
    python scripts/evaluate_checkpoint.py checkpoints/v7_fixed_epoch_299.pt \
        --sizes 15,20,30,40 --per-size 64 --condition null --ref-npz chembl_3d.npz
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np
import torch

from dataset import QM9Dataset, sample_molecule_sizes
from diagnose_scale import bond_angles
from diffusion import EquivariantMoleculeDiffusion
from evaluate import mol_from_coords, n_fragments
from generate import molecules_from_output
from heavy_atom_eval import heavy_atom_checks
from generate_curated import ring_sizes
from utils import COND_NEGATIVE, COND_NULL, COND_POSITIVE, IDX2ATOM, check_arch_version, make_condition_batch

CONDITIONS = {"positive": COND_POSITIVE, "negative": COND_NEGATIVE, "null": COND_NULL}

BUCKETS = [(1, 34), (35, 49), (50, 999)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def geometry(coords: np.ndarray) -> tuple[float, float]:
    x = coords - coords.mean(0)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1) + np.eye(len(x)) * 1e9
    return d.min(1).mean(), np.sqrt((x ** 2).sum(1).mean())


def analyse(molecules, no_hydrogens: bool = False) -> dict:
    """molecules: list of (symbols, coords). Same pipeline for real and generated.

    no_hydrogens=True switches to the heavy-atom checks (heavy_atom_eval.py),
    which is mandatory for heavy-atom-only data: RDKit's DetermineBonds needs
    explicit hydrogens to assign bond orders. There, "valid" means the
    valences are physically possible and "single connected" means that plus
    one connected component. The two modes are NOT comparable with each other.
    """
    out = {"n": len(molecules), "geom": [], "valid": 0, "single": 0, "frags": Counter(),
           "rings": Counter(), "angles": []}
    for symbols, coords in molecules:
        out["geom"].append((len(symbols), *geometry(coords)))
        out["angles"].append(bond_angles(coords))
        if no_hydrogens:
            res = heavy_atom_checks(symbols, coords)
            if not res["perceived"]:
                continue
            out["valid"] += res["valence_ok"]
            out["single"] += res["plausible"]
            out["frags"][res["n_fragments"]] += 1
            if res["plausible"]:   # rings of a broken molecule are meaningless
                out["rings"].update(res["ring_sizes"])
            continue
        mol = mol_from_coords(symbols, coords)
        if mol is None:
            continue
        out["valid"] += 1
        f = n_fragments(mol)
        out["frags"][f] += 1
        out["single"] += f == 1
        out["rings"].update(ring_sizes(mol))
    nonempty = [a for a in out["angles"] if len(a)]
    out["angles"] = np.concatenate(nonempty) if nonempty else np.array([])
    return out


def bucket_means(geom, lo, hi):
    g = [(nn, rg) for n, nn, rg in geom if lo <= n <= hi]
    return (np.mean(g, 0), len(g)) if g else (None, 0)


def report(name: str, res: dict, ref: dict | None):
    n = res["n"]
    print(f"\n=== {name} (n={n}) ===")
    for lo, hi in BUCKETS:
        m, k = bucket_means(res["geom"], lo, hi)
        if m is None:
            continue
        label = f"atoms {lo}-{hi}" if hi < 999 else f"atoms {lo}+"
        if ref is None:
            print(f"  {label:12s} (n={k:3d}): NN {m[0]:.3f} Å, Rg {m[1]:.2f} Å")
        else:
            r, _ = bucket_means(ref["geom"], lo, hi)
            if r is None:
                print(f"  {label:12s} (n={k:3d}): NN {m[0]:.3f} Å, Rg {m[1]:.2f} Å  (no real molecules in this bucket)")
                continue
            print(f"  {label:12s} (n={k:3d}): NN x{m[0]/r[0]:.2f}, Rg x{m[1]/r[1]:.2f}  (Rg {m[1]:.2f} vs {r[1]:.2f} Å)")
    for key, label in (("valid", "valid"), ("single", "single connected")):
        lo, hi = wilson(res[key], n)
        print(f"  {label:17s}: {res[key]/n:6.1%}  [95% CI {lo:.1%}-{hi:.1%}]")
    if res["valid"]:
        frags = ", ".join(f"{k}:{v}" for k, v in sorted(res["frags"].items()))
        print(f"  fragments among valid (count:molecules): {frags}")
        total = sum(res["rings"].values())
        if total:
            dist = ", ".join(f"{s}-ring {c/total:.0%}" for s, c in sorted(res["rings"].items()))
            print(f"  ring sizes among valid ({total} rings): {dist}")
        else:
            print("  ring sizes among valid: no rings")
    a = res["angles"]
    if len(a):
        print(f"  angles in 50-70° band: {((a >= 50) & (a <= 70)).mean():.1%}  (median angle {np.median(a):.0f}°)")


def size_sweep_row(molecules, no_hydrogens: bool = False) -> dict:
    """Per-size summary: near-60° angle share, validity, single-connected, 3-ring share."""
    res = analyse(molecules, no_hydrogens=no_hydrogens)
    a, rings = res["angles"], res["rings"]
    return {"n": res["n"], "band": ((a >= 50) & (a <= 70)).mean() if len(a) else float("nan"),
            "valid": res["valid"] / res["n"], "single": res["single"] / res["n"],
            "ring3": rings[3] / sum(rings.values()) if sum(rings.values()) else float("nan"),
            "n_rings": sum(rings.values())}


def generate(model, sizes, condition, args, device):
    torch.manual_seed(args.seed)
    molecules = []
    for start in range(0, len(sizes), args.batch):
        chunk = sizes[start:start + args.batch]
        nm = torch.zeros(len(chunk), max(chunk), 1, device=device)
        for b, n in enumerate(chunk):
            nm[b, :n] = 1.0
        cond = make_condition_batch(condition, len(chunk), device=device)
        with torch.no_grad():
            x, h = model.sample(nm, n_steps=args.n_steps, device=device, cond=cond,
                                sampler=args.sampler, ddim_noise=args.ddim_noise)
        molecules += [(s, c.astype(np.float64)) for s, c in molecules_from_output(x.cpu(), h.cpu(), nm.cpu())]
    return molecules


def run_size_sweep(args, device):
    """Generate --per-size molecules at each exact size in --sizes and compare each
    with real molecules of exactly that size from --ref-npz."""
    ref = QM9Dataset(args.ref_npz)
    rng = np.random.default_rng(args.seed)
    sizes = [int(v) for v in args.sizes.split(",")]
    print(f"size sweep: condition={args.condition}, reference={args.ref_npz}, {args.per_size} samples per size")
    for path in args.ckpts:
        model, epoch = load_model(path, device)
        print(f"\n=== {path} (epoch {epoch}) ===")
        print(" atoms | generated: angles50-70  valid  single  3-ring share (rings) | real: n  angles50-70  valid")
        for size in sizes:
            gen = size_sweep_row(generate(model, [size] * args.per_size, CONDITIONS[args.condition], args, device),
                                 no_hydrogens=args.no_hydrogens)
            idx = [i for i in range(len(ref)) if len(ref.atom_types[i]) == size]
            idx = rng.choice(idx, min(len(idx), args.per_size), replace=False) if idx else []
            real = size_sweep_row([([IDX2ATOM[int(t)] for t in ref[int(i)]["types"]], ref[int(i)]["pos"].astype(np.float64))
                                   for i in idx], no_hydrogens=args.no_hydrogens) if len(idx) else None
            real_txt = f"{real['n']:3d}  {real['band']:10.1%}  {real['valid']:5.0%}" if real else "  0  (no real molecules)"
            print(f"  {size:4d} |           {gen['band']:6.1%}  {gen['valid']:5.0%}  {gen['single']:5.0%}  "
                  f"{gen['ring3']:6.0%} ({gen['n_rings']:3d})          |       {real_txt}", flush=True)


def load_model(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    check_arch_version(ckpt)
    a = ckpt["args"]
    model = EquivariantMoleculeDiffusion(hidden_dim=a["hidden_dim"], n_layers=a["n_layers"],
                                         timesteps=a["timesteps"], cond_dim=a.get("cond_dim", 0))
    model.load_state_dict(ckpt["ema_state"])
    return model.to(device).eval(), ckpt.get("epoch")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--parp-npz", default="parp_3d.npz")
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--sampler", choices=["ddim", "ancestral"], default="ddim")
    p.add_argument("--ddim-noise", type=float, default=0.2,
                   help="DDIM sigma as a fraction of sqrt(1 - ab_s); 0 = deterministic DDIM")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--sizes", default=None,
                   help="size-sweep mode: comma-separated exact atom counts, e.g. 15,20,30,40")
    p.add_argument("--per-size", type=int, default=64)
    p.add_argument("--condition", choices=list(CONDITIONS), default="positive")
    p.add_argument("--ref-npz", default=None, help="size-sweep reference set (default: --parp-npz)")
    p.add_argument("--no-hydrogens", action="store_true",
                   help="Data/checkpoint are heavy-atom-only: use the heavy-atom checks "
                        "(see heavy_atom_eval.py). Numbers are not comparable to with-H runs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    if args.sizes:
        args.ref_npz = args.ref_npz or args.parp_npz
        run_size_sweep(args, device)
        return

    parp = QM9Dataset(args.parp_npz)
    rng = np.random.default_rng(args.seed)
    real_idx = rng.choice(len(parp), min(args.n_samples, len(parp)), replace=False)
    real = [([IDX2ATOM[int(t)] for t in parp[int(i)]["types"]], parp[int(i)]["pos"].astype(np.float64))
            for i in real_idx]
    ref = analyse(real, no_hydrogens=args.no_hydrogens)
    report("REAL PARP conformers (calibration)"
           + (" [heavy atoms only]" if args.no_hydrogens else ""), ref, None)

    sizes = sample_molecule_sizes(parp, args.n_samples, seed=args.seed)
    for path in args.ckpts:
        model, epoch = load_model(path, device)
        molecules = generate(model, sizes, COND_POSITIVE, args, device)
        res = analyse(molecules, no_hydrogens=args.no_hydrogens)
        sampler = args.sampler if args.sampler == "ancestral" else f"ddim noise={args.ddim_noise}"
        report(f"{path} (epoch {epoch}, PARP-conditioned, {sampler}, {args.n_steps} steps)", res, ref)


if __name__ == "__main__":
    main()
