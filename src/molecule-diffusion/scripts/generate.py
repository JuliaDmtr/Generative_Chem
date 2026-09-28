"""
Generate new molecules from a trained checkpoint.

Usage:
    python generate.py --ckpt checkpoint.pt --n-samples 20 --out-dir samples/

By default molecule sizes are sampled uniformly from [min-atoms, max-atoms];
pass --sizes 9,12,15 to fix specific sizes, or point --size-dataset at the
training data (synthetic or .npz) to sample from its real size distribution
(recommended — matches EDM's protocol and gives more realistic outputs).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from dataset import QM9Dataset, SyntheticMoleculeDataset, sample_molecule_sizes
from diffusion import EquivariantMoleculeDiffusion
from utils import COND_NEGATIVE, COND_NULL, COND_POSITIVE, IDX2ATOM, make_condition_batch

CONDITION_NAME_TO_LABEL = {"negative": COND_NEGATIVE, "positive": COND_POSITIVE, "null": COND_NULL}


def write_xyz(path: str, atom_symbols: list[str], coords: np.ndarray, comment: str = ""):
    with open(path, "w") as f:
        f.write(f"{len(atom_symbols)}\n{comment}\n")
        for sym, (x, y, z) in zip(atom_symbols, coords):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")


def molecules_from_output(x_t: torch.Tensor, h_t: torch.Tensor, node_mask: torch.Tensor):
    """Convert raw model output into a list of (atom_symbols, coords)."""
    atom_idx = h_t.argmax(dim=-1)  # (B, N)
    out = []
    for b in range(x_t.shape[0]):
        n = int(node_mask[b].sum().item())
        symbols = [IDX2ATOM[int(atom_idx[b, i])] for i in range(n)]
        coords = x_t[b, :n].cpu().numpy()
        out.append((symbols, coords))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--sizes", type=str, default=None, help="Comma-separated fixed sizes, e.g. 9,12,15")
    p.add_argument("--min-atoms", type=int, default=4)
    p.add_argument("--max-atoms", type=int, default=19)
    p.add_argument("--size-dataset-npz", type=str, default=None,
                    help="Sample sizes from this dataset's empirical distribution instead of uniform/fixed.")
    p.add_argument("--n-steps", type=int, default=None, help="Override sampling steps (default: model's trained timesteps).")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--out-dir", type=str, default="samples")
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", type=str, default=None, choices=list(CONDITION_NAME_TO_LABEL),
                    help="Only for checkpoints trained with conditioning (train_conditional.py): "
                         "'positive' (PARP-inhibitor-like), 'negative' (generic ChEMBL-like), or "
                         "'null' (unconditional background class).")
    p.add_argument("--guidance-scale", type=float, default=1.0,
                    help="Classifier-free-guidance-style extrapolation between --condition and the "
                         "null class: eps = eps_null + guidance_scale * (eps_cond - eps_null). "
                         "1.0 (default) = no extrapolation, just use --condition directly. Requires "
                         "--condition to be 'positive' or 'negative' (not 'null').")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    margs = ckpt["args"]
    cond_dim = margs.get("cond_dim", 0)  # 0 for older unconditional checkpoints
    model = EquivariantMoleculeDiffusion(
        hidden_dim=margs["hidden_dim"], n_layers=margs["n_layers"], timesteps=margs["timesteps"],
        cond_dim=cond_dim,
    ).to(device)
    state = ckpt["ema_state"] if args.use_ema else ckpt["model_state"]
    model.load_state_dict(state)
    model.eval()

    if args.condition is not None and cond_dim == 0:
        raise ValueError("--condition was given but this checkpoint has cond_dim=0 (trained unconditionally).")
    if args.condition is None and cond_dim > 0:
        raise ValueError("This checkpoint was trained with conditioning (train_conditional.py) — pass --condition.")
    if args.guidance_scale != 1.0 and args.condition == "null":
        raise ValueError("--guidance-scale only makes sense with --condition positive/negative (extrapolates away from null).")

    # --- decide molecule sizes ---
    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
        sizes = (sizes * (args.n_samples // len(sizes) + 1))[: args.n_samples]
    elif args.size_dataset_npz:
        sizes = sample_molecule_sizes(QM9Dataset(args.size_dataset_npz), args.n_samples, seed=args.seed)
    else:
        rng = np.random.default_rng(args.seed)
        sizes = rng.integers(args.min_atoms, args.max_atoms + 1, size=args.n_samples).tolist()

    n_max = max(sizes)
    node_mask = torch.zeros(args.n_samples, n_max, 1, device=device)
    for i, n in enumerate(sizes):
        node_mask[i, :n] = 1.0

    if cond_dim > 0:
        cond = make_condition_batch(CONDITION_NAME_TO_LABEL[args.condition], args.n_samples, device=device)
        null_cond = make_condition_batch(COND_NULL, args.n_samples, device=device) if args.guidance_scale != 1.0 else None
        x_t, h_t = model.sample(node_mask, n_steps=args.n_steps, device=device,
                                 cond=cond, null_cond=null_cond, guidance_scale=args.guidance_scale)
    else:
        x_t, h_t = model.sample(node_mask, n_steps=args.n_steps, device=device)
    molecules = molecules_from_output(x_t, h_t, node_mask)

    for i, (symbols, coords) in enumerate(molecules):
        write_xyz(os.path.join(args.out_dir, f"mol_{i:04d}.xyz"), symbols, coords,
                   comment=f"generated by equivariant molecule diffusion, {len(symbols)} atoms")

    print(f"Wrote {len(molecules)} molecules to {args.out_dir}/")


if __name__ == "__main__":
    main()
