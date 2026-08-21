"""
Train the equivariant molecule diffusion model.

Usage (synthetic smoke test, no external data needed):
    python train.py --synthetic --epochs 5 --n-molecules 256 --out ckpt.pt

Usage (real data, from a preprocessed QM9 .npz — see dataset.py):
    python train.py --data-npz path/to/qm9.npz --epochs 200 --out ckpt.pt
"""

from __future__ import annotations

import argparse
import functools
import time

import torch
from torch.utils.data import DataLoader

from dataset import QM9Dataset, SyntheticMoleculeDataset, collate_molecules
from diffusion import EquivariantMoleculeDiffusion
from utils import EMA


def build_dataset(args):
    if args.synthetic:
        return SyntheticMoleculeDataset(n_molecules=args.n_molecules, seed=args.seed)
    if not args.data_npz:
        raise ValueError("Pass --synthetic for a smoke test, or --data-npz <path> for real data.")
    return QM9Dataset(args.data_npz)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="Use the built-in synthetic dataset (no download needed).")
    p.add_argument("--data-npz", type=str, default=None, help="Path to a preprocessed QM9-style .npz file.")
    p.add_argument("--n-molecules", type=int, default=512, help="Size of the synthetic dataset (if used).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoint.pt")
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print("Device:", device)
    dataset = build_dataset(args)
    max_atoms = max(dataset[i]["n_nodes"] for i in range(len(dataset)))
    collate = functools.partial(collate_molecules, max_atoms=max_atoms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = EquivariantMoleculeDiffusion(
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, timesteps=args.timesteps,
    ).to(device)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-12)

    step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        running = {"loss": 0.0, "loss_x": 0.0, "loss_h": 0.0, "n": 0}
        for x0, h0, node_mask in loader:
            x0, h0, node_mask = x0.to(device), h0.to(device), node_mask.to(device)

            loss, parts = model.loss(x0, h0, node_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model)

            running["loss"] += loss.item()
            running["loss_x"] += parts["loss_x"]
            running["loss_h"] += parts["loss_h"]
            running["n"] += 1
            step += 1

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            n = max(running["n"], 1)
            print(f"epoch {epoch:4d} | loss {running['loss']/n:.4f} "
                  f"(x {running['loss_x']/n:.4f}, h {running['loss_h']/n:.4f}) "
                  f"| {time.time()-t0:.1f}s")

    torch.save({
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "args": vars(args),
    }, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
