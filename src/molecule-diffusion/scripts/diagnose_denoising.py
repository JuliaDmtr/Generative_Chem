"""
Where do the near-60° angles (3-membered rings) come from?

Part 1 — one-step denoising of REAL molecules. Noise real PARP conformers to
level t exactly as in training, let the network predict the clean molecule
x0_pred in one step, and measure it. If x0_pred already contains excess
near-60° angles at low t, the network's local precision is the bottleneck.
Reported next to the naive estimate x_t / sqrt(ab_t) (no network), so we can
see whether the network removes or adds triangles.

Part 2 — along actual sampling from pure noise. Record the network's x0_pred
at a range of t during generation, to see at which noise level the excess
appears and whether it is ever corrected.

Usage (from src/molecule-diffusion):
    python scripts/diagnose_denoising.py checkpoints/v7_fixed_epoch_299.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from dataset import QM9Dataset, collate_molecules
from diagnose_scale import bond_angles
from evaluate_checkpoint import load_model
from utils import COND_POSITIVE, H_SCALE, build_edge_mask, cosine_alpha_bar, make_condition_batch, remove_mean_with_mask

T_ONE_STEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
T_TRAJECTORY = [0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]


def angle_band(xs, node_mask) -> float:
    """Fraction of bond angles (neighbors within 1.8 Å) in the 50-70° band."""
    angles = []
    for b in range(xs.shape[0]):
        n = int(node_mask[b].sum())
        a = bond_angles(xs[b, :n].double().numpy())
        if len(a):
            angles.append(a)
    a = np.concatenate(angles)
    return ((a >= 50) & (a <= 70)).mean()


def mean_nn(xs, node_mask) -> float:
    out = []
    for b in range(xs.shape[0]):
        n = int(node_mask[b].sum())
        x = xs[b, :n]
        d = torch.cdist(x, x) + torch.eye(n) * 1e9
        out.append(d.min(1).values.mean().item())
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--parp-npz", default="parp_3d.npz")
    p.add_argument("--n-molecules", type=int, default=128)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    model, epoch = load_model(args.ckpt, device)

    parp = QM9Dataset(args.parp_npz)
    rng = np.random.default_rng(args.seed)
    items = [parp[int(i)] for i in rng.choice(len(parp), args.n_molecules, replace=False)]
    x0, h0, nm = collate_molecules(items)
    x0 = remove_mean_with_mask(x0, nm)
    B = x0.shape[0]
    cond = make_condition_batch(COND_POSITIVE, B, device=device)
    edge_mask = build_edge_mask(nm).to(device)

    print(f"checkpoint {args.ckpt} (epoch {epoch}); {B} real PARP conformers")
    print(f"real molecules: angles 50-70° {angle_band(x0, nm):.1%}, mean NN distance {mean_nn(x0, nm):.3f} Å\n")

    print("== Part 1: one-step denoising of real molecules ==")
    print("   t    ab_t | naive x_t/sqrt(ab): angles  NN   | network x0_pred: angles  NN     RMSD   type acc")
    torch.manual_seed(args.seed)
    for t_val in T_ONE_STEP:
        t = torch.full((B,), t_val)
        ab = cosine_alpha_bar(t).view(B, 1, 1)
        eps_x = remove_mean_with_mask(torch.randn_like(x0), nm)
        eps_h = torch.randn_like(h0) * nm
        x_t = remove_mean_with_mask(ab.sqrt() * x0 + (1 - ab).sqrt() * eps_x, nm)
        h_t = (ab.sqrt() * h0 * H_SCALE + (1 - ab).sqrt() * eps_h) * nm
        with torch.no_grad():
            eh, ex = model.net(h_t.to(device), x_t.to(device), t.view(B, 1).to(device), nm.to(device), edge_mask, cond=cond)
        ex, eh = ex.cpu(), eh.cpu()
        x0_pred = remove_mean_with_mask((x_t - (1 - ab).sqrt() * ex) / ab.sqrt(), nm)
        h0_pred = (h_t - (1 - ab).sqrt() * eh) / ab.sqrt()
        naive = x_t / ab.sqrt()
        real = nm.squeeze(-1).bool()
        rmsd = ((x0_pred - x0) ** 2).sum(-1)[real].mean().sqrt().item()
        acc = (h0_pred.argmax(-1) == h0.argmax(-1))[real].float().mean().item()
        print(f"  {t_val:.2f}  {ab.flatten()[0]:.3f} |          {angle_band(naive, nm):6.1%}  {mean_nn(naive, nm):.2f} |"
              f"          {angle_band(x0_pred, nm):6.1%}  {mean_nn(x0_pred, nm):.2f}  {rmsd:5.2f} Å   {acc:6.1%}")

    print("\n== Part 2: network's x0_pred along sampling from pure noise ==")
    records = []
    orig_forward = model.net.forward

    def spy(h, x, t, *a, **k):
        eh, ex = orig_forward(h, x, t, *a, **k)
        records.append((t[0, 0].item(), x.detach().cpu(), ex.detach().cpu()))
        return eh, ex

    model.net.forward = spy
    sizes = [it["n_nodes"] for it in items]
    snm = torch.zeros(B, max(sizes), 1)
    for b, n in enumerate(sizes):
        snm[b, :n] = 1.0
    torch.manual_seed(args.seed)
    with torch.no_grad():
        x_final, _ = model.sample(snm.to(device), n_steps=args.n_steps, device=device, cond=cond)
    del model.net.forward

    ts = np.array([r[0] for r in records])
    print("   t    | x0_pred: angles  NN")
    for t_val in T_TRAJECTORY:
        t_rec, x_t, ex = records[int(np.abs(ts - t_val).argmin())]
        ab = cosine_alpha_bar(torch.tensor(t_rec))
        x0_pred = remove_mean_with_mask((x_t - (1 - ab).sqrt() * ex) / ab.sqrt(), snm)
        print(f"  {t_rec:.3f} |          {angle_band(x0_pred, snm):6.1%}  {mean_nn(x0_pred, snm):.2f}")
    print(f"  final  |          {angle_band(x_final.cpu(), snm):6.1%}  {mean_nn(x_final.cpu(), snm):.2f}")


if __name__ == "__main__":
    main()
