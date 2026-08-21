"""
Equivariant diffusion process over (atom positions x, atom types h).

Follows Hoogeboom et al. 2022, "Equivariant Diffusion for Molecule
Generation in 3D" (EDM): a single variance-preserving DDPM-style diffusion
jointly over continuous 3D coordinates and continuous relaxations of
categorical atom-type one-hot vectors, with an E(3)-equivariant denoiser
(EGNN, see egnn.py).

Key ideas (details in references/approach.md):
  1. x lives in the zero-center-of-mass subspace (translation invariance).
  2. Noise added to x is likewise zero-CoM, so the whole forward/reverse
     process stays on that subspace -> the model only needs to be
     rotation/reflection equivariant, not translation equivariant, on top.
  3. h (atom types) is treated as continuous data in R^{num_types} via the
     one-hot encoding and diffused the same way; at sampling time we take
     an argmax to recover discrete atom types.
  4. Loss is the standard simple noise-prediction objective (Ho et al. 2020
     DDPM), applied jointly to x and h with a relative weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from egnn import EGNN
from utils import NUM_ATOM_TYPES, build_edge_mask, cosine_alpha_bar, remove_mean_with_mask


class EquivariantMoleculeDiffusion(nn.Module):
    def __init__(self, num_atom_types: int = NUM_ATOM_TYPES, hidden_dim: int = 128,
                 n_layers: int = 4, timesteps: int = 1000, h_loss_weight: float = 1.0,
                 cond_dim: int = 0):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.timesteps = timesteps
        self.h_loss_weight = h_loss_weight
        self.cond_dim = cond_dim
        # in_node_dim = one-hot atom type; time (and optionally a condition
        # vector) are appended inside EGNN
        self.net = EGNN(in_node_dim=num_atom_types, hidden_dim=hidden_dim, n_layers=n_layers, cond_dim=cond_dim)

    # ---------------------------------------------------------------- #
    # Forward (noising) process
    # ---------------------------------------------------------------- #
    def q_sample(self, x0, h0, t, node_mask):
        """Sample z_t ~ q(z_t | x0, h0) for continuous timestep t in [0,1].

        Returns (x_t, h_t, eps_x, eps_h) — the noise targets are returned
        too since the training loss regresses the network onto them.
        """
        alpha_bar = cosine_alpha_bar(t).view(-1, 1, 1)          # (B,1,1)
        sqrt_ab = alpha_bar.sqrt()
        sqrt_1mab = (1 - alpha_bar).sqrt()

        eps_x = torch.randn_like(x0)
        eps_x = remove_mean_with_mask(eps_x, node_mask)  # keep noise on zero-CoM subspace
        eps_h = torch.randn_like(h0) * node_mask

        x_t = sqrt_ab * x0 + sqrt_1mab * eps_x
        h_t = sqrt_ab * h0 + sqrt_1mab * eps_h
        x_t = remove_mean_with_mask(x_t, node_mask)
        return x_t, h_t, eps_x, eps_h

    # ---------------------------------------------------------------- #
    # Training loss
    # ---------------------------------------------------------------- #
    def loss(self, x0, h0, node_mask, cond=None):
        """
        x0: (B, N, 3) ground-truth atom coordinates (will be CoM-centered)
        h0: (B, N, num_atom_types) ground-truth one-hot atom types
        node_mask: (B, N, 1)
        cond: (B, cond_dim) optional per-molecule condition vector, required
              iff this model was constructed with cond_dim > 0.
        """
        B = x0.shape[0]
        device = x0.device
        x0 = remove_mean_with_mask(x0, node_mask)

        t = torch.rand(B, device=device)  # continuous t in [0, 1), one per molecule
        x_t, h_t, eps_x, eps_h = self.q_sample(x0, h0, t, node_mask)

        edge_mask = build_edge_mask(node_mask)
        eps_h_pred, eps_x_pred = self.net(h_t, x_t, t.view(B, 1), node_mask, edge_mask, cond=cond)


        n_real = node_mask.sum(dim=(1, 2)).clamp(min=1.0)  # atoms per molecule
        # sum over atoms & coord dims, then mean over the batch (normalized
        # by size so molecules of different lengths contribute comparably)
        loss_x = ((eps_x_pred - eps_x) ** 2 * node_mask).sum(dim=(1, 2)) / n_real
        loss_h = ((eps_h_pred - eps_h) ** 2 * node_mask).sum(dim=(1, 2)) / n_real

        loss = (loss_x + self.h_loss_weight * loss_h).mean()
        return loss, {"loss_x": loss_x.mean().item(), "loss_h": loss_h.mean().item()}

    # ---------------------------------------------------------------- #
    # Reverse (sampling) process
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def sample(self, node_mask, n_steps: int | None = None, device=None,
               cond=None, null_cond=None, guidance_scale: float = 1.0):
        """Generate new molecules by ancestral sampling from pure noise.

        node_mask: (B, N, 1) — determines how many atoms each generated
                   molecule has (you choose molecule sizes up front, e.g.
                   sampled from the size distribution of the training set).
        cond: (B, cond_dim) optional condition vector (e.g. a one-hot class
              label), required iff this model was built with cond_dim > 0.
        null_cond: (B, cond_dim) optional "unconditional"/null condition
              vector. If provided together with `cond` and
              `guidance_scale != 1.0`, each step extrapolates between the
              null and conditioned noise predictions:
                  eps = eps_null + guidance_scale * (eps_cond - eps_null)
              (classifier-free-guidance-style). With guidance_scale == 1.0
              (default) or null_cond=None, this reduces to plain
              conditional sampling using `cond` directly.
        Returns (x0_pred, h0_pred): coordinates and one-hot-ish atom types.
        """
        n_steps = n_steps or self.timesteps
        device = device or node_mask.device
        B, N, _ = node_mask.shape
        edge_mask = build_edge_mask(node_mask)
        use_guidance = null_cond is not None and guidance_scale != 1.0

        x_t = remove_mean_with_mask(torch.randn(B, N, 3, device=device), node_mask)
        h_t = torch.randn(B, N, self.num_atom_types, device=device) * node_mask

        ts = torch.linspace(1.0, 1.0 / n_steps, n_steps, device=device)
        for i, t_scalar in enumerate(ts):
            t = t_scalar.expand(B)
            t_next = (ts[i + 1] if i + 1 < n_steps else torch.zeros_like(t_scalar)).expand(B)

            ab_t = cosine_alpha_bar(t).view(B, 1, 1)
            ab_next = cosine_alpha_bar(t_next).view(B, 1, 1)

            if use_guidance:
                eps_h_cond, eps_x_cond = self.net(h_t, x_t, t.view(B, 1), node_mask, edge_mask, cond=cond)
                eps_h_null, eps_x_null = self.net(h_t, x_t, t.view(B, 1), node_mask, edge_mask, cond=null_cond)
                eps_h_pred = eps_h_null + guidance_scale * (eps_h_cond - eps_h_null)
                eps_x_pred = eps_x_null + guidance_scale * (eps_x_cond - eps_x_null)
            else:
                eps_h_pred, eps_x_pred = self.net(h_t, x_t, t.view(B, 1), node_mask, edge_mask, cond=cond)
            eps_x_pred = remove_mean_with_mask(eps_x_pred, node_mask)

            # predict x0 from current noisy sample + predicted noise
            x0_pred = (x_t - (1 - ab_t).sqrt() * eps_x_pred) / ab_t.sqrt().clamp(min=1e-8)
            h0_pred = (h_t - (1 - ab_t).sqrt() * eps_h_pred) / ab_t.sqrt().clamp(min=1e-8)
            x0_pred = remove_mean_with_mask(x0_pred, node_mask)

            # DDIM-style deterministic-ish step toward t_next (with a small
            # amount of injected noise, except on the very last step)
            noise_x = remove_mean_with_mask(torch.randn_like(x_t), node_mask)
            noise_h = torch.randn_like(h_t) * node_mask
            sigma = (1 - ab_next).sqrt() * 0.2 if i < n_steps - 1 else 0.0

            x_t = ab_next.sqrt() * x0_pred + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps_x_pred + sigma * noise_x
            h_t = ab_next.sqrt() * h0_pred + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps_h_pred + sigma * noise_h
            x_t = remove_mean_with_mask(x_t, node_mask)
            h_t = h_t * node_mask

        return x_t, h_t
