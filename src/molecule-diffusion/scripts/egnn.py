"""
E(n)-Equivariant Graph Neural Network (EGNN) layers.

Reference: Satorras, Hoogeboom & Welling, "E(n) Equivariant Graph Neural
Networks", ICML 2021. https://arxiv.org/abs/2102.09844

This is the backbone used as the noise-prediction network inside the
diffusion model (see diffusion.py), following the architecture used in
Hoogeboom et al. 2022 "Equivariant Diffusion for Molecule Generation in 3D"
(EDM), https://arxiv.org/abs/2203.17003.

Design notes
------------
- Operates on fully-connected graphs per molecule (all pairs of atoms),
  with `edge_mask` zeroing out padded atoms and self-loops. The mask is
  applied to everything that reaches a real atom: the messages AND the
  coordinate weights (coord_mlp(0) != 0 because of its biases, so masking
  only the messages would still let padded atoms push real ones).
- Coordinates `x` are updated *equivariantly*: only via relative vectors
  (x_i - x_j) scaled by a learned scalar. This guarantees the update
  commutes with any rotation/reflection/translation applied to the input.
  As in EDM, the relative vector is normalized by (||x_i - x_j|| + 1) and
  the scalar is bounded by tanh * coords_range, so each layer's update stays
  bounded however large the coordinates get.
- Output is padding invariant: neither the amount of padding nor what sits
  in the padded slots affects the real atoms (tests/test_invariances.py).
- Node features `h` are updated *invariantly*: they only ever see
  distances (invariant scalars) and other node features, never raw
  coordinates directly.
- The coordinate output eps_x is projected onto the zero-CoM subspace,
  the same subspace as the noise target, so training and sampling see
  the same prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils import remove_mean_with_mask


def mlp(sizes, act=nn.SiLU, final_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_act:
            layers.append(act())
    return nn.Sequential(*layers)


class EGNNLayer(nn.Module):
    """One equivariant message-passing layer, operating on a dense
    (batch, n_nodes, n_nodes, ...) representation so that padded/variable
    -size molecules can be batched together via masks.
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int = 0, coords_range: float = 15.0,
                 norm_constant: float = 1.0):
        super().__init__()
        self.coords_range = coords_range
        self.norm_constant = norm_constant

        # phi_e: edge message from (h_i, h_j, ||x_i-x_j||^2, edge_attr) -> message
        self.edge_mlp = mlp([2 * hidden_dim + 1 + edge_feat_dim, hidden_dim, hidden_dim])

        # phi_x: scalar weight applied to the relative coordinate vector
        self.coord_mlp = mlp([hidden_dim, hidden_dim, 1], final_act=False)
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
        nn.init.zeros_(self.coord_mlp[-1].bias)

        # phi_h: node update from (h_i, aggregated messages)
        self.node_mlp = mlp([hidden_dim + hidden_dim, hidden_dim, hidden_dim], final_act=False)

        # attention gate on messages (stabilizes training, standard EDM trick)
        self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, h, x, node_mask, edge_mask, edge_attr=None):
        """
        h: (B, N, hidden_dim)   invariant node features
        x: (B, N, 3)            equivariant coordinates
        node_mask: (B, N, 1)    1 for real atoms, 0 for padding
        edge_mask: (B, N, N, 1) 1 for real (i != j, both real) edges
        edge_attr: (B, N, N, edge_feat_dim) or None
        """
        B, N, _ = h.shape

        rel = x.unsqueeze(2) - x.unsqueeze(1)              # (B, N, N, 3)
        dist2 = (rel ** 2).sum(-1, keepdim=True)            # (B, N, N, 1)

        h_i = h.unsqueeze(2).expand(B, N, N, -1)
        h_j = h.unsqueeze(1).expand(B, N, N, -1)
        edge_in = [h_i, h_j, dist2]
        if edge_attr is not None:
            edge_in.append(edge_attr)
        m_ij = self.edge_mlp(torch.cat(edge_in, dim=-1))    # (B, N, N, hidden)
        m_ij = m_ij * self.att_mlp(m_ij)                    # gated messages
        m_ij = m_ij * edge_mask                              # zero out padded/self edges

        # --- equivariant coordinate update ---
        rel_unit = rel / (torch.sqrt(dist2 + 1e-8) + self.norm_constant)
        coord_w = torch.tanh(self.coord_mlp(m_ij)) * self.coords_range
        coord_w = coord_w * edge_mask                        # (B, N, N, 1), masked AFTER the MLP
        agg = (rel_unit * coord_w).sum(dim=2)                # (B, N, 3), sum over real j
        n_neighbors = edge_mask.sum(dim=2).clamp(min=1.0)    # real neighbors only
        x_out = (x + agg / n_neighbors) * node_mask          # padded atoms stay at 0

        # --- invariant node feature update ---
        m_i = m_ij.sum(dim=2)                                 # (B, N, hidden)
        h_out = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        h_out = h_out * node_mask                             # keep padding at 0

        return h_out, x_out


class EGNN(nn.Module):
    """Stack of EGNN layers with an input/output projection. Predicts, per
    atom, an update to the equivariant coordinates and to the invariant
    features — used as epsilon_theta(z_t, t) in the diffusion model.
    """

    def __init__(self, in_node_dim: int, hidden_dim: int = 128, n_layers: int = 4,
                 time_dim: int = 1, cond_dim: int = 0):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.embed_in = nn.Linear(in_node_dim + time_dim + cond_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_layers)])
        self.embed_out = nn.Linear(hidden_dim, in_node_dim)

    def forward(self, h, x, t, node_mask, edge_mask, cond=None):
        """
        h: (B, N, in_node_dim)  noised categorical/invariant features
        x: (B, N, 3)            noised coordinates
        t: (B, 1) or (B, N, 1)  diffusion timestep, normalized to [0, 1]
        cond: (B, cond_dim) or (B, N, cond_dim) or None — optional per-molecule
              conditioning vector (e.g. a one-hot class label), broadcast to
              every atom exactly like the timestep embedding. Only used when
              this EGNN was constructed with cond_dim > 0.
        Returns predicted noise for (h, x), same shapes as inputs.
        """
        B, N, _ = h.shape
        if t.dim() == 2:
            t = t.unsqueeze(1).expand(B, N, self.time_dim)
        h_in = [h, t]
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("This EGNN was built with cond_dim > 0; a `cond` tensor is required.")
            if cond.dim() == 2:
                cond = cond.unsqueeze(1).expand(B, N, self.cond_dim)
            h_in.append(cond)
        h_in = torch.cat(h_in, dim=-1)
        h_i = self.embed_in(h_in) * node_mask
        x_i = x

        for layer in self.layers:
            h_i, x_i = layer(h_i, x_i, node_mask, edge_mask)

        eps_h = self.embed_out(h_i) * node_mask
        # network predicts the *displacement* as eps_x, on the zero-CoM subspace
        eps_x = remove_mean_with_mask(x_i - x, node_mask)
        return eps_h, eps_x
