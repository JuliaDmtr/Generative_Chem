"""
vae_loss.py

Full VAE loss = reconstruction loss (atom cross-entropy + bond cross-entropy)
              + KL divergence (pulls encoder's mu/logvar toward N(0, I))

Because the decoder always outputs a fixed MAX_ATOMS-sized prediction but
real molecules have varying atom counts, we need to:
  1. Build padded ground-truth targets (pad with a "no atom" / "no bond" class)
  2. Mask out the loss contribution from padding slots, so the model isn't
     penalized for slots beyond the real molecule's size.
"""

import torch
import torch.nn.functional as F

from vae import MAX_ATOMS, NUM_ATOM_TYPES, NUM_BOND_TYPES

# Reserve the last atom-type index as "padding / no atom" class.
# NOTE: this means the atom_head must actually be sized NUM_ATOM_TYPES + 1
# to have a slot for it - see the TODO below.
PAD_ATOM_IDX = NUM_ATOM_TYPES        # requires atom_head output dim = NUM_ATOM_TYPES + 1
NO_BOND_IDX = NUM_BOND_TYPES - 1     # last bond channel = "no bond between these atoms"


def build_dense_targets(data_batch, max_atoms=MAX_ATOMS):
    """
    Converts a PyG batch of variable-sized graphs into dense, padded
    ground-truth tensors matching the decoder's fixed-size output shape.

    Returns:
        atom_targets: [batch_size, max_atoms]                (class index per atom slot)
        bond_targets: [batch_size, max_atoms, max_atoms]      (class index per atom pair)
        atom_mask:    [batch_size, max_atoms]                 (1 = real atom, 0 = padding)
    """
    device = data_batch.x.device
    batch_size = data_batch.num_graphs

    atom_targets = torch.full((batch_size, max_atoms), PAD_ATOM_IDX,
                               dtype=torch.long, device=device)
    bond_targets = torch.full((batch_size, max_atoms, max_atoms), NO_BOND_IDX,
                               dtype=torch.long, device=device)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.float, device=device)

    # atom type one-hot occupies the first NUM_ATOM_TYPES dims of node features (see smiles_to_graph.py)
    atom_type_one_hot = data_batch.x[:, :NUM_ATOM_TYPES]  # first block = atom type
    atom_type_idx = atom_type_one_hot.argmax(dim=1)       # -> class index per real atom, [total_atoms]

    batch_vec = data_batch.batch      # [total_atoms] which graph each atom belongs to
    ptr = data_batch.ptr              # [batch_size+1] start index of each graph in flattened batch

    # local index of every atom within its own molecule, fully vectorized (no python loop)
    # e.g. graph starts at ptr[g]; local_idx = global_idx - ptr[batch_vec[global_idx]]
    total_atoms = atom_type_idx.size(0)
    global_idx = torch.arange(total_atoms, device=device)
    local_idx = global_idx - ptr[batch_vec]

    valid = local_idx < max_atoms  # safety mask in case a molecule exceeds max_atoms
    g_valid = batch_vec[valid]
    local_valid = local_idx[valid]
    atom_type_valid = atom_type_idx[valid]

    # scatter all atom types into the padded tensor in one shot
    atom_targets[g_valid, local_valid] = atom_type_valid
    atom_mask[g_valid, local_valid] = 1.0

    # --- Bond targets, vectorized ---
    edge_index = data_batch.edge_index      # [2, num_edges]
    edge_attr = data_batch.edge_attr        # [num_edges, NUM_BOND_TYPES]
    edge_bond_idx = edge_attr.argmax(dim=1)  # [num_edges]

    i_global = edge_index[0]
    j_global = edge_index[1]
    g_edge = batch_vec[i_global]                 # graph id for each edge (i and j are always same graph)
    i_local = i_global - ptr[g_edge]
    j_local = j_global - ptr[g_edge]

    edge_valid = (i_local < max_atoms) & (j_local < max_atoms)
    bond_targets[g_edge[edge_valid], i_local[edge_valid], j_local[edge_valid]] = \
        edge_bond_idx[edge_valid]

    return atom_targets, bond_targets, atom_mask


def build_negative_sampling_mask(bond_targets, pair_mask, no_bond_idx):
    """
    For each molecule, builds a loss mask that includes:
      - ALL real (positive) bonded pairs
      - an EQUAL number of randomly sampled "no bond" pairs (matched per-molecule
        to that molecule's own positive-bond count)

    This directly balances the classes the loss sees, rather than reweighting
    an already-imbalanced full set (which is fragile - too strong a weight
    causes the model to over-predict bonds everywhere, too weak causes it to
    under-predict; sampling avoids needing to tune that tradeoff at all).

    Fully vectorized (no python loop with .item() per molecule) using an
    argsort-rank-threshold trick, since torch.topk needs a fixed k but each
    molecule has a different number of real bonds.
    """
    batch_size, max_atoms, _ = bond_targets.shape
    device = bond_targets.device

    eye = torch.eye(max_atoms, dtype=torch.bool, device=device).unsqueeze(0)  # [1, max_atoms, max_atoms]
    valid_pair_mask = pair_mask.bool() & (~eye)  # exclude diagonal and padding pairs

    is_bonded = (bond_targets != no_bond_idx) & valid_pair_mask
    is_candidate_negative = (bond_targets == no_bond_idx) & valid_pair_mask

    # how many negatives to sample per molecule = that molecule's own positive count
    counts = is_bonded.view(batch_size, -1).sum(dim=1)  # [batch_size], vectorized, no .item()

    # assign random priority to every candidate negative pair, -1 sentinel elsewhere
    rand_scores = torch.rand(batch_size, max_atoms, max_atoms, device=device)
    rand_scores = rand_scores.masked_fill(~is_candidate_negative, -1.0)
    flat_scores = rand_scores.view(batch_size, -1)

    # sort descending: the first `counts[g]` entries per row are this molecule's
    # sampled negatives (any entry with score -1 means "not a real candidate",
    # filtered out via the score > -1 check below)
    sorted_vals, sorted_idx = flat_scores.sort(dim=1, descending=True)
    rank = torch.arange(flat_scores.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
    selected = (rank < counts.unsqueeze(1)) & (sorted_vals > -1.0)

    neg_mask_flat = torch.zeros_like(flat_scores, dtype=torch.bool)
    neg_mask_flat.scatter_(1, sorted_idx, selected)
    neg_mask = neg_mask_flat.view(batch_size, max_atoms, max_atoms)

    loss_mask = is_bonded | neg_mask
    return loss_mask


def vae_loss(atom_logits, bond_logits, mu, logvar, data_batch, kl_weight=1.0):
    """
    atom_logits: [batch, max_atoms, NUM_ATOM_TYPES + 1]  (+1 for padding class)
    bond_logits: [batch, max_atoms, max_atoms, NUM_BOND_TYPES]
    mu, logvar:  [batch, latent_dim]

    Bond class imbalance (~95%+ pairs are "no bond") is handled here via
    NEGATIVE SAMPLING rather than class-weighting: for each molecule, we use
    all its real bonded pairs plus an equal number of randomly sampled
    "no bond" pairs, so the loss only ever sees a balanced 1:1 set - no
    weight hyperparameter to tune, and no risk of over/under-correcting.
    """
    atom_targets, bond_targets, atom_mask = build_dense_targets(data_batch)

    # --- Atom reconstruction loss (cross-entropy per atom slot) ---
    atom_loss = F.cross_entropy(
        atom_logits.reshape(-1, atom_logits.size(-1)),
        atom_targets.reshape(-1),
        reduction="none",
    )
    atom_loss = atom_loss.reshape(atom_targets.shape)
    real_atom_loss = (atom_loss * atom_mask).sum() / atom_mask.sum().clamp(min=1)

    # --- Bond reconstruction loss (cross-entropy per atom pair), negative-sampled ---
    pair_mask = atom_mask.unsqueeze(2) * atom_mask.unsqueeze(1)  # [batch, max_atoms, max_atoms]
    loss_mask = build_negative_sampling_mask(bond_targets, pair_mask, NO_BOND_IDX).float()

    bond_loss = F.cross_entropy(
        bond_logits.reshape(-1, bond_logits.size(-1)),
        bond_targets.reshape(-1),
        reduction="none",
    )
    bond_loss = bond_loss.reshape(bond_targets.shape)
    real_bond_loss = (bond_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

    recon_loss = real_atom_loss + real_bond_loss

    # --- KL divergence: encourages mu/logvar -> N(0, I) ---
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = kl_div.mean()

    total_loss = recon_loss + kl_weight * kl_div
    return total_loss, real_atom_loss.item(), real_bond_loss.item(), kl_div.item()