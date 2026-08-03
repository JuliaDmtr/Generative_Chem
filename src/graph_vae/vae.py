
"""
vae.py
 
Graph VAE: encodes a molecule graph into a 128-dim latent vector,
and decodes a latent vector back into atom-type / bond predictions.
 
Encoder: GCNConv layers -> global mean pooling -> (mu, logvar)
Decoder: MLP that expands the latent vector back into
         - per-atom type logits (for a fixed max number of atoms)
         - a bond-type prediction for every atom pair (dense adjacency style)
 
NOTE ON DECODER DESIGN:
Molecules have a variable number of atoms, but a plain MLP decoder needs a
fixed-size output. The standard trick (used in early graph-VAE papers, e.g.
GraphVAE) is to decode into a fixed MAX_ATOMS x MAX_ATOMS dense adjacency/
bond-type tensor and a MAX_ATOMS x ATOM_TYPES atom-type tensor, then mask
out the unused atom slots during loss computation and generation.
This is the simplest correct starting point; more advanced decoders
(autoregressive, iterative graph-building) can replace this later.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
 
from graph_vae.smiles_to_graph import ATOM_TYPES, BOND_TYPES, MAX_ATOMS
 
NODE_FEAT_DIM = len(ATOM_TYPES) + 3   # atom type one-hot + degree + charge + aromatic
NUM_ATOM_TYPES = len(ATOM_TYPES)
NUM_BOND_TYPES = len(BOND_TYPES) + 1  # +1 for "no bond" class                       # covers the vast majority of drug-like ChEMBL molecules
LATENT_DIM = 128
 
 
class Encoder(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, hidden_dim=256, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
 
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
 
    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
 
        # pool all atom-level embeddings in each molecule into one vector
        h_graph = global_mean_pool(h, batch)  # [num_molecules_in_batch, hidden_dim]
 
        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        return mu, logvar
 
 
def reparameterize(mu, logvar):
    """
    The reparameterization trick: sample z = mu + std * eps,
    where eps ~ N(0,1). This keeps sampling differentiable so
    gradients can flow back through mu/logvar during training.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
 
 
class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=512,
                 max_atoms=MAX_ATOMS, num_atom_types=NUM_ATOM_TYPES,
                 num_bond_types=NUM_BOND_TYPES):
        super().__init__()
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
 
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
 
        # atom-type logits for every atom slot
        # +1 class reserved for "padding / no atom" (see vae_loss.py PAD_ATOM_IDX)
        self.atom_head = nn.Linear(hidden_dim, max_atoms * (num_atom_types + 1))
 
        # bond-type logits for every pair of atom slots (dense, symmetric target)
        self.bond_head = nn.Linear(hidden_dim, max_atoms * max_atoms * num_bond_types)
 
    def forward(self, z):
        h = self.shared(z)
        batch_size = z.size(0)
 
        atom_logits = self.atom_head(h).view(
            batch_size, self.max_atoms, self.num_atom_types + 1
        )
        bond_logits = self.bond_head(h).view(
            batch_size, self.max_atoms, self.max_atoms, self.num_bond_types
        )
        return atom_logits, bond_logits
 
 
class GraphVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
 
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encoder(x, edge_index, batch)
        z = reparameterize(mu, logvar)
        atom_logits, bond_logits = self.decoder(z)
        return atom_logits, bond_logits, mu, logvar
 
