"""
Train the E(3)-equivariant molecule diffusion model with PARP-inhibitor
conditioning, combined with the base validity-fix work (bigger model,
more training, atom-count capping) in a single run.

This is a 3-way *class*-conditional model (simpler than textbook
classifier-free-guidance dropout), with a one-hot condition vector of
dim COND_DIM (see utils.py):
    - COND_POSITIVE: a real PARP-inhibitor molecule (from parp_3d.npz)
    - COND_NEGATIVE: a generic ChEMBL molecule, size-matched (+/- a few
      atoms) to a PARP inhibitor in the same batch, so the model can't
      trivially tell the classes apart by size alone
    - COND_NULL: a representative random ChEMBL molecule (not size-matched
      — this is the background/unconditional distribution), used for the
      null branch so `generate.py --condition null` and
      `--guidance-scale` (classifier-free-guidance-style extrapolation)
      have something meaningful to sample from/against.

Every epoch:
  1. A *balanced* stream is rebuilt: every PARP conformer (label=positive)
     paired with a freshly-sampled size-matched ChEMBL molecule
     (label=negative). No condition dropout here — labels are exact.
  2. A *representative* stream of plain random ChEMBL molecules
     (label=null) of the same size as the balanced stream, capped to
     --max-atoms molecules only (EGNN is O(N^2) dense — large ChEMBL
     outliers are excluded to keep training tractable).
  3. Both streams are concatenated into one shuffled batch stream per
     epoch and trained on jointly.

Every --diag-every epochs, in-memory validity diagnostics are run (via
evaluate.mol_from_coords, no file I/O) separately for the null and
positive conditions, so validity-vs-condition curves can be inspected
without waiting for the full run to finish.

Usage:
    python train_conditional.py --parp-npz parp_3d.npz --chembl-npz ../chembl_3d.npz \
        --epochs 300 --hidden-dim 192 --n-layers 6 --out conditional_ckpt.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import QM9Dataset, collate_molecules
from diffusion import EquivariantMoleculeDiffusion
from evaluate import mol_from_coords, n_fragments
from generate import molecules_from_output
from heavy_atom_eval import heavy_atom_checks
from utils import (ARCH_VERSION, COND_DIM, COND_NEGATIVE, COND_NULL, COND_POSITIVE, EMA, check_arch_version,
                   condition_labels_to_tensor)


class LabeledMoleculeList(Dataset):
    """A plain in-memory list of {'pos', 'types', 'n_nodes', 'label'} dicts,
    rebuilt fresh each epoch (see build_epoch_items).
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_with_cond(batch, max_atoms=None):
    x0, h0, node_mask = collate_molecules(batch, max_atoms=max_atoms)
    labels = [b["label"] for b in batch]
    cond = condition_labels_to_tensor(labels)
    return x0, h0, node_mask, cond


def match_negative_index(target_size, size_tolerance, size_to_indices, filtered_sizes,
                          filtered_indices, rng, max_tol_multiplier=8):
    """Find a random ChEMBL molecule (global index into chembl_ds) whose
    atom count is within +/- tol of target_size, widening tol if nothing
    matches. Falls back to the single nearest-size molecule in the whole
    filtered pool if even the widest window is empty (should be rare).
    """
    tol = size_tolerance
    while tol <= size_tolerance * max_tol_multiplier:
        candidates = [idx for size, idxs in size_to_indices.items()
                      if abs(size - target_size) <= tol for idx in idxs]
        if candidates:
            return int(rng.choice(candidates))
        tol *= 2
    diffs = np.abs(filtered_sizes - target_size)
    return int(filtered_indices[np.argmin(diffs)])


def build_epoch_items(parp_ds, parp_sizes, chembl_ds, filtered_indices, filtered_sizes,
                       size_to_indices, rng, size_tolerance, n_representative):
    items = []
    for i in range(len(parp_ds)):
        pos_item = parp_ds[i]
        items.append({**pos_item, "label": COND_POSITIVE})

        neg_global_idx = match_negative_index(
            parp_sizes[i], size_tolerance, size_to_indices, filtered_sizes, filtered_indices, rng,
        )
        neg_item = chembl_ds[neg_global_idx]
        items.append({**neg_item, "label": COND_NEGATIVE})

    rep_global_idx = rng.choice(filtered_indices, size=n_representative, replace=True)
    for gi in rep_global_idx:
        item = chembl_ds[int(gi)]
        items.append({**item, "label": COND_NULL})

    order = rng.permutation(len(items))
    return [items[i] for i in order]


@torch.no_grad()
def reset_optimizer_lr(optimizer, lr: float) -> None:
    """Force every param group to `lr`.

    Call this after loading a checkpoint's optimizer state and BEFORE building
    the LR scheduler: schedulers capture base_lrs from the optimizer at
    construction, so a restored LR would silently override --lr for the whole
    resumed run (see tests/test_resume_lr.py).
    """
    for group in optimizer.param_groups:
        group["lr"] = lr


def selection_margin(rate_a, rate_b, n_samples):
    """95% noise floor on the difference between two diagnostic rates.

    Both are binomial rates over the same number of sampled molecules, so the
    difference has standard error sqrt(p(1-p)/n + q(1-q)/n).
    """
    n = max(int(n_samples), 1)
    var = (max(rate_a, 0.0) * (1.0 - max(rate_a, 0.0)) +
           max(rate_b, 0.0) * (1.0 - max(rate_b, 0.0))) / n
    return 1.96 * math.sqrt(max(var, 0.0))


def should_replace_best(best_rate, candidate_rate, n_samples):
    """Should this epoch replace the checkpoint kept as "best"?

    Keep the LATER checkpoint unless the diagnostic says it is significantly
    worse than the best rate seen so far. Selecting on the raw rate picks
    noise: at the default 64 molecules its 95% interval is about +/-8 points,
    while real differences between late epochs are 1-2 points. In v11 epochs
    430, 450 and 490 all scored exactly 89.1% and a strict `>` kept the
    earliest, which a 256-sample evaluation then showed to be the worst of the
    three (66.4% usable molecules against 68.0% for the final epoch).

    Training loss is still falling at the end of these runs, so a later
    checkpoint is the better prior; the diagnostic only overrides it when the
    drop is larger than its own noise floor. Raising --diag-n-samples shrinks
    that floor and makes the rule more discriminating.
    See tests/test_best_checkpoint_selection.py, which replays the real v11
    history.
    """
    if best_rate is None:
        return True
    return candidate_rate >= best_rate - selection_margin(best_rate, candidate_rate, n_samples)


def run_validity_diagnostics(model, condition_label, sizes_pool, n_samples, n_steps, device, rng,
                              no_hydrogens: bool = False):
    """Sample n_samples molecules under a fixed condition label, sizes
    drawn from sizes_pool, and report (validity_rate, single_molecule_rate)
    — both in-memory (no file I/O).

    With explicit hydrogens (no_hydrogens=False): validity_rate is the plain
    RDKit sanitization rate (bond perception + valence rules);
    single_molecule_rate is the stricter subset of those that also form one
    connected molecule rather than 2+ disconnected fragments (RDKit's
    SanitizeMol doesn't require connectivity, so a "valid" result can still
    be several disjoint pieces — e.g. a SMILES containing '.').

    On heavy-atom-only data (no_hydrogens=True) that pipeline cannot be used
    at all: RDKit's DetermineBonds assigns bond orders by balancing valences
    against the explicit hydrogens, so without them it fails on essentially
    every molecule and would report ~0% regardless of quality. Instead the
    heavy-atom checks are used (see heavy_atom_eval.py, calibrated to 100% on
    real stripped conformers): validity_rate = valences physically possible,
    single_molecule_rate = that AND one connected component ("plausible").
    The two modes are NOT comparable to each other.
    """
    sizes = rng.choice(sizes_pool, size=n_samples, replace=True)
    n_max = int(max(sizes))
    node_mask = torch.zeros(n_samples, n_max, 1, device=device)
    for i, n in enumerate(sizes):
        node_mask[i, :int(n)] = 1.0
    cond = condition_labels_to_tensor([condition_label] * n_samples, device=device)

    x_t, h_t = model.sample(node_mask, cond=cond, n_steps=n_steps, device=device)
    molecules = molecules_from_output(x_t, h_t, node_mask)
    if no_hydrogens:
        checks = [heavy_atom_checks(symbols, coords) for symbols, coords in molecules]
        n_valid = sum(1 for c in checks if c["valence_ok"])
        n_single = sum(1 for c in checks if c["plausible"])
    else:
        mols = [mol_from_coords(symbols, coords) for symbols, coords in molecules]
        n_valid = sum(1 for m in mols if m is not None)
        n_single = sum(1 for m in mols if m is not None and n_fragments(m) == 1)
    return n_valid / len(molecules), n_single / len(molecules)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parp-npz", type=str, default="parp_3d.npz")
    p.add_argument("--chembl-npz", type=str, default="chembl_3d.npz")
    p.add_argument("--max-atoms", type=int, default=80,
                    help="Cap ChEMBL molecules to this many atoms (incl. H) — EGNN is dense O(N^2).")
    p.add_argument("--size-tolerance", type=int, default=5,
                    help="Initial +/- atom-count window used to size-match ChEMBL negatives to PARP positives.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-min", type=float, default=1e-5,
                    help="Final LR at the end of this invocation's cosine decay schedule "
                         "(runs over --epochs epochs, so a --resume run gets its own fresh anneal).")
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="conditional_checkpoint.pt")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint to initialize model/EMA/optimizer from and continue training.")
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=25,
                    help="Also overwrite --out with a checkpoint every N epochs (0 disables), "
                         "so a long run isn't lost if interrupted. The final epoch is always saved.")
    p.add_argument("--diag-every", type=int, default=25, help="Epochs between validity diagnostics (0 disables).")
    p.add_argument("--diag-n-samples", type=int, default=64)
    p.add_argument("--diag-n-steps", type=int, default=200, help="Diffusion steps used for diagnostic sampling (fewer = faster).")
    p.add_argument("--no-hydrogens", action="store_true",
                    help="The datasets contain heavy atoms only (built with strip_hydrogens.py): "
                         "use the heavy-atom validity checks for the periodic diagnostics, since the "
                         "with-H RDKit pipeline reports ~0%% on such data regardless of quality.")
    p.add_argument("--angle-loss-weight", type=float, default=0.0,
                    help="Weight for the auxiliary angle-consistency penalty (0 disables it, "
                         "fully backward compatible). Penalizes near-60-degree angles among each "
                         "atom's nearest predicted neighbors -- targets the strained 3-membered-ring "
                         "geometry diagnose_scale.py's angle diagnostic confirmed. See "
                         "diffusion.angle_consistency_penalty() for the exact mechanism.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    print("Device:", device)

    parp_ds = QM9Dataset(args.parp_npz)
    chembl_ds = QM9Dataset(args.chembl_npz)
    parp_sizes = np.array([len(p) for p in parp_ds.positions])
    chembl_sizes = np.array([len(p) for p in chembl_ds.positions])

    filtered_indices = np.where(chembl_sizes <= args.max_atoms)[0]
    filtered_sizes = chembl_sizes[filtered_indices]
    if len(filtered_indices) == 0:
        raise ValueError(f"No ChEMBL molecules with <= {args.max_atoms} atoms; raise --max-atoms.")
    print(f"PARP conformers: {len(parp_ds):,} (sizes {parp_sizes.min()}-{parp_sizes.max()})")
    print(f"ChEMBL pool: {len(chembl_ds):,} total, {len(filtered_indices):,} within "
          f"--max-atoms={args.max_atoms} (sizes {filtered_sizes.min()}-{filtered_sizes.max()})")

    size_to_indices: dict[int, list[int]] = {}
    for global_idx, size in zip(filtered_indices, filtered_sizes):
        size_to_indices.setdefault(int(size), []).append(int(global_idx))

    n_representative = 2 * len(parp_ds)  # match the balanced stream's size

    model = EquivariantMoleculeDiffusion(
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, timesteps=args.timesteps, cond_dim=COND_DIM,
        angle_loss_weight=args.angle_loss_weight,
    ).to(device)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-12)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        check_arch_version(ckpt)
        model.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema_state"])
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        else:
            print("   [resume] checkpoint has no optimizer_state (older format) — starting Adam moments fresh.")
        # the restored state carries the LR the previous run ended on; --lr wins
        reset_optimizer_lr(opt, args.lr)
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from {args.resume} (checkpoint epoch {ckpt.get('epoch')}); "
              f"continuing as global epoch {start_epoch}, running {args.epochs} more epochs "
              f"with LR annealed {args.lr:g} -> {args.lr_min:g} over this invocation.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)

    def build_checkpoint_payload(global_epoch):
        return {
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": opt.state_dict(),
            "args": vars(args) | {"cond_dim": COND_DIM},
            "epoch": global_epoch,
            "arch_version": ARCH_VERSION,
        }

    def save_checkpoint(global_epoch, also_tag=False):
        # Always overwrite args.out (the "latest" pointer, used by --resume
        # and generate.py by default). When also_tag is set, ALSO write a
        # distinct epoch-tagged file that is never overwritten again — this
        # is what lets a later run rebuild a real epoch-vs-validity trend
        # (see eval_checkpoint_trend.py), unlike periodic saves that all
        # clobbered the same --out path in earlier runs.
        payload = build_checkpoint_payload(global_epoch)
        torch.save(payload, args.out)
        if also_tag:
            base, ext = os.path.splitext(args.out)
            tagged_path = f"{base}_epoch_{global_epoch}{ext}"
            torch.save(payload, tagged_path)
            return tagged_path
        return None

    # Keeps one checkpoint chosen by the PARP-conditioned single-molecule rate
    # (the stricter diagnostic metric — one fully connected, chemically valid
    # molecule, not just RDKit-sanitized fragments), tracked independently of
    # --ckpt-every. The rate is noisy at --diag-n-samples molecules, so the
    # rule is "keep the later epoch unless it is significantly worse": see
    # should_replace_best(). best_pos_single is the best rate ever observed,
    # which is what a candidate is judged against; kept_epoch/kept_rate
    # describe the checkpoint actually on disk, and the two can differ.
    best_pos_single = None
    best_path = None
    kept_epoch, kept_rate = None, None

    for epoch in range(args.epochs):
        global_epoch = start_epoch + epoch
        t0 = time.time()
        items = build_epoch_items(
            parp_ds, parp_sizes, chembl_ds, filtered_indices, filtered_sizes,
            size_to_indices, rng, args.size_tolerance, n_representative,
        )
        # Each batch is padded only to its own largest molecule: the model is
        # padding invariant, and compute grows with N^2.
        loader = DataLoader(
            LabeledMoleculeList(items), batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_with_cond,
        )

        running = {"loss": 0.0, "loss_x": 0.0, "loss_h": 0.0, "loss_angle": 0.0, "n": 0}
        for x0, h0, node_mask, cond in loader:
            x0, h0, node_mask, cond = x0.to(device), h0.to(device), node_mask.to(device), cond.to(device)

            loss, parts = model.loss(x0, h0, node_mask, cond=cond)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model)

            running["loss"] += loss.item()
            running["loss_x"] += parts["loss_x"]
            running["loss_h"] += parts["loss_h"]
            running["loss_angle"] += parts.get("loss_angle", 0.0)
            running["n"] += 1

        scheduler.step()

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            n = max(running["n"], 1)
            angle_part = f", angle {running['loss_angle']/n:.4f}" if args.angle_loss_weight > 0 else ""
            print(f"epoch {global_epoch:4d} | loss {running['loss']/n:.4f} "
                  f"(x {running['loss_x']/n:.4f}, h {running['loss_h']/n:.4f}{angle_part}) "
                  f"| lr {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if args.diag_every and (epoch % args.diag_every == 0 or epoch == args.epochs - 1) and epoch > 0:
            # Sample from the EMA shadow weights, not the raw model — EMA is
            # what generate.py uses by default, and is far less noisy than
            # the just-updated raw weights, especially mid-training.
            null_validity, null_single = run_validity_diagnostics(
                ema.shadow, COND_NULL, filtered_sizes, args.diag_n_samples, args.diag_n_steps, device, rng,
                no_hydrogens=args.no_hydrogens,
            )
            pos_validity, pos_single = run_validity_diagnostics(
                ema.shadow, COND_POSITIVE, parp_sizes, args.diag_n_samples, args.diag_n_steps, device, rng,
                no_hydrogens=args.no_hydrogens,
            )
            print(f"   [diagnostics] validity (single-molecule) — "
                  f"null: {null_validity:.1%} ({null_single:.1%}), "
                  f"PARP-conditioned: {pos_validity:.1%} ({pos_single:.1%})")

            if should_replace_best(best_pos_single, pos_single, args.diag_n_samples):
                base, ext = os.path.splitext(args.out)
                best_path = f"{base}_best{ext}"
                torch.save(build_checkpoint_payload(global_epoch), best_path)
                kept_epoch, kept_rate = global_epoch, pos_single
                how = ("first diagnostic" if best_pos_single is None
                       else f"within {selection_margin(best_pos_single, pos_single, args.diag_n_samples):.1%} "
                            f"of the best seen ({best_pos_single:.1%})")
                print(f"   [checkpoint] keeping epoch {global_epoch} as best "
                      f"({pos_single:.1%}, {how}) — saved to {best_path}")
            else:
                print(f"   [checkpoint] epoch {global_epoch} ({pos_single:.1%}) is significantly "
                      f"below the best seen ({best_pos_single:.1%}); keeping epoch {kept_epoch}")
            best_pos_single = pos_single if best_pos_single is None else max(best_pos_single, pos_single)

        if args.ckpt_every and epoch > 0 and (epoch % args.ckpt_every == 0):
            tagged = save_checkpoint(global_epoch, also_tag=True)
            print(f"   [checkpoint] saved epoch {global_epoch} to {args.out} (and {tagged})")

    final_tagged = save_checkpoint(start_epoch + args.epochs - 1, also_tag=True)
    print(f"Saved final checkpoint to {args.out} (and {final_tagged})")
    if best_path:
        print(f"Kept epoch {kept_epoch} as {best_path} (diagnostic {kept_rate:.1%}; "
              f"best rate seen during the run was {best_pos_single:.1%}).")
        print(f"The diagnostic samples only {args.diag_n_samples} molecules at "
              f"{args.diag_n_steps} steps, so differences below roughly "
              f"{selection_margin(best_pos_single, best_pos_single, args.diag_n_samples):.0%} "
              f"are noise and it cannot rank late epochs. It is used to reject a collapsed "
              f"epoch, not to pick a winner: for that, evaluate the last few tagged "
              f"checkpoints with scripts/compare_chemistry.py (256 samples, 500 steps).")


if __name__ == "__main__":
    main()
