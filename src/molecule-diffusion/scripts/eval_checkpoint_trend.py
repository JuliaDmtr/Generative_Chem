"""
Build a real epoch-vs-validity trend for the phase-1 training run
(epochs 0-149), using the untouched historical checkpoints that were
saved with distinct filenames (unlike the later v2 run, which overwrote
a single --out path each time).

For each checkpoint: samples --n-samples molecules under the
PARP-conditioned (COND_POSITIVE) label, with sizes drawn from the real
PARP conformer size distribution (same recipe as generate.py
--size-dataset-npz), using a fixed seed so every checkpoint is evaluated
on directly comparable molecule sizes. Reports:
    - valid_pct:  RDKit bond-perception + sanitization success rate
    - single_pct: subset of valid_pct that is also one connected molecule
    - ring_histogram: ring-size counts pooled across all valid molecules

This is a validity/quality *metric*, not a loss — there is no held-out
denoising loss computed here, so don't conflate the resulting plot's
right-hand axis with "validation loss".

Usage (run from src/molecule-diffusion/scripts/):
    python eval_checkpoint_trend.py
    python eval_checkpoint_trend.py --n-samples 60 --out-png ../checkpoints/phase1_trend.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from dataset import QM9Dataset, sample_molecule_sizes
from diffusion import EquivariantMoleculeDiffusion
from evaluate import mol_from_coords, n_fragments
from generate import molecules_from_output
from utils import COND_POSITIVE, condition_labels_to_tensor

CHECKPOINTS = [
    (25, "../checkpoints/conditional_checkpoint_epoch_25.pt"),
    (50, "../checkpoints/conditional_checkpoint_epoch_50.pt"),
    (75, "../checkpoints/conditional_checkpoint_epoch_75.pt"),
    (100, "../checkpoints/conditional_checkpoint_epoch_100.pt"),
    (125, "../checkpoints/conditional_checkpoint_epoch_125.pt"),
    (149, "../checkpoints/conditional_checkpoint_epoch_149.pt"),
]

# The old in-training diagnostics for these same checkpoints, transcribed
# from the phase-1 run's terminal output. Only n=16 samples each (6.25%
# resolution per sample) — kept here for reference/comparison, not as a
# trustworthy trend on their own.
OLD_N16_DIAGNOSTICS = {
    25: {"null": 0.000, "positive": 0.062},
    50: {"null": 0.125, "positive": 0.062},
    75: {"null": 0.000, "positive": 0.062},
    100: {"null": 0.062, "positive": 0.000},
    125: {"null": 0.000, "positive": 0.000},
    149: {"null": 0.000, "positive": 0.062},
}


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, sizes, n_steps, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    margs = ckpt["args"]
    model = EquivariantMoleculeDiffusion(
        hidden_dim=margs["hidden_dim"], n_layers=margs["n_layers"],
        timesteps=margs["timesteps"], cond_dim=margs.get("cond_dim", 0),
    ).to(device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()

    n_samples = len(sizes)
    n_max = int(max(sizes))
    node_mask = torch.zeros(n_samples, n_max, 1, device=device)
    for i, n in enumerate(sizes):
        node_mask[i, :int(n)] = 1.0
    cond = condition_labels_to_tensor([COND_POSITIVE] * n_samples, device=device)

    x_t, h_t = model.sample(node_mask, cond=cond, n_steps=n_steps, device=device)
    molecules = molecules_from_output(x_t, h_t, node_mask)
    mols = [mol_from_coords(symbols, coords) for symbols, coords in molecules]

    n_valid = sum(1 for m in mols if m is not None)
    n_single = sum(1 for m in mols if m is not None and n_fragments(m) == 1)
    ring_hist: dict[int, int] = {}
    for m in mols:
        if m is None:
            continue
        for r in m.GetRingInfo().AtomRings():
            ring_hist[len(r)] = ring_hist.get(len(r), 0) + 1

    return {
        "n_samples": n_samples,
        "valid_pct": n_valid / n_samples,
        "single_pct": n_single / n_samples,
        "ring_histogram": ring_hist,
    }


def load_training_log(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def make_plot(training_log, results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    epochs = [r["epoch"] for r in training_log]
    ax1.plot(epochs, [r["loss"] for r in training_log], label="total loss", color="black")
    ax1.plot(epochs, [r["loss_x"] for r in training_log], label="loss_x (position)", color="tab:blue")
    ax1.plot(epochs, [r["loss_h"] for r in training_log], label="loss_h (atom type)", color="tab:orange")
    ax1.set_ylabel("training loss")
    ax1.set_title("Phase 1 (epochs 0-149): training loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ckpt_epochs = sorted(results.keys())
    valid_pct = [results[e]["valid_pct"] * 100 for e in ckpt_epochs]
    single_pct = [results[e]["single_pct"] * 100 for e in ckpt_epochs]
    ax2.plot(ckpt_epochs, valid_pct, "o-", label=f"valid % (n={results[ckpt_epochs[0]]['n_samples']}, fresh eval)", color="tab:green")
    ax2.plot(ckpt_epochs, single_pct, "o-", label="single-molecule %", color="tab:red")

    old_epochs = sorted(OLD_N16_DIAGNOSTICS.keys())
    old_pos = [OLD_N16_DIAGNOSTICS[e]["positive"] * 100 for e in old_epochs]
    ax2.plot(old_epochs, old_pos, "x--", label="old n=16 diagnostic (noisy, for reference)", color="gray", alpha=0.6)

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("generation validity (%)")
    ax2.set_title("Phase 1: PARP-conditioned generation validity (not a loss)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parp-npz", type=str, default="../parp_3d.npz")
    p.add_argument("--training-log-csv", type=str, default="../training_log_phase1.csv")
    p.add_argument("--n-samples", type=int, default=40)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=57)
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out-json", type=str, default="../checkpoints/phase1_trend.json")
    p.add_argument("--out-png", type=str, default="../checkpoints/phase1_trend.png")
    args = p.parse_args()

    device = torch.device(args.device)
    # Fixed seed -> identical sizes list reused for every checkpoint, so
    # differences in the result are due to the model, not sampled sizes.
    sizes = sample_molecule_sizes(QM9Dataset(args.parp_npz), args.n_samples, seed=args.seed)

    results = {}
    for epoch, path in CHECKPOINTS:
        print(f"evaluating epoch {epoch} ({path}) ...")
        res = evaluate_checkpoint(path, sizes, args.n_steps, device)
        results[epoch] = res
        print(f"   epoch {epoch:4d} | valid {res['valid_pct']:.1%} | single {res['single_pct']:.1%} "
              f"| rings {res['ring_histogram']}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out_json}")

    training_log = load_training_log(args.training_log_csv)
    make_plot(training_log, results, args.out_png)


if __name__ == "__main__":
    main()
