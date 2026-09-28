"""
Build a real epoch-vs-validity trend across BOTH training phases run so
far, using every checkpoint that was saved with a distinct filename:
  - Phase 1 (epochs 0-149): the original untouched historical checkpoints.
  - Phase 2 / "v2" (epochs 301-399): checkpoints saved after fixing the
    save_checkpoint(also_tag=True) bug (see train_conditional.py) that
    previously let every periodic save clobber the same --out path.
Epochs 150-300 have NO surviving checkpoint or training-log history —
that run predates both fixes (unbuffered logging + tagged checkpoints)
and only its final overwritten checkpoint (epoch 300) survived, which is
what phase 2 resumed from. The plot below shows this as a real gap
rather than papering over it with an interpolated line.

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
    python eval_checkpoint_trend.py --n-samples 60 --out-png ../checkpoints/full_trend.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

import numpy as np
import torch

from dataset import QM9Dataset, sample_molecule_sizes
from diffusion import EquivariantMoleculeDiffusion
from evaluate import mol_from_coords, n_fragments
from generate import molecules_from_output
from utils import COND_POSITIVE, condition_labels_to_tensor

PHASE1_CHECKPOINTS = [
    (25, "../checkpoints/conditional_checkpoint_epoch_25.pt"),
    (50, "../checkpoints/conditional_checkpoint_epoch_50.pt"),
    (75, "../checkpoints/conditional_checkpoint_epoch_75.pt"),
    (100, "../checkpoints/conditional_checkpoint_epoch_100.pt"),
    (125, "../checkpoints/conditional_checkpoint_epoch_125.pt"),
    (149, "../checkpoints/conditional_checkpoint_epoch_149.pt"),
]

PHASE2_CHECKPOINTS = [
    (326, "../checkpoints/conditional_checkpoint_v2_epoch_326.pt"),
    (351, "../checkpoints/conditional_checkpoint_v2_epoch_351.pt"),
    (376, "../checkpoints/conditional_checkpoint_v2_epoch_376.pt"),
    (399, "../checkpoints/conditional_checkpoint_v2_epoch_399.pt"),
]

CHECKPOINTS = PHASE1_CHECKPOINTS + PHASE2_CHECKPOINTS

# The old in-training diagnostics for the phase-1 checkpoints, transcribed
# from the phase-1 run's terminal output. Only n=16 samples each (6.25%
# resolution per sample) — kept here for reference/comparison, not as a
# trustworthy trend on their own. Phase 2 has no equivalent because its
# own in-training diagnostics (n=64, printed live to train_v2_log.txt)
# are already large enough to trust more than phase 1's n=16 ones.
OLD_N16_DIAGNOSTICS = {
    25: {"null": 0.000, "positive": 0.062},
    50: {"null": 0.125, "positive": 0.062},
    75: {"null": 0.000, "positive": 0.062},
    100: {"null": 0.062, "positive": 0.000},
    125: {"null": 0.000, "positive": 0.000},
    149: {"null": 0.000, "positive": 0.062},
}


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, sizes, n_steps, device, seed):
    # Reset torch's global RNG identically before every checkpoint so the
    # reverse-diffusion sampler's own injected noise (torch.randn calls in
    # diffusion.py: sample, not covered by the numpy seed used for `sizes`)
    # is IDENTICAL across checkpoints too -- otherwise differences between
    # checkpoints are confounded with differences in sampling noise, and
    # results aren't even reproducible run-to-run.
    torch.manual_seed(seed)
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


_LOG_LINE_RE = re.compile(
    r"epoch\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\(x\s+([\d.]+),\s*h\s+([\d.]+)\)"
)


def parse_training_log(log_path):
    """Parse 'epoch NNN | loss L (x LX, h LH) | ...' lines directly out of a
    raw train_conditional.py stdout log (e.g. train_v2_log.txt). Unlike
    training_log_phase1.csv (manually transcribed from a chat-pasted log,
    because the original run's own log file was empty due to output
    buffering), this reads the real file directly — no transcription step,
    no risk of losing this data again.
    """
    rows = []
    with open(log_path) as f:
        for line in f:
            m = _LOG_LINE_RE.search(line)
            if m:
                epoch, loss, loss_x, loss_h = m.groups()
                rows.append({"epoch": float(epoch), "loss": float(loss),
                             "loss_x": float(loss_x), "loss_h": float(loss_h)})
    return rows


def write_training_log_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "loss_x", "loss_h"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plot(phase1_log, phase2_log, results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Plot phase 1 and phase 2 as separate line segments (not a single
    # continuous line) so the untracked epochs 150-300 show up as a real
    # gap rather than a misleading interpolated line.
    for log, label_suffix in ((phase1_log, " (phase 1)"), (phase2_log, " (phase 2)")):
        if not log:
            continue
        epochs = [r["epoch"] for r in log]
        ax1.plot(epochs, [r["loss"] for r in log], color="black",
                  label="total loss" + label_suffix if log is phase1_log else None)
        ax1.plot(epochs, [r["loss_x"] for r in log], color="tab:blue",
                  label="loss_x (position)" + label_suffix if log is phase1_log else None)
        ax1.plot(epochs, [r["loss_h"] for r in log], color="tab:orange",
                  label="loss_h (atom type)" + label_suffix if log is phase1_log else None)
    ax1.axvspan(149, 301, color="gray", alpha=0.15)
    ax1.set_ylabel("training loss")
    ax1.set_title("Training loss (epochs 150-300 untracked: buffering + checkpoint-overwrite bugs, shaded)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ckpt_epochs = sorted(results.keys())
    phase1_epochs = [e for e in ckpt_epochs if e <= 149]
    phase2_epochs = [e for e in ckpt_epochs if e > 149]
    for epochs, label_suffix in ((phase1_epochs, ""), (phase2_epochs, "")):
        if not epochs:
            continue
        valid_pct = [results[e]["valid_pct"] * 100 for e in epochs]
        single_pct = [results[e]["single_pct"] * 100 for e in epochs]
        ax2.plot(epochs, valid_pct, "o-", color="tab:green",
                  label=f"valid % (n={results[epochs[0]]['n_samples']}, fresh eval)" if epochs is phase1_epochs else None)
        ax2.plot(epochs, single_pct, "o-", color="tab:red",
                  label="single-molecule %" if epochs is phase1_epochs else None)
    ax2.axvspan(149, 301, color="gray", alpha=0.15)

    old_epochs = sorted(OLD_N16_DIAGNOSTICS.keys())
    old_pos = [OLD_N16_DIAGNOSTICS[e]["positive"] * 100 for e in old_epochs]
    ax2.plot(old_epochs, old_pos, "x--", label="old n=16 diagnostic (phase 1, noisy, for reference)", color="gray", alpha=0.6)

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("generation validity (%)")
    ax2.set_title("PARP-conditioned generation validity across both training phases (not a loss)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parp-npz", type=str, default="../parp_3d.npz")
    p.add_argument("--phase1-log-csv", type=str, default="../training_log_phase1.csv")
    p.add_argument("--phase2-raw-log", type=str, default="../train_v2_log.txt",
                    help="Raw train_conditional.py stdout log to parse directly (no manual transcription needed).")
    p.add_argument("--phase2-log-csv", type=str, default="../training_log_v2.csv",
                    help="Where to persist the parsed phase-2 log, so it survives even if --phase2-raw-log is later deleted.")
    p.add_argument("--n-samples", type=int, default=40)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=57)
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out-json", type=str, default="../checkpoints/full_trend.json")
    p.add_argument("--out-png", type=str, default="../checkpoints/full_trend.png")
    args = p.parse_args()

    device = torch.device(args.device)
    # Fixed seed -> identical sizes list reused for every checkpoint, so
    # differences in the result are due to the model, not sampled sizes.
    sizes = sample_molecule_sizes(QM9Dataset(args.parp_npz), args.n_samples, seed=args.seed)

    results = {}
    for epoch, path in CHECKPOINTS:
        print(f"evaluating epoch {epoch} ({path}) ...")
        res = evaluate_checkpoint(path, sizes, args.n_steps, device, args.seed)
        results[epoch] = res
        print(f"   epoch {epoch:4d} | valid {res['valid_pct']:.1%} | single {res['single_pct']:.1%} "
              f"| rings {res['ring_histogram']}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out_json}")

    phase1_log = load_training_log(args.phase1_log_csv)

    phase2_log = []
    if os.path.exists(args.phase2_raw_log):
        phase2_log = parse_training_log(args.phase2_raw_log)
        if phase2_log:
            write_training_log_csv(phase2_log, args.phase2_log_csv)
            print(f"Parsed {len(phase2_log)} phase-2 epoch rows from {args.phase2_raw_log} -> {args.phase2_log_csv}")
    elif os.path.exists(args.phase2_log_csv):
        phase2_log = load_training_log(args.phase2_log_csv)

    make_plot(phase1_log, phase2_log, results, args.out_png)


if __name__ == "__main__":
    main()
