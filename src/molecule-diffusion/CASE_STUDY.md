# Debugging a 3D Molecule Diffusion Model: A Case Study

**What this document is:** a written record of diagnosing and partially fixing a real
generative-chemistry failure mode in the E(3)-equivariant diffusion model in this
repository — including a fix that worked, and a follow-up experiment that didn't,
reported honestly rather than discarded.

## Correction (2026-10-08): the diagnosis below is invalid

After this case study was written, three implementation bugs were found in
the model it describes. They invalidate its central conclusions; all models,
samples and galleries produced with that code have been deleted.

1. **Padded atoms leaked into the coordinate update.** `egnn.py` masked the
   edge messages *before* `coord_mlp`, but `coord_mlp(0) != 0` because of its
   biases, so every padded slot pushed every real atom radially. With the
   trained v6 model, the same 30-atom molecule sampled with no padding came
   out collapsed (Rg 1.6 Å, real 3.9 Å) and with padding to 80 exploded
   (Rg 127 Å). Molecule *scale* was set by padding, not by the data.
2. **EDM's coordinate normalization was missing.** The update used the raw
   `x_i − x_j` instead of `(x_i − x_j)/(‖x_i − x_j‖ + 1)` with a `tanh`
   bound, so the padding error grew without limit during sampling.
3. **The atom-type scaling trick ("H_SCALE") had no effect.** It scaled only
   the network input, which the first linear layer absorbs exactly. EDM
   scales the diffused data itself.

Consequences for the text below:

- **Hypothesis 2 was not actually ruled out.** The averaged nearest-neighbour
  ratio (~0.97–0.99×) hid a padding-dependent scale error: molecules with
  little padding collapsed, which crowds atoms together and plausibly
  produces exactly the 3-membered rings and ~60° angles observed.
- **"Implemented the paper's input-scaling trick" is false**; it was a no-op.
- **The angle-loss results were measured on a broken model.** Whether a
  3-membered-ring problem exists at all with correct code is open.
- The original test suite missed all three bugs: it only checked padding on a
  freshly initialized network (where `coord_mlp(0) ≈ 0`) and was apparently
  never run — one of its tests returned NaN.

All three are fixed and covered by `tests/test_invariances.py` (each test
fails on the old code). The remaining differences from the EDM reference are
listed in `references/approach.md`. This document will be rewritten once a
model has been retrained with the fixed code.

## The problem

The model (an EDM-style architecture, Hoogeboom et al. 2022) generates 3D atom
coordinates and atom types jointly, which are then converted to molecules via
RDKit's distance-based bond perception. A large fraction of generated
PARP-inhibitor-candidate molecules came out chemically implausible: heavily
dominated by 3-membered rings — a ring size that is extremely rare and highly
strained in real drug-like chemistry (cyclopropane-type geometry, ~60° bond
angles).

The easy, wrong conclusion would have been "the bond-perception step is too
permissive" or "the model just needs more training." Neither turned out to be
right, and figuring out why required building purpose-made diagnostics rather
than guessing.

## Diagnosis: ruling out the obvious explanations first

**Hypothesis 1 — bond perception is too permissive.** RDKit's `DetermineBonds`
has a `covFactor` parameter controlling how generous its distance cutoff is for
calling two atoms bonded. Swept it across its full practical range, down to the
tightest cutoff that still produced any valid molecules at all. Result: the
3-membered-ring pattern was **unchanged at every setting**, including the
strictest one. (A subtlety caught along the way: `covFactor` is silently a
no-op under RDKit's default bond-perception method unless a different flag is
also set — an easy way to run a misleading experiment and not notice, which is
exactly why the sweep was re-run properly once that was found.) This ruled out
an overly permissive bond-perception step as the cause.

**Hypothesis 2 — a coordinate-scale/normalization bug.** *(Invalid — see
the correction above: a scale bug existed and was missed.)* Compared the
nearest-neighbor interatomic distance distribution of generated molecules
against real training-set conformers, measured with identical code. Ratio:
~0.97–0.99x — essentially identical. This ruled out a unit/scaling bug, which
was the leading hypothesis before measurement.

**Hypothesis 3 (confirmed) — a genuine 3D angular defect.** Built a custom
diagnostic measuring the distribution of angles formed at each atom by its
nearest neighbors, run on generated molecules vs. real conformers. Generated
molecules showed a **6–8x excess of angles clustered near 60°** (the angle a
3-membered ring forces at each vertex) relative to real molecules, at matched
bond lengths. This was hard, direct evidence: the model's bond *lengths* were
fine, but its 3D *angular* structure was systematically wrong — not an
artifact of how bonds were being inferred after the fact.

## Reading the primary source before designing a fix

Before writing new model code, read the original EDM paper in full (not just
recalling it) to answer two questions: is this a bug in this from-scratch
implementation, or an inherent property of the method — and is there anything
the reference implementation does that this codebase was missing?

Both answers turned out to matter:

- The architecture (a pairwise-distance-only equivariant GNN) has **no angle
  or 3-body term anywhere in the original paper either** — this wasn't an
  implementation bug, it's a known property of this whole model family.
- The paper's own ablation study (buried in an appendix, not the headline
  results) showed that feeding the atom-type channel into the network at a
  smaller scale than the coordinate channel — a small, specific input-scaling
  choice — took molecule stability from 46.9% to 82.0% on the paper's
  benchmark, with no other change. This repository had never implemented it.

## The fix, and the honest negative result

1. Implemented the paper's input-scaling trick. *(Invalid — the
   implementation was a no-op; see the correction above.)*
2. Designed and implemented an auxiliary training-time loss that penalizes
   near-60° angles among each atom's nearest predicted neighbors — computed
   in cosine space (avoiding `arccos`'s unstable gradient near its domain
   boundary) and restricted to each atom's k-nearest neighbors (keeping the
   cost linear in atom count instead of cubic), with unit tests confirming it
   scores an exact 60° configuration near its maximum and a 180° configuration
   near zero.
3. At a moderate loss weight, retrained the conditional model: the rate of
   generating a single, fully connected, valid molecule roughly **doubled**
   (from ~4-5% to ~8-10%), and the ring-size distribution shifted from
   dominated by 3-membered rings (~90% of all rings) to a healthier mix with
   real representation of 5-, 6-, 7-, and 8-membered rings (3-membered rings
   dropped to ~60%).
4. **Then tested whether more of the same would help more, and it didn't.**
   Retraining with a ~7x stronger version of that same loss made *every*
   tracked metric worse — including the angle-excess ratio the loss is
   specifically designed to reduce — and continued to get worse with
   additional training rather than recovering. Visualized this directly: the
   stronger-weighted run's core position-reconstruction loss plateaus at a
   noticeably worse value from very early on and never approaches the
   moderate-weight run's, a clean signature of the auxiliary loss overwhelming
   the primary training objective rather than complementing it. This was
   reported as a real finding, not quietly dropped.

## Knowing where the ceiling is

Cross-checked the original paper's own results at drug-molecule scale (their
GEOM-Drugs experiments, trained for 5.5 GPU-days on dedicated hardware — far
more compute than this project has access to). Even there, the paper reports
real, persistent failure modes — disconnected molecules and unrealistic ring
sizes — and states plainly that the model "does not feature any regularization
to prevent these phenomena." That reframes the realistic target: not "zero
invalid molecules," which the reference implementation itself doesn't achieve
at any practical scale, but "a high enough valid-output rate that rejection
sampling produces usable candidates at a reasonable cost." A rejection-sampling
pipeline with ring-size and force-field strain-energy filters (calibrated
against real, relaxed reference conformers) was built for exactly that
purpose, and is what actually produces this project's curated output today.

## What this demonstrates

- Diagnosing a subtle model failure with purpose-built measurements instead of
  guessing, and being willing to rule out one's own leading hypothesis when the
  data doesn't support it.
- ~~Reading and correctly applying a detail from primary literature that the
  existing codebase had missed.~~ *(Not true — the detail was applied in the
  wrong place and had no effect; see the correction above.)*
- Designing a differentiable, numerically stable, efficiently-scoped auxiliary
  training objective from scratch, with unit tests for its core claim.
- Running the follow-up experiment that could have made the story better, and
  reporting it honestly when it didn't — including *why*, visualized in a
  loss-curve comparison, not just a metric that went down.
- Recognizing a model family's real ceiling from the primary source rather
  than assuming more tuning would eventually close the gap, and building a
  practical mitigation (rejection sampling + strain filtering) instead of
  chasing an unrealistic target.

## Honest limitations and what's next

This is a small-scale, single-machine research prototype, not a production
system, and it is not state-of-the-art relative to current 3D molecule
generation methods — newer approaches diffuse bond order explicitly instead of
inferring it after the fact (sidestepping this entire class of bug), or
condition generation directly on a target protein's binding pocket rather than
a coarse class label. Extending toward either of those is a larger rebuild,
not a tuning exercise, and is the natural next step for this line of work.
