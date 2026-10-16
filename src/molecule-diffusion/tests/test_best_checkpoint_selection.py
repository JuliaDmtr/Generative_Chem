"""
Which checkpoint gets saved as "best".

v11 exposed the problem: the in-training diagnostic scored epochs 430, 450 and
490 all at exactly 89.1%, a strict `>` kept the first of them, and a proper
256-sample evaluation then showed epoch 499 (which scored 82.8% on the
diagnostic) to be the best of the lot — 68.0% usable molecules against 66.4%
for epoch 430.

The diagnostic is a binomial rate over --diag-n-samples molecules (64 by
default) at 200 diffusion steps, so its 95% interval is about +/-8 points,
while real differences between late epochs are 1-2 points. It cannot rank them.
Training loss is still falling, so when the evidence cannot separate two
checkpoints the later one is the better bet; an earlier one should be kept only
when the later one is worse by more than the noise floor.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from train_conditional import selection_margin, should_replace_best

N = 64

# The real v11 diagnostic history (PARP-conditioned single-molecule rate, %),
# read off training_log_v11_noH.txt. diag_every=10 from global epoch 310.
V11_EPOCHS = list(range(310, 500, 10)) + [499]
V11_SCORES = [85.9, 68.8, 65.6, 76.6, 75.0, 79.7, 79.7, 82.8, 73.4, 82.8,
              76.6, 76.6, 89.1, 75.0, 89.1, 78.1, 84.4, 79.7, 89.1, 82.8]


def replay(rule):
    """Run a selection rule over the v11 history; return the epoch it keeps."""
    kept, best = None, None
    for epoch, score in zip(V11_EPOCHS, [s / 100.0 for s in V11_SCORES]):
        if rule(best, score):
            kept = epoch
        best = score if best is None else max(best, score)
    return kept


def test_the_old_strict_rule_kept_the_wrong_epoch():
    """Documents the behaviour being replaced, so the regression stays visible."""
    assert replay(lambda best, s: best is None or s > best) == 430


def test_v11_history_now_keeps_the_final_epoch():
    """Epoch 499 measured best at 256 samples; the rule must land on it."""
    assert replay(lambda best, s: should_replace_best(best, s, N)) == 499


def test_first_diagnostic_is_always_kept():
    assert should_replace_best(None, 0.30, N)


def test_a_tie_goes_to_the_later_epoch():
    assert should_replace_best(0.891, 0.891, N)


def test_a_later_epoch_inside_the_noise_floor_is_kept():
    """89.1% vs 82.8% at n=64 is a 6.3-point gap against a 12-point noise floor."""
    assert selection_margin(0.891, 0.828, N) > 0.063
    assert should_replace_best(0.891, 0.828, N)


def test_a_significantly_worse_later_epoch_is_rejected():
    """A real collapse must still be caught, or the rule is just 'take the last'."""
    assert not should_replace_best(0.891, 0.50, N)
    assert not should_replace_best(0.891, 0.0, N)


def test_the_margin_shrinks_as_the_diagnostic_grows():
    """More diagnostic samples must make the rule more willing to discriminate."""
    assert selection_margin(0.891, 0.828, 1024) < selection_margin(0.891, 0.828, 64)
    assert not should_replace_best(0.891, 0.828, 4096)


def test_margin_matches_the_binomial_formula():
    got = selection_margin(0.9, 0.8, N)
    want = 1.96 * math.sqrt(0.9 * 0.1 / N + 0.8 * 0.2 / N)
    assert abs(got - want) < 1e-12
