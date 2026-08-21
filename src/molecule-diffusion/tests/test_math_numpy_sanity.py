"""
Pure-NumPy reimplementation of the cosine noise schedule and forward
diffusion equations, used to sanity-check the math independent of the
torch implementation (and runnable with zero dependencies beyond numpy,
which is why this file doesn't need `pytest.importorskip`).

Run directly:  python tests/test_math_numpy_sanity.py
Or via pytest: pytest tests/test_math_numpy_sanity.py
"""
import math

import numpy as np


def cosine_alpha_bar_np(t: np.ndarray, s: float = 0.008) -> np.ndarray:
    f = np.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    f0 = math.cos((s / (1 + s)) * math.pi / 2) ** 2
    return np.clip(f / f0, 1e-5, 1.0)


def test_schedule_boundary_conditions():
    ab = cosine_alpha_bar_np(np.array([0.0, 1.0]))
    assert abs(ab[0] - 1.0) < 1e-6, "alpha_bar(0) should be ~1 (no noise at t=0)"
    assert ab[1] < 0.01, "alpha_bar(1) should be ~0 (pure noise at t=1)"


def test_schedule_is_monotonically_non_increasing():
    t = np.linspace(0, 1, 200)
    ab = cosine_alpha_bar_np(t)
    assert np.all(np.diff(ab) <= 1e-9)


def test_variance_preserving_identity():
    """For a variance-preserving process, Var(z_t) should equal
    Var(x0) when x0 has unit variance and alpha_bar + (1-alpha_bar) = 1,
    i.e. sqrt(ab)^2 * Var(x0) + sqrt(1-ab)^2 * Var(eps) == Var(x0) when
    Var(x0) == Var(eps) == 1. This is the identity the whole forward
    process (q_sample in diffusion.py) relies on.
    """
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=200_000)
    eps = rng.normal(size=200_000)
    for t_val in [0.1, 0.5, 0.9]:
        ab = cosine_alpha_bar_np(np.array([t_val]))[0]
        z_t = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
        # z_t should still have ~unit variance, for any t
        assert abs(np.var(z_t) - 1.0) < 0.05, f"variance drifted at t={t_val}"


def test_center_of_mass_removal_is_idempotent_and_zero_sum():
    """Mirrors utils.remove_mean_with_mask: subtracting the per-molecule
    centroid should leave a set of points summing to (numerically) zero,
    and doing it twice should be a no-op.
    """
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(10, 3))

    def remove_mean(p):
        return p - p.mean(axis=0, keepdims=True)

    once = remove_mean(pts)
    twice = remove_mean(once)
    assert np.allclose(once.sum(axis=0), 0.0, atol=1e-10)
    assert np.allclose(once, twice, atol=1e-12)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nAll {len(tests)} pure-NumPy sanity checks passed.")
