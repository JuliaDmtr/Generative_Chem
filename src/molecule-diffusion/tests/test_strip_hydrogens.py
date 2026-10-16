"""Stripping hydrogens must remove every H and leave heavy atoms untouched."""
import numpy as np
import pytest

from strip_hydrogens import H_INDEX, strip_hydrogens
from utils import ATOM2IDX


def molecule(types, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(len(types), 3)).astype(np.float32), np.array(types, dtype=np.int64)


def test_removes_only_hydrogens_and_keeps_heavy_coordinates():
    C, N, O = ATOM2IDX["C"], ATOM2IDX["N"], ATOM2IDX["O"]
    pos, types = molecule([C, H_INDEX, N, H_INDEX, H_INDEX, O])
    (out_pos,), (out_types,), dropped = strip_hydrogens([pos], [types])
    assert dropped == 0
    assert list(out_types) == [C, N, O], "heavy atom types or their order changed"
    assert np.array_equal(out_pos, pos[[0, 2, 5]]), "heavy atom coordinates changed"
    assert not (out_types == H_INDEX).any()


def test_molecule_without_hydrogens_is_unchanged():
    types = [ATOM2IDX[s] for s in ("C", "C", "N", "O")]
    pos, t = molecule(types, seed=1)
    (out_pos,), (out_types,), dropped = strip_hydrogens([pos], [t])
    assert dropped == 0 and np.array_equal(out_pos, pos) and np.array_equal(out_types, t)


def test_molecules_too_small_after_stripping_are_dropped():
    C = ATOM2IDX["C"]
    water = molecule([ATOM2IDX["O"], H_INDEX, H_INDEX])          # 1 heavy atom
    ethane = molecule([C, C, H_INDEX, H_INDEX, H_INDEX], seed=2)  # 2 heavy atoms
    big = molecule([C, C, C, H_INDEX], seed=3)                    # 3 heavy atoms
    pos, types, dropped = strip_hydrogens([water[0], ethane[0], big[0]],
                                          [water[1], ethane[1], big[1]], min_heavy_atoms=3)
    assert dropped == 2 and len(pos) == 1 and len(types[0]) == 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_dtypes_match_dataset_expectations(dtype):
    C = ATOM2IDX["C"]
    pos, types = molecule([C, C, C, H_INDEX])
    (out_pos,), (out_types,), _ = strip_hydrogens([pos.astype(dtype)], [types])
    assert out_pos.dtype == np.float32 and out_types.dtype == np.int64
