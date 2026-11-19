"""Optional backend equivalence, topology, and dependency-boundary tests."""

from __future__ import annotations

import builtins
import math
import sys
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from pytfce import ptfce_exact


def assert_identical(reference, compiled):
    for key in ("p", "logp", "Z_enhanced"):
        assert reference[key].shape == compiled[key].shape
        assert reference[key].dtype == compiled[key].dtype
        assert reference[key].tobytes() == compiled[key].tobytes(), key
    assert reference["smoothness"] == compiled["smoothness"]
    assert reference["fwer_z_thresh"] == compiled["fwer_z_thresh"]
    np.testing.assert_array_equal(
        reference["Z_enhanced"] >= reference["fwer_z_thresh"],
        compiled["Z_enhanced"] >= compiled["fwer_z_thresh"],
    )
    for key in ("n_positive_voxels", "n_thresholds_used", "n_pvox_cache_entries"):
        assert reference["diagnostics"].get(key) == compiled["diagnostics"].get(key)


def small_case(kind):
    rng = np.random.default_rng(2913)
    z = rng.normal(size=(9, 10, 11))
    mask = rng.random(z.shape) > 0.15
    if kind == "plateaus":
        z = np.round(z)
    elif kind == "nonfinite":
        z[0, 0, 0], z[0, 0, 1], z[0, 0, 2] = np.nan, np.inf, -np.inf
        mask[0, 0, :3] = True
        z[~mask] = 1e9
    elif kind == "noncontiguous":
        z, mask = z[::-1, ::-1, ::-1], mask[::-1, ::-1, ::-1]
    elif kind == "fortran":
        z, mask = np.asfortranarray(z), np.asfortranarray(mask)
    elif kind == "one-positive":
        z[:] = -1.0
        z[4, 5, 6], mask[4, 5, 6] = 6.0, True
    elif kind == "zero":
        z[:] = 0.0
    elif kind == "negative":
        z[:] = -1.0
    elif kind == "empty-mask":
        mask[:] = False
    elif kind == "extreme-z":
        z[4, 5, 6], mask[4, 5, 6] = 39.0, True
    z.flags.writeable = False
    mask.flags.writeable = False
    return z, mask


@pytest.mark.parametrize(
    "kind",
    ["random", "plateaus", "nonfinite", "noncontiguous", "fortran",
     "one-positive", "zero", "negative", "empty-mask", "extreme-z"],
)
@pytest.mark.parametrize("nh", [1, 2, 31, 500])
def test_backend_bitwise_identity(kind, nh):
    pytest.importorskip("numba")
    z, mask = small_case(kind)
    kw: dict[str, Any] = {"max_thresholds": nh, "lut_density": 16, "verbose": False}
    reference = ptfce_exact(z, mask, uf_backend="python", **kw)
    compiled = ptfce_exact(z, mask, uf_backend="numba", **kw)
    assert_identical(reference, compiled)


@pytest.mark.parametrize("use_lut", [False, True])
def test_smoothness_override_and_progress(use_lut):
    pytest.importorskip("numba")
    z, mask = small_case("plateaus")
    smooth = {"V": int(mask.sum()), "Rd": 10.0, "n_resels": 2.0}

    class Progress:
        def __init__(self):
            self.events = []

        def put(self, event):
            self.events.append(event)

    python_progress, numba_progress = Progress(), Progress()
    kw: dict[str, Any] = {
        "max_thresholds": 103, "lut_density": 16, "verbose": False,
        "smooth_override": smooth, "use_lut": use_lut,
    }
    a = ptfce_exact(z, mask, uf_backend="python", progress_queue=python_progress, **kw)
    b = ptfce_exact(z, mask, uf_backend="numba", progress_queue=numba_progress, **kw)
    assert_identical(a, b)
    assert python_progress.events == numba_progress.events


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 6, 7), (6, 1, 7), (6, 7, 1), (7, 8, 9)])
def test_component_sizes_against_independent_scipy_ccl(shape):
    pytest.importorskip("numba")
    from pytfce.core import _union_find_numba as kernel

    rng = np.random.default_rng(519)
    z = rng.integers(-2, 6, size=shape).astype(np.float64)
    mask = rng.random(shape) > 0.2
    flat = np.flatnonzero((z.ravel() > 0) & mask.ravel())
    order = np.argsort(-z.ravel()[flat])
    sorted_flat = flat[order]
    vals = z.ravel()[sorted_flat]
    n = len(flat)
    to_rank = np.full(z.size, -1, dtype=np.intp)
    to_rank[sorted_flat] = np.arange(n)
    parent = np.arange(n, dtype=np.intp)
    rank = np.zeros(n, dtype=np.intp)
    size = np.ones(n, dtype=np.intp)
    active = np.zeros(n, dtype=bool)
    seen = np.zeros(n + 1, dtype=bool)
    unique = np.empty(n, dtype=np.intp)
    weights = np.zeros(n + 1)
    accum = np.zeros(n)
    expected_accum = np.zeros(n)
    offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                        [0, 0, 1], [0, 0, -1]], dtype=np.intp)
    cursor = 0
    for h in (6.0, 5.0, 3.0, 1.0, 0.0, -1e10):
        cursor = kernel.activate(vals, sorted_flat, to_rank, parent, rank, size,
                                 active, offsets, cursor, h, *shape)
        selected = mask & (z > 0) & (z >= h)
        labels = np.empty(shape, dtype=np.intp)
        ndimage.label(selected, ndimage.generate_binary_structure(3, 1), output=labels)
        counts = np.bincount(labels.ravel())
        expected = counts[labels.ravel()[sorted_flat[:cursor]]]
        actual = np.array([size[kernel._find(parent, i)] for i in range(cursor)])
        np.testing.assert_array_equal(expected, actual)
        got_unique = kernel.collect_sizes(parent, size, cursor, seen, unique)
        np.testing.assert_array_equal(np.sort(got_unique), np.unique(expected))
        assert not seen.any()
        for c in got_unique:
            weights[c] = -math.log(max(math.exp(-float(c)), 1e-300))
        kernel.accumulate(parent, size, cursor, weights, accum)
        expected_accum[:cursor] += weights[expected]
        assert accum.tobytes() == expected_accum.tobytes()


def test_invalid_backend():
    z, mask = small_case("random")
    with pytest.raises(ValueError, match="uf_backend"):
        ptfce_exact(z, mask, uf_backend="invalid", verbose=False)


@pytest.fixture
def uncached_kernel(monkeypatch):
    """Force the lazy import to exercise the optional-dependency boundary."""
    import pytfce.core as core

    monkeypatch.delitem(sys.modules, "pytfce.core._union_find_numba", raising=False)
    monkeypatch.delattr(core, "_union_find_numba", raising=False)


def test_numba_absent_fallback(monkeypatch, uncached_kernel):
    # Model a missing optional dependency without mutating the environment.
    original_import = builtins.__import__
    numba_imports = []

    def missing_numba(name, *args, **kwargs):
        if name == "numba":
            numba_imports.append(name)
            raise ModuleNotFoundError("No module named 'numba'", name="numba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_numba)
    z, mask = small_case("random")
    kw: dict[str, Any] = {"max_thresholds": 5, "lut_density": 8, "verbose": False}
    default = ptfce_exact(z, mask, **kw)
    reference = ptfce_exact(z, mask, uf_backend="python", **kw)
    assert_identical(reference, default)
    assert default["diagnostics"]["uf_backend"] == "python"
    assert not numba_imports
    auto = ptfce_exact(z, mask, uf_backend="auto", **kw)
    assert_identical(reference, auto)
    assert auto["diagnostics"]["uf_backend"] == "python"
    assert numba_imports
    with pytest.raises(ImportError, match="accelerated"):
        ptfce_exact(z, mask, uf_backend="numba", **kw)


def test_auto_selects_numba():
    pytest.importorskip("numba")
    z, mask = small_case("random")
    kw: dict[str, Any] = {"max_thresholds": 5, "lut_density": 8, "verbose": False}
    reference = ptfce_exact(z, mask, uf_backend="python", **kw)
    auto = ptfce_exact(z, mask, uf_backend="auto", **kw)
    assert_identical(reference, auto)
    assert auto["diagnostics"]["uf_backend"] == "numba"


@pytest.mark.parametrize("backend", ["auto", "numba"])
@pytest.mark.parametrize("failure", ["missing-dependency", "broken-import"])
def test_broken_numba_does_not_fall_back(monkeypatch, uncached_kernel, backend, failure):
    original_import = builtins.__import__

    def broken_numba(name, *args, **kwargs):
        if name == "numba":
            if failure == "missing-dependency":
                raise ModuleNotFoundError("No module named 'llvmlite'", name="llvmlite")
            raise ImportError("Numba dependency is incompatible")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_numba)
    z, mask = small_case("random")
    with pytest.raises(ImportError, match=r"llvmlite|incompatible"):
        ptfce_exact(z, mask, uf_backend=backend, max_thresholds=2,
                    use_lut=False, verbose=False)


@pytest.mark.parametrize("backend", ["auto", "numba"])
def test_compilation_failure_does_not_fall_back(monkeypatch, backend):
    pytest.importorskip("numba")
    from pytfce.core import _union_find_numba as kernel

    def failed_compile(*args, **kwargs):
        raise RuntimeError("Numba compilation failed")

    monkeypatch.setattr(kernel, "activate", failed_compile)
    z, mask = small_case("random")
    with pytest.raises(RuntimeError, match="compilation failed"):
        ptfce_exact(z, mask, uf_backend=backend, max_thresholds=2,
                    use_lut=False, verbose=False)
