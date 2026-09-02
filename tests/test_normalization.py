"""
Tests for preprocessing/normalization.py: reference building (mean
reduction, missing-flat/dark handling) and the normalize() formula --
including that normalize_slice() and normalize_rows() actually agree
with each other and with a plain manual computation, which is the
concrete promise behind "one function, many shapes" from how this
module is designed.
"""
import numpy as np
import pytest

from core.handles import RamHandle
from core.metadata import MetadataBundle
from core.project import ProjectObject
from preprocessing.normalization import (
    MissingFlatFieldError,
    build_reference,
    normalize,
    normalize_rows,
    normalize_slice,
)


def _project(projections, flats=None, darks=None):
    meta = MetadataBundle()
    meta.set("projections_path", "/projections", tier="mandatory_hard")
    handles = {"projections": RamHandle(projections)}
    if flats is not None:
        handles["flats"] = RamHandle(flats)
    if darks is not None:
        handles["darks"] = RamHandle(darks)
    return ProjectObject(format="hdf5", storage_mode="ram", metadata=meta, data_handles=handles)


def _synthetic(seed=0, n_angles=5, height=4, width=6, n_flats=3, n_darks=3):
    rng = np.random.default_rng(seed)
    projections = rng.uniform(50, 200, size=(n_angles, height, width)).astype(np.float32)
    flats = rng.uniform(180, 220, size=(n_flats, height, width)).astype(np.float32)
    darks = rng.uniform(5, 15, size=(n_darks, height, width)).astype(np.float32)
    return projections, flats, darks


def test_build_reference_raises_without_flats():
    project = _project(np.zeros((3, 2, 2), dtype=np.float32))

    with pytest.raises(MissingFlatFieldError):
        build_reference(project)


def test_build_reference_defaults_dark_to_scalar_zero_without_darks():
    projections, flats, _ = _synthetic()
    project = _project(projections, flats=flats)

    reference = build_reference(project)

    assert reference.dark == 0.0
    assert not isinstance(reference.dark, np.ndarray)


def test_build_reference_reduces_by_mean():
    projections, flats, darks = _synthetic()
    project = _project(projections, flats=flats, darks=darks)

    reference = build_reference(project)

    np.testing.assert_allclose(reference.flat, np.mean(flats, axis=0), rtol=1e-5)
    np.testing.assert_allclose(reference.dark, np.mean(darks, axis=0), rtol=1e-5)


def test_build_reference_median_reducer():
    projections, flats, darks = _synthetic()
    project = _project(projections, flats=flats, darks=darks)

    reference = build_reference(project, reducer="median")

    np.testing.assert_allclose(reference.flat, np.median(flats, axis=0), rtol=1e-5)


def test_normalize_matches_manual_formula():
    projections, flats, darks = _synthetic()
    project = _project(projections, flats=flats, darks=darks)
    reference = build_reference(project)

    result = normalize(projections, reference)

    expected = (projections - reference.dark) / (reference.flat - reference.dark)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_normalize_slice_matches_manual_row_extraction():
    projections, flats, darks = _synthetic()
    project = _project(projections, flats=flats, darks=darks)
    reference = build_reference(project)

    row = 2
    result = normalize_slice(project, row, reference)

    expected = (projections[:, row, :] - reference.dark[row]) / (reference.flat[row] - reference.dark[row])
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result.shape == (projections.shape[0], projections.shape[2])


def test_normalize_rows_whole_volume_matches_per_slice_calls():
    """The core promise: normalizing the whole volume in one call gives
    exactly the same result as normalizing each sinogram individually."""
    projections, flats, darks = _synthetic()
    project = _project(projections, flats=flats, darks=darks)
    reference = build_reference(project)

    whole_volume = normalize_rows(project, slice(None), reference)

    for row in range(projections.shape[1]):
        one_slice = normalize_slice(project, row, reference)
        np.testing.assert_allclose(whole_volume[:, row, :], one_slice, rtol=1e-5)


def test_normalize_rows_batch_matches_per_slice_calls():
    projections, flats, darks = _synthetic()
    project = _project(projections, flats=flats, darks=darks)
    reference = build_reference(project)

    batch = normalize_rows(project, slice(1, 3), reference)

    for offset, row in enumerate(range(1, 3)):
        one_slice = normalize_slice(project, row, reference)
        np.testing.assert_allclose(batch[:, offset, :], one_slice, rtol=1e-5)


def test_normalize_without_dark_field_reduces_to_raw_over_flat():
    projections, flats, _ = _synthetic()
    project = _project(projections, flats=flats)
    reference = build_reference(project)

    result = normalize(projections, reference)

    np.testing.assert_allclose(result, projections / reference.flat, rtol=1e-5)
