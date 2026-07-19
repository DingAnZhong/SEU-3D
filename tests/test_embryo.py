"""
Unit tests for seu_3d.embryo.Embryo.

Run with: pytest tests/
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

# Import embryo.py directly so the tests do not trigger the package
# __init__ (which pulls in napari/magicgui).
_EMBRYO_PY = Path(__file__).resolve().parent.parent / "seu_3d" / "embryo.py"
_spec = importlib.util.spec_from_file_location("seu_3d_embryo", _EMBRYO_PY)
_embryo_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_embryo_mod)
Embryo = _embryo_mod.Embryo


def _make_adata(dims=3, with_z=True, n=8):
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {"tissue": (["a", "b", "c", "a", "b", "c", "a", "b"] * (n // 8 + 1))[:n]},
        index=[f"cell_{i}" for i in range(n)],
    )
    if with_z:
        obs["z"] = np.arange(n, dtype=float)
    adata = AnnData(X=np.ones((n, 4)), obs=obs)
    adata.obsm["spatial"] = rng.random((n, dims))
    return adata


class TestEmbryo:
    def test_3d_coordinates_kept(self):
        adata = _make_adata(dims=3)
        emb = Embryo("dummy.h5ad", "tissue", "spatial", adata)
        assert emb.coordinate_3d.shape == (8, 3)

    def test_2d_coordinates_stacked_with_z(self):
        adata = _make_adata(dims=2, with_z=True)
        emb = Embryo("dummy.h5ad", "tissue", "spatial", adata)
        assert emb.coordinate_3d.shape == (8, 3)
        np.testing.assert_allclose(emb.coordinate_3d[:, 2], adata.obs["z"].values)
        # stacked coordinates are written back to obsm
        assert adata.obsm["spatial"].shape == (8, 3)

    def test_2d_without_z_raises(self):
        adata = _make_adata(dims=2, with_z=False)
        with pytest.raises(ValueError):
            Embryo("dummy.h5ad", "tissue", "spatial", adata)

    def test_missing_obsm_key_raises(self):
        adata = _make_adata(dims=3)
        with pytest.raises(KeyError):
            Embryo("dummy.h5ad", "tissue", "not_a_key", adata)

    def test_missing_tissue_column_raises(self):
        adata = _make_adata(dims=3)
        with pytest.raises(KeyError):
            Embryo("dummy.h5ad", "not_a_column", "spatial", adata)

    def test_all_tissues(self):
        adata = _make_adata(dims=3)
        emb = Embryo("dummy.h5ad", "tissue", "spatial", adata)
        assert sorted(emb.all_tissues) == ["a", "b", "c"]
