# -*- coding: utf-8 -*-
"""
Offscreen end-to-end smoke test for seu-3d 2.0.0.

Runs the real plugin code against a synthetic h5ad file with a headless
napari viewer (QT_QPA_PLATFORM=offscreen). Exercises:
  - async ReadAdata loading (thread_worker path)
  - load_embryo wiring
  - display() / tissue_filter() incl. the obs['slices'] fix
  - named-layer reuse (no layer stacking)
  - _normalize_exp constant-expression guard
  - _show_gene_expression
  - show_flatten layer reuse
  - _similar_genes_worker (sparse X)
  - _moran_worker (squidpy)
  - deterministic tissue colors

Run:  python tests/smoke_offscreen.py
Exit code 0 = all checks passed.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

FAILURES = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        FAILURES.append(name)


def spin(predicate, timeout=60):
    """Pump the Qt event loop until predicate() is True or timeout."""
    from qtpy.QtCore import QCoreApplication

    t0 = time.time()
    while time.time() - t0 < timeout:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.05)
    return False


def make_adata(n=600, g=60, dims=3):
    rng = np.random.default_rng(42)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    obs["tissue"] = pd.Categorical([["brain", "heart", "limb"][i % 3] for i in range(n)])
    obs["slices"] = pd.Categorical([["s1", "s2", "s3"][i // 200] for i in range(n)])
    obs["germ_layer"] = pd.Categorical([["ecto", "meso", "endo"][(i // 7) % 3] for i in range(n)])
    obs["z"] = (obs["slices"].astype(str).map({"s1": 0.0, "s2": 20.0, "s3": 40.0})).values
    obs["x_flatten"] = rng.random(n)
    obs["y_flatten"] = rng.random(n)
    X = sp.random(n, g, density=0.2, format="csr", random_state=42, dtype=np.float32)
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f"gene_{i}" for i in range(g)]
    adata.obsm["spatial"] = rng.random((n, dims)).astype(np.float32)
    return adata


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    h5ad_path = os.path.join(here, "_smoke_data.h5ad")
    adata = make_adata()
    adata.write_h5ad(h5ad_path)

    import napari
    viewer = napari.Viewer(show=False)

    from seu_3d.load import ReadAdata
    from seu_3d.embryo import Embryo
    from seu_3d.display import (
        DisplayEmbryo,
        _similar_genes_worker,
        _moran_worker,
    )

    # ---- 1. async ReadAdata path -------------------------------------------
    w = ReadAdata(viewer)
    w.h5ad_file.value = h5ad_path
    w.show_components()
    ok = spin(lambda: hasattr(w, "adata"), timeout=120)
    check("ReadAdata async load finishes", ok)
    check("no error label text", w.error_label.text() == "")

    # ---- 2. load_embryo wiring (uses widget defaults: tissue / spatial) ----
    w.load_embryo()
    check("widget load_embryo creates embryo", hasattr(w, "embryo"))

    # ---- 3. DisplayEmbryo basics -------------------------------------------
    emb = Embryo(h5ad_path, "tissue", "spatial", w.adata)
    disp = DisplayEmbryo(viewer, emb)
    check("seu_3d_cells layer added with 600 pts",
          "seu_3d_cells" in viewer.layers and len(viewer.layers["seu_3d_cells"].data) == 600)
    check("3 tissue colors", len(disp.tissue_color_map) == 3)

    # ---- 4. tissue_filter incl. obs['slices'] fix --------------------------
    disp.tissue_select.value = ["brain"]
    disp.slice_select.value = ["s1"]
    disp.tissue_filter()
    expected = int(((adata.obs["tissue"] == "brain") & (adata.obs["slices"] == "s1")).sum())
    check(f"filter brain&s1 -> {expected} cells (slices fix works)",
          disp.selected_adata.n_obs == expected and expected > 0)

    # ---- 5. named-layer reuse (call twice, still one layer) ----------------
    disp.tissue_filter()
    n_cell_layers = sum(1 for l in viewer.layers if l.name == "seu_3d_cells")
    check("no duplicate seu_3d_cells layers", n_cell_layers == 1)
    check("re-filtered layer has filtered size",
          len(viewer.layers["seu_3d_cells"].data) == expected)

    # ---- 6. _normalize_exp constant guard ----------------------------------
    const = DisplayEmbryo._normalize_exp(np.full(10, 5.0))
    check("constant expression -> zeros (no NaN)",
          np.all(const == 0) and not np.any(np.isnan(const)))
    ramp = DisplayEmbryo._normalize_exp(np.arange(10.0))
    check("normalization bounds", ramp.min() == 0.0 and ramp.max() == 1.0)

    # ---- 7. _show_gene_expression (single reusable layer) ------------------
    disp._show_gene_expression(disp.selected_adata, "gene_2")
    check("gene layer added", "gene_expression" in viewer.layers)
    check("gene recorded in layer metadata",
          viewer.layers["gene_expression"].metadata.get("gene") == "gene_2")
    disp._show_gene_expression(disp.selected_adata, "gene_3")
    n_gene_layers = sum(1 for l in viewer.layers if l.name == "gene_expression")
    check("gene layer reused (no stacking)", n_gene_layers == 1)
    check("cell_gene_color cached", len(disp.cell_gene_color) == disp.selected_adata.n_obs)

    # ---- 8. show_flatten layer reuse ---------------------------------------
    disp.show_flatten()
    disp.show_flatten()
    n_flat = sum(1 for l in viewer.layers if l.name == "flatten")
    check("flatten layer reused", n_flat == 1)

    # ---- 9. similar-genes worker (sparse X) --------------------------------
    sub = disp.selected_adata
    result = {}
    worker = _similar_genes_worker(sub, "gene_1")
    worker.returned.connect(lambda r: result.setdefault("genes", r))
    worker.errored.connect(lambda e: result.setdefault("err", e))
    worker.start()
    ok = spin(lambda: "genes" in result or "err" in result, timeout=120)
    check("similar-genes worker finishes", ok and "genes" in result)
    if "genes" in result:
        check("returns 10 genes excl. query",
              len(result["genes"]) == 10 and "gene_1" not in result["genes"])
    if "err" in result:
        print("  worker error:", result["err"])

    # ---- 10. Moran worker ---------------------------------------------------
    res2 = {}
    worker2 = _moran_worker(sub, "spatial")
    worker2.returned.connect(lambda r: res2.setdefault("df", r))
    worker2.errored.connect(lambda e: res2.setdefault("err", e))
    worker2.start()
    ok = spin(lambda: "df" in res2 or "err" in res2, timeout=180)
    check("Moran worker finishes", ok and "df" in res2)
    if "df" in res2:
        check("Moran result has I for all genes",
              "I" in res2["df"].columns and len(res2["df"]) == sub.n_vars)
    if "err" in res2:
        print("  worker error:", res2["err"])

    # ---- 11. deterministic colors ------------------------------------------
    disp2 = DisplayEmbryo(viewer, Embryo(h5ad_path, "tissue", "spatial", adata))
    check("tissue colors deterministic", disp2.tissue_color_map == disp.tissue_color_map)

    # ---- 12. 2D coords + z stacking ----------------------------------------
    adata2d = make_adata(n=60, g=10, dims=2)
    emb2d = Embryo(h5ad_path, "tissue", "spatial", adata2d)
    check("2D+z stacked to 3D", emb2d.coordinate_3d.shape == (60, 3))
    check("stacked coords written back to obsm", adata2d.obsm["spatial"].shape == (60, 3))

    print("\n==== %d checks failed ====" % len(FAILURES))
    if FAILURES:
        for f in FAILURES:
            print(" -", f)
    try:
        os.remove(h5ad_path)
    except OSError:
        pass
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
