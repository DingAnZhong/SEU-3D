# SEU-3D

3D visualization and analysis plugin for spatial transcriptomics embryo data based on napari.

[![PyPI version](https://img.shields.io/pypi/v/seu-3d.svg)](https://pypi.org/project/seu-3d/)
[![License](https://img.shields.io/pypi/l/seu-3d.svg)](https://github.com/DingAnZhong/SEU-3D)

## Features

- **3D Visualization**: Render spatial transcriptomics data in 3D with napari
- **Tissue Filtering**: Filter cells by tissue type, slice, germ layer, and XYZ range
- **Gene Expression Analysis**: Single, dual, and triple gene expression coloring
- **Similarity Search**: Find genes with similar expression patterns
- **Moran's I Spatial Autocorrelation**: Compute spatial gene enrichment
- **Differential Expression**: Identify tissue-specific marker genes
- **Annotation**: Cluster and label cell populations in 3D space
- **Surface Reconstruction**: Generate 3D surfaces for tissue regions (pyvista)
- **Automatic Color Mapping**: Deterministic tissue colors from a discrete colormap

## Installation

```bash
pip install seu-3d
```

Then open napari and look for "Load spatial transcriptomics data" in the plugins menu.

## Environment

All runtime dependencies are declared in `pyproject.toml` and installed automatically via pip:

```bash
pip install seu-3d              # core functionality
pip install "seu-3d[surface]"   # + pyvista, for 3D surface reconstruction
```

- Python >= 3.9

## Data Format

The plugin reads `.h5ad` files. Two fields are **chosen in the widget** when
loading; everything else is auto-detected by fixed names:

| Field | Location | Required | Used for |
|---|---|---|---|
| *(your choice)* | `adata.obsm` key | Yes | Spatial coordinates. 3D `(x, y, z)` preferred; if 2D `(x, y)`, `obs['z']` is used as z |
| *(your choice)* | `adata.obs` column | Yes | Tissue / cell-type labels (coloring, filtering, diff-expression) |
| `z` | `adata.obs` | Only when the coordinate key is 2D | Z position of each cell |
| `slices` | `adata.obs` | Optional | Slice/sample filter (falls back to `orig.ident`) |
| `orig.ident` | `adata.obs` | Optional | Fallback slice/sample filter |
| `germ_layer` | `adata.obs` | Optional | Germ-layer filter |
| `x_flatten`, `y_flatten` | `adata.obs` | Optional | "Show Flatten" 2D projection |

Missing optional fields simply disable the corresponding tab; the rest of the
plugin keeps working.

## Quick Start

```python
import napari
viewer = napari.Viewer()
# Add spatial transcriptomics h5ad file via the napari plugin menu
```

## Update Log

### [2.0.0] — 2026-07-19
- **Packaging**:
  - Declare runtime dependencies in `pyproject.toml` (previously `pip install seu-3d` pulled in nothing)
  - Restrict `packages.find` to `seu_3d*` so stray directories are never published
  - Remove the unused, broken `weiwei/` source tree and the dead `_umap_selection` module
  - Single-source the version from `seu_3d.__version__`
- **Bug Fixes**:
  - Fix slice filtering never applying (`obs['slice']` vs `obs['slices']` mismatch)
  - Fix differential-expression tissue selection reading only the first character of the tissue name
  - Fix annotation coordinate matching failing on floats (cKDTree nearest-neighbour matching) and guard empty selections
  - Fix saving annotations before annotating raised `AttributeError`
  - Fix NaN colors for genes with constant expression
  - Deterministic tissue colors (removed random shuffle)
  - Re-filtering and XY preview now reuse their napari layers instead of piling up new ones
- **Performance**:
  - Remove redundant full `AnnData.copy()` in filtering and surface generation; filtering now applies one combined mask (a single copy per click instead of four)
  - Annotations accumulate in one persistent copy (no per-click dataset copy, and repeated annotations no longer reset earlier ones)
  - Move h5ad reading, Moran's I, surface reconstruction and similar-gene search to worker threads (no more GUI freezing)
  - Sparse-aware cosine similarity and column means (no full-matrix densification)
- **UX**:
  - Gene-expression views reuse a single `gene_expression` layer instead of stacking a new view per click
  - Dock panel width is bounded and scrollable; AnnData info box is read-only and height-capped
  - Documented the expected `obs`/`obsm` field names (see Data Format above)
- **Code Quality**: remove duplicate/unused imports, unify gene-expression coloring into shared helpers
- **Housekeeping**:
  - Centralize 2D->3D coordinate stacking in `Embryo` (written back to `obsm`)
  - Add MIT `LICENSE` and unit tests for `Embryo`

### [1.1.17] — 2026-05-30
- First public release as a napari manifest plugin: h5ad loading, tissue/slice/germ-layer/XY filtering, 1-3 gene expression coloring, similar-gene search, Moran's I, differential expression, annotation, pyvista surface reconstruction

### [1.1.6]
First release with napari manifest support

### [1.0.4]
Fix error that plugin could not show

### [1.0.3]
Fix napari.yaml

### [1.0.2]
Forgot to upload napari.yaml

### [1.0.1]
First release

## Acknowledgements

- [sc3D](https://github.com/GuignardLab/sc3D)
- [napari-sc3D-viewer](https://github.com/GuignardLab/napari-sc3D-viewer)

## License

Released under the [MIT License](LICENSE), like the upstream [sc3D](https://github.com/GuignardLab/sc3D) project it is derived from.
