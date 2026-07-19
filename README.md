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
- **Performance**: remove redundant full `AnnData.copy()` in filtering and surface generation
- Move h5ad reading, Moran's I, surface reconstruction and similar-gene search to worker threads (no more GUI freezing)
- Sparse-aware cosine similarity and column means (no full-matrix densification)
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
