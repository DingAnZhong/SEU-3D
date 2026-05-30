# SEU-3D

3D visualization and analysis plugin for spatial transcriptomics embryo data based on napari.

[![PyPI version](https://img.shields.io/pypi/v/seu-3d.svg)](https://pypi.org/project/seu-3d/)
[![License](https://img.shields.io/pypi/l/seu-3d.svg)](https://github.com/DingAnZhong/SEU-3D)

## Features

- **3D Visualization**: Render spatial transcriptomics data in 3D with napari
- **Tissue Filtering**: Filter cells by tissue type, slice, germ layer, and XYZ range
- **Gene Expression Analysis**: Single, dual, and triple gene expression coloring
- **Similarity Search**: Find genes with similar expression patterns (cached)
- **Moran's I Spatial Autocorrelation**: Compute spatial gene enrichment
- **Differential Expression**: Identify tissue-specific marker genes
- **Annotation**: Cluster and label cell populations in 3D space
- **Surface Reconstruction**: Generate 3D surfaces for tissue regions (pyvista)
- **Automatic Color Mapping**: Deterministic tissue colors with JSON override support

## Installation

```bash
pip install seu-3d
```

Then open napari and look for "Load spatial transcriptomics data" in the plugins menu.

## Environment

Required dependencies are installed automatically via pip. For full functionality:

- Python >= 3.9
- napari >= 0.4.0
- anndata >= 0.10
- scanpy
- squidpy
- pyvista (optional, for surface generation)

## Quick Start

```python
import napari
viewer = napari.Viewer()
# Add spatial transcriptomics h5ad file via the napari plugin menu
```

## Update Log

### [1.1.17] — 2026-05-30
Full code optimization release:
- **Bug Fixes**:
  - Fix annotation float coordinate matching failure (replaced `np.isin` with `cKDTree` distance-based matching)
  - Fix `threshold_3gene` widget variable name inconsistency
  - Fix 2D→3D coordinate stacking producing view references (added `.copy()`)
  - Fix missing `.raw` layer checks across multiple methods
- **New Features**:
  - `get_expression()` utility for safe gene expression retrieval (supports AnnData & Embryo objects)
  - `require_active_points()` decorator for robust layer selection checking
  - `_color_by_gene()` and `_normalize_gene_exp()` common methods to reduce code duplication
  - `_similarity_cache` for cosine similarity results in gene similarity search
  - `config.py` centralized constant management
- **UX Improvements**:
  - Improved colorbar display height and label visibility
  - Automatic colormap fallback for tissues exceeding hardcoded color count (hash-deterministic)
  - Structured AnnData info summary replacing verbose string output
  - Progress status text updates for long-running operations
- **Code Quality**:
  - Cleaned up duplicate imports
  - 52 unit tests covering all core functionality

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
