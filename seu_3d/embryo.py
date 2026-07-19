'''
Refer to sc3d:https://github.com/GuignardLab/sc3D
'''

import numpy as np
from pathlib import Path


class Embryo:
    """
    Class to handle the reading and processing of 3D spatial single-cell omics datasets.
    """
    def __init__(self, data_path, tissue_name, coordinate_3d_key, adata ,**kwargs):
        """
        Initialize the Embryo object with the given parameters(from widget.ReadAdata).

        Args:
            data_path (str): Path to the anndata file.
            tissue_name (str): Column name in adata.obs for tissue identification.
            coordinate_3d_key (str): Key in adata.obsm for 3D spatial coordinates.
            adata (anndata.AnnData): Anndata object containing the dataset.
            **kwargs: Additional parameters (currently unused).
        """
        self.data_path = Path(data_path)
        self.tissue_name = tissue_name
        self.coordinate_3d_key = coordinate_3d_key
        self.adata = adata
        if tissue_name not in adata.obs:
            raise KeyError(
                f"'{tissue_name}' not found in adata.obs "
                f"(available: {list(adata.obs.columns)})"
            )
        if coordinate_3d_key not in adata.obsm:
            raise KeyError(
                f"'{coordinate_3d_key}' not found in adata.obsm "
                f"(available: {list(adata.obsm.keys())})"
            )
        self.coordinate_3d = adata.obsm[coordinate_3d_key]
        self.cells_id = list(range(adata.n_obs))
        self.cell_names = {
            i: name.split("_")[-1]
            for i, name in zip(range(adata.n_obs), adata.obs_names)
        }
        self.all_tissues = adata.obs[self.tissue_name].unique().tolist()
        if self.coordinate_3d.shape[-1] == 2:
            if 'z' in self.adata.obs.columns:
                z = self.adata.obs['z'].values
                self.coordinate_3d = np.column_stack(
                    [self.coordinate_3d[:, 0], self.coordinate_3d[:, 1], z]
                )
                # Persist the stacked 3D coordinates so downstream code can
                # rely on obsm[coordinate_3d_key] always being 3D.
                self.adata.obsm[coordinate_3d_key] = self.coordinate_3d.copy()
            else:
                raise ValueError("The coordinate_3d is 2D, but 'z' column not found in adata.obs. Please provide a valid 3D coordinate.")