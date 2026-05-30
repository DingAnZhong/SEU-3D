import os
os.environ["QT_ENABLE_GLYPH_CACHE_SHARING"] = "1"
from qtpy import QtCore, QtWidgets
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget
from magicgui import widgets
from ._umap_selection import UmapSelection
from ._utils import error_points_selection, safe_toarray
from napari.utils.colormaps import ALL_COLORMAPS, Colormap
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib import cm, colors
import numpy as np
import logging
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
import random
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import colorcet as cc
import scanpy as sc
import squidpy as sq
from tqdm import tqdm
import sys
from scipy.sparse import issparse
try:
    from pyvista import PolyData

    pyvista = True
except Exception as e:
    print(
        (
            "pyvista is not installed. No surfaces can be generated\n"
            "Try pip install pyvista or conda install pyvista to install it"
        )
    )
    pyvista = False

class DisplayEmbryo():
    '''
    Class to display and analyse the Embryo data in a napari viewer.
    '''
    def color_set(self):
        rainbow_colors = cc.cm['rainbow4']
        n_colors = len(self.embryo.all_tissues)
        colors = rainbow_colors(np.linspace(0, 1, n_colors))
        discrete_cmap = mcolors.ListedColormap(colors)
        colors_hex = [mcolors.rgb2hex(color) for color in discrete_cmap.colors]
        return colors_hex
    
    def legend_tab(self):
        '''
        Create the legend tab
        '''
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots()
            ax.axis('off')
            colors_hex = self.color_set()
            idx = list(range(len(colors_hex)))
            random.shuffle(idx)
            colors_hex = [colors_hex[i] for i in idx]
            for color,tissue in zip(colors_hex,self.embryo.all_tissues):
                ax.plot([], [], 's', markersize=7, color=color, label=tissue)
            canvas = FigureCanvas(fig)
            legend = fig.legend(
                loc='center',
                bbox_to_anchor=(0.5,0.5),
                frameon=False,
                fontsize=7,
            )

        self.tissue_color_map = {tissue: color for tissue, color in zip(self.embryo.all_tissues, colors_hex)}
        legend_tab = QTabWidget()
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        legend_tab.setObjectName("legend_tab")
        legend_tab.setLayout(layout)

        return legend_tab
    
    def tissue_filter(self):
        '''
        Create a tissue filter fuction
        '''
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection()
        adata = self.embryo.adata.copy()

        selected_tissue = self.tissue_select.value
        selected_slice = self.slice_select.value
        selected_germ_layer = self.germ_layer_select.value
        select_xy = [[self.show_x_min.value, self.show_x_max.value],
                    [self.show_y_min.value, self.show_y_max.value]]
        
        filter_1 = adata[adata.obs[self.embryo.tissue_name].isin(selected_tissue)]

        if 'slice' in adata.obs.columns:
            filter_2 = filter_1[filter_1.obs['slice'].isin(selected_slice)]
        elif 'orig.ident' in adata.obs.columns:
            filter_2 = filter_1[filter_1.obs['orig.ident'].isin(selected_slice)]
        else:
            filter_2 = filter_1

        if 'germ_layer' in adata.obs.columns:
            filter_3 = filter_2[filter_2.obs['germ_layer'].isin(selected_germ_layer)]
        else:
            filter_3 = filter_2

        filter_4 = filter_3[
            (filter_3.obsm[self.embryo.coordinate_3d_key][:, 0] >= select_xy[0][0]) &
            (filter_3.obsm[self.embryo.coordinate_3d_key][:, 0] <= select_xy[0][1]) &
            (filter_3.obsm[self.embryo.coordinate_3d_key][:, 1] >= select_xy[1][0]) &
            (filter_3.obsm[self.embryo.coordinate_3d_key][:, 1] <= select_xy[1][1])
        ]

        self.selected_adata = filter_4

        self.display()
        
    def tissue_tab(self):
        def select_tissue():
            self.tissue_select = widgets.Select(
                choices=self.embryo.all_tissues,
                value=self.embryo.all_tissues,
            )
            run_tissue_filter = widgets.FunctionGui(
                self.tissue_filter,
                call_button="Select Tissue",
                layout="vertical",
            )
            run_show_flatten = widgets.FunctionGui(
                self.show_flatten,
                call_button="Show Flatten",
                layout="vertical",
            )
            select_tissue_container = widgets.Container(
                widgets=[self.tissue_select, run_tissue_filter, run_show_flatten],
                layout="vertical",
            )
            return select_tissue_container
        
        tissue_tab = QTabWidget()
        tissue_tab.setObjectName("Tissue")
        tissue_tab.addTab(select_tissue().native, "Select Tissue")

        return tissue_tab
    
    def show_flatten(self):
        adata = self.selected_adata
        if 'x_flatten' not in adata.obs.columns or 'y_flatten' not in adata.obs.columns:
            logging.warning(
                "x_flatten and y_flatten columns not found in adata.obs. "
            )
            return
        else:
            x_flatten = adata.obs['y_flatten'].values
            y_flatten = adata.obs['x_flatten'].values
            xy_flatten = np.column_stack((x_flatten, y_flatten))
            if hasattr(self, 'cell_gene_color'):
                self.viewer.add_points(
                    xy_flatten,
                    size=10,
                    face_color=list(self.cell_gene_color.values()),
                    features=adata.obs,
                    name='flatten',
                )
            else:
                self.viewer.add_points(
                    xy_flatten,
                    size=10,
                    face_color=[self.tissue_color_map[t] for t in adata.obs[self.embryo.tissue_name]],
                    features=adata.obs,
                    name='flatten',
                )

    def slice_tab(self):
        def select_slice():
            if 'slices' in self.embryo.adata.obs.columns:
                self.slice_select = widgets.Select(
                    choices=self.embryo.adata.obs['slices'].unique().tolist(),
                    value=self.embryo.adata.obs['slices'].unique().tolist(),
                )
            elif 'orig.ident' in self.embryo.adata.obs.columns:
                self.slice_select = widgets.Select(
                    choices=self.embryo.adata.obs['orig.ident'].unique().tolist(),
                    value=self.embryo.adata.obs['orig.ident'].unique().tolist(),
                )
            else:
                self.slice_select = widgets.Select(
                    choices=["No slice available"],
                    value="No slice available",
                    enabled=False,
                )
            run_slice_filter = widgets.FunctionGui(
                self.tissue_filter,
                call_button="Select Slice",
                layout="vertical",
            )
            select_slice_container = widgets.Container(
                widgets=[self.slice_select, run_slice_filter],
                layout="vertical",
            )
            return select_slice_container

        slice_tab = QTabWidget()
        slice_tab.setObjectName("Slice")
        slice_tab.addTab(select_slice().native, "Select Slice")

        return slice_tab
    
    def germ_layer_tab(self):
        def select_germ_layer():
            if 'germ_layer' in self.embryo.adata.obs.columns:
                self.germ_layer_select = widgets.Select(
                    choices=self.embryo.adata.obs['germ_layer'].unique().tolist(),
                    value=self.embryo.adata.obs['germ_layer'].unique().tolist(),
                )
            else:
                self.germ_layer_select = widgets.Select(
                    choices=["No slice available"],
                    value="No slice available",
                    enabled=False,
                )
            run_germ_layer_filter = widgets.FunctionGui(
                self.tissue_filter,
                call_button="Select Germ Layer",
                layout="vertical",
            )
            select_germ_layer_container = widgets.Container(
                widgets=[self.germ_layer_select, run_germ_layer_filter],
                layout="vertical",
            )
            return select_germ_layer_container

        germ_layer_tab = QTabWidget()
        germ_layer_tab.setObjectName("Germ Layer")
        germ_layer_tab.addTab(select_germ_layer().native, "Select Germ Layer")

        return germ_layer_tab
    
    def selectXY_tab(self):
        coordinate_3d = self.embryo.coordinate_3d
        x_min, x_max = coordinate_3d[:, 0].min(), coordinate_3d[:, 0].max()
        y_min, y_max = coordinate_3d[:, 1].min(), coordinate_3d[:, 1].max()
        z_min, z_max = coordinate_3d[:, 2].min(), coordinate_3d[:, 2].max()
        def tissue_filter_preview():
            x_min = self.show_x_min.value
            x_max = self.show_x_max.value
            y_min = self.show_y_min.value
            y_max = self.show_y_max.value

            vertices = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ])
            faces = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [3, 2, 6], [3, 6, 7],
                [0, 3, 7], [0, 7, 4],
                [1, 2, 6], [1, 6, 5]
            ])
            self.viewer.add_surface(
                (vertices, faces),
                colormap='gray',
                opacity=0.5,
                shading='flat',
                name = 'preview_box',
            )
        def select_xy():
            self.show_x_min = widgets.FloatSpinBox(
                value=x_min, min=x_min - (abs(x_min) * 0.9), max=x_max * 1.1, step=0.1, label="X Min"
            )
            self.show_x_max = widgets.FloatSpinBox(
                value=x_max, min=x_min - (abs(x_min) * 0.9), max=x_max * 1.1, step=0.1, label="X Max"
            )
            self.show_y_min = widgets.FloatSpinBox(
                value=y_min, min=y_min - (abs(y_min) * 0.9), max=y_max * 1.1, step=0.1, label="Y Min"
            )
            self.show_y_max = widgets.FloatSpinBox(
                value=y_max, min=y_min - (abs(y_min) * 0.9), max=y_max * 1.1, step=0.1, label="Y Max"
            )
            run_xy_filter_preview = widgets.FunctionGui(
                tissue_filter_preview,
                call_button="Preview select XY",
                layout="vertical",
            )
            run_xy_filter = widgets.FunctionGui(
                self.tissue_filter,
                call_button="Select XY",
                layout="vertical",
            )
            select_xy_container = widgets.Container(
                widgets=[
                    self.show_x_min,
                    self.show_x_max,
                    self.show_y_min,
                    self.show_y_max,
                    run_xy_filter_preview,
                    run_xy_filter,
                ],
                layout="vertical",
            )
            return select_xy_container
        
        selectXY_tab = QTabWidget()
        selectXY_tab.setObjectName("Select XY")
        selectXY_tab.addTab(select_xy().native, "Select XY")

        return selectXY_tab
    
    def surface_tab(self):
        def show_surf():
            surf_tissue = self.surf_tissue.value
            adata = self.embryo.adata.copy()
            adata_surf = adata[adata.obs[self.embryo.tissue_name].isin(surf_tissue)]
            points = adata_surf.obsm[self.embryo.coordinate_3d_key]
            pd = PolyData(points)
            mesh = pd.delaunay_3d().extract_surface()
            face_list = list(mesh.faces.copy())
            face_sizes = {}
            faces = []
            while 0 < len(face_list):
                nb_P = face_list.pop(0)
                if not nb_P in face_sizes:
                    face_sizes[nb_P] = 0
                face_sizes[nb_P] += 1
                curr_face = []
                for _ in range(nb_P):
                    curr_face.append(face_list.pop(0))
                faces.append(curr_face)
            faces = np.array(faces)
            self.viewer.add_surface(
                (mesh.points, faces),
                colormap=self.tissue_color_map[surf_tissue[0]],
                opacity=0.5,
                name = 'surface_' + surf_tissue[0],
            )
        def select_surf():
            if pyvista:
                surf_label = widgets.Label(value="Choose tissue")
                self.surf_tissue = widgets.Select(
                    choices=self.embryo.all_tissues, value=self.embryo.all_tissues[0]
                )
                select_surf_label = widgets.Container(
                    widgets=[surf_label, self.surf_tissue],
                )
                surf_run = widgets.FunctionGui(
                    show_surf, 
                    call_button="Compute and show surface"
                )
                surf_container = widgets.Container(
                    widgets=[
                        select_surf_label,
                        surf_run,
                    ],
                    layout="vertical",
                )
            else:
                surf_container = widgets.Container()
                logging.warning(
                    "pyvista is not installed. No surfaces can be generated.\n"
                    "Try pip install pyvista or conda install pyvista to install it."
                )
            return surf_container
        
        surf_tab = QTabWidget()
        surf_tab.setObjectName("Surface")
        surf_tab.addTab(select_surf().native, "Select Surface")
        return surf_tab

    def annotate_tab(self):
        def annotate():
            points = self.viewer.layers.selection.active
            selected_id = list(points.selected_data)
            selected_points = points.data[selected_id]
        
            adata = self.embryo.adata.copy()
            selected_points = np.array(selected_points)
            mask = np.isin(adata.obsm[self.embryo.coordinate_3d_key], selected_points).all(axis=1)

            if column_name.value not in adata.obs.columns:
                adata.obs[column_name.value] = adata.obs[self.embryo.tissue_name].copy()
                
            if not adata.obs[column_name.value].dtype.name == 'category':
                adata.obs[column_name.value] = adata.obs[column_name.value].astype('str')
                adata.obs[column_name.value] = adata.obs[column_name.value].astype('category')

            cat = adata.obs[column_name.value].cat.categories
            if cluster_anno.value not in cat:
                new_cat = sorted(list(cat) + [cluster_anno.value])
                adata.obs[column_name.value] = adata.obs[column_name.value].cat.set_categories(new_cat)
            adata.obs.loc[mask, column_name.value] = cluster_anno.value

            self.embryo.adata_anno = adata
            print(adata.obs[column_name.value].value_counts())

        def save_annotations():
            path = save_path.value
            self.embryo.adata_anno.obs[column_name.value] = self.embryo.adata_anno.obs[column_name.value].cat.remove_unused_categories()
            self.embryo.adata_anno.write_h5ad(path)
            print(f"adata saved to {path}")

        path = os.getcwd()
        run_annotation = widgets.FunctionGui(
            annotate, call_button="Annotation to selected points"
        )
        run_save = widgets.FunctionGui(
            save_annotations, call_button="Save Annotations"
        )
        cluster_anno = widgets.LineEdit(value='cluster 1',label="Cluster Name")
        column_name = widgets.LineEdit(value='new column 1', label="Column Name")
        save_path = widgets.LineEdit(value=os.path.join(path, 'napari.h5ad'), label="Save Path")
        annotation_container = widgets.Container(
            widgets=[
                cluster_anno,
                column_name,
                run_annotation,
                save_path,
                run_save
            ],
        )
        annotate_tab = QTabWidget()
        annotate_tab.setObjectName("Annotate")
        annotate_tab.addTab(annotation_container.native, "Annotate")
        return annotate_tab

    def one_gene_tab(self):
        adata = self.selected_adata
        viewer = self.viewer
        def show_gene():
            """
            Colour cells according to their gene expression
            """
            select_gene = self.select_gene.value
            if select_gene is None or select_gene == "Select a gene to see similar genes" or select_gene == []:
                gene = self.gene.value
            else:
                self.gene.value = select_gene
                gene = select_gene
            gene_exp = safe_toarray(adata[:, gene].X)[:,0]
            min_exp, max_exp = gene_exp.min(), gene_exp.max()
            gene_exp_norm = (gene_exp - min_exp) / (max_exp - min_exp)
            features = {}
            features['gene_exp'] = gene_exp
            viewer.add_points(
                adata.obsm[self.embryo.coordinate_3d_key],
                size=10,
                face_color = cc.cm["CET_L4"](gene_exp_norm / 2),
                features= features,
                name=f'gene_{gene}',
            )
            self.cell_gene_color = {
                i: color
                for i, color in zip(list(range(len(gene_exp))), cc.cm["CET_L4"](gene_exp_norm / 2))
            }

        def show_similar_genes():
            all_gene_exp = safe_toarray(adata.X).T
            gene_exp = safe_toarray(adata[:, self.gene.value].X)[:,0]
            similarity_list = cosine_similarity(all_gene_exp, gene_exp.reshape(1, -1)).flatten()
            similarity_dict = {
                adata.var_names[i]: similarity_list[i]
                for i in range(len(adata.var_names))
            }
            sorted_genes = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
            self.similar_genes = [gene for gene, _ in sorted_genes if gene != self.gene.value][:10]
            self.select_gene.choices = self.similar_genes
        
        def container():
            """
            Create a container for the one gene tab.
            """
            self.gene = widgets.LineEdit(
                value=adata.var_names[0],
                label="Gene",
            )
            run_show_flatten = widgets.FunctionGui(
                self.show_flatten,
                call_button="Show Flatten",
                layout="vertical",
            )
            run_show_gene = widgets.FunctionGui(
                show_gene,
                call_button="Show Gene",
                layout="vertical",
            )
            run_show_similar_genes = widgets.FunctionGui(
                show_similar_genes,
                call_button="Show Similar Genes",
                layout="vertical",
            )
            if hasattr(self, 'similar_genes'):
                similar_genes = self.similar_genes
            else:
                similar_genes = [self.gene.value]
            self.select_gene = widgets.Select(
                choices=similar_genes,
                value=similar_genes[0],
                label="Select Similar Gene\nif chose a choice\nshow exp immediately",
                )
            run_show_similar_genes_exp = widgets.FunctionGui(
                show_gene,
                call_button="Show Similar Gene Exp",
                layout="vertical",
            )
            container = widgets.Container(
                widgets=[
                    self.gene,
                    run_show_gene,
                    run_show_flatten,
                    run_show_similar_genes,
                    self.select_gene,
                    run_show_similar_genes_exp,
                    ],
                layout="vertical",
            )
            return container
        
        one_gene_tab = QTabWidget()
        one_gene_tab.addTab(container().native,'Select Gene')

        return one_gene_tab
    
    def two_genes_tab(self):
        adata = self.selected_adata
        viewer = self.viewer
        def show_two_genes():
            """
            Colour cells according to their gene expression
            """
            gene_1 = self.gene_1.value
            gene_2 = self.gene_2.value
            gene_exp_1 = safe_toarray(self.selected_adata[:, gene_1].X)[:,0]
            gene_exp_2 = safe_toarray(self.selected_adata[:, gene_2].X)[:,0]
            min_exp_1, max_exp_1 = gene_exp_1.min(), gene_exp_1.max()
            min_exp_2, max_exp_2 = gene_exp_2.min(), gene_exp_2.max()
            with np.errstate(divide='ignore', invalid='ignore'):
                gene_exp_norm_1 = np.clip((gene_exp_1 - min_exp_1) / (max_exp_1 - min_exp_1), 0, 1)
                gene_exp_norm_2 = np.clip((gene_exp_2 - min_exp_2) / (max_exp_2 - min_exp_2), 0, 1)
            colors = np.zeros((len(gene_exp_norm_1), 3))
            colors[:, 0] = gene_exp_norm_1
            colors[:, 1] = gene_exp_norm_2
            features = {}
            features[f'gene_exp_{gene_1}'] = gene_exp_1
            features[f'gene_exp_{gene_2}'] = gene_exp_2
            viewer.add_points(
                self.selected_adata.obsm[self.embryo.coordinate_3d_key],
                size=10,
                face_color=colors,
                features=features,
                name=f'genes_{gene_1}_{gene_2}',
            )
            self.cell_gene_color = {
                i: color
                for i,color in zip(
                    list(range(len(gene_exp_norm_1))),
                    colors,
                )
            }

        def container():
            """
            Create a container for the two genes tab.
            """
            self.gene_1 = widgets.LineEdit(
                value=adata.var_names[0],
                label="Gene 1",
            )
            self.gene_2 = widgets.LineEdit(
                value=adata.var_names[1],
                label="Gene 2",
            )
            run_show_two_genes = widgets.FunctionGui(
                show_two_genes,
                call_button="Show Two Genes",
                layout="vertical",
            )
            run_show_flatten = widgets.FunctionGui(
                self.show_flatten,
                call_button="Show Flatten",
            )
            container = widgets.Container(
                widgets=[
                    self.gene_1,
                    self.gene_2,
                    run_show_two_genes,
                    run_show_flatten,
                ],
                layout="vertical",
            )
            return container
        
        two_genes_tab = QTabWidget()
        two_genes_tab.addTab(container().native, 'Select Two Genes')
        return two_genes_tab
    
    def three_genes_tab(self):
        adata = self.selected_adata
        viewer = self.viewer
        def show_three_genes():
            """
            Colour cells according to their gene expression
            """
            gene_1 = self.gene_1.value
            gene_2 = self.gene_2.value
            gene_3 = self.gene_3.value
            gene_exp_1 = safe_toarray(adata[:, gene_1].X)[:,0]
            gene_exp_2 = safe_toarray(adata[:, gene_2].X)[:,0]
            gene_exp_3 = safe_toarray(adata[:, gene_3].X)[:,0]
            min_exp_1, max_exp_1 = gene_exp_1.min(), gene_exp_1.max()
            min_exp_2, max_exp_2 = gene_exp_2.min(), gene_exp_2.max()
            min_exp_3, max_exp_3 = gene_exp_3.min(), gene_exp_3.max()
            with np.errstate(divide='ignore', invalid='ignore'):
                gene_exp_norm_1 = np.clip((gene_exp_1 - min_exp_1) / (max_exp_1 - min_exp_1), 0, 1)
                gene_exp_norm_2 = np.clip((gene_exp_2 - min_exp_2) / (max_exp_2 - min_exp_2), 0, 1)
                gene_exp_norm_3 = np.clip((gene_exp_3 - min_exp_3) / (max_exp_3 - min_exp_3), 0, 1)
            colors = np.zeros((len(gene_exp_norm_1), 3))
            colors[:, 0] = gene_exp_norm_1
            colors[:, 1] = gene_exp_norm_2
            colors[:, 2] = gene_exp_norm_3
            features = {}
            features[f'gene_exp_{gene_1}'] = gene_exp_1
            features[f'gene_exp_{gene_2}'] = gene_exp_2
            features[f'gene_exp_{gene_3}'] = gene_exp_3
            viewer.add_points(
                adata.obsm[self.embryo.coordinate_3d_key],
                size=10,
                face_color=colors,
                features=features,
                name=f'genes_{gene_1}_{gene_2}_{gene_3}',
            )
            self.cell_gene_color = {
                i: color
                for i, color in zip(
                    list(range(len(gene_exp_norm_1))),
                    colors,
                )
            }
        def container():
            """
            Create a container for the three genes tab.
            """
            self.gene_1 = widgets.LineEdit(
                value=adata.var_names[0],
                label="Gene 1",
            )
            self.gene_2 = widgets.LineEdit(
                value=adata.var_names[1],
                label="Gene 2",
            )
            self.gene_3 = widgets.LineEdit(
                value=adata.var_names[2],
                label="Gene 3",
            )
            run_show_three_genes = widgets.FunctionGui(
                show_three_genes,
                call_button="Show Three Genes",
                layout="vertical",
            )
            run_show_flatten = widgets.FunctionGui(
                self.show_flatten,
                call_button="Show Flatten",
            )
            container = widgets.Container(
                widgets=[
                    self.gene_1,
                    self.gene_2,
                    self.gene_3,
                    run_show_three_genes,
                    run_show_flatten,
                ],
                layout="vertical",
            )
            return container
        
        three_genes_tab = QTabWidget()
        three_genes_tab.addTab(container().native, 'Select Three Genes')
        return three_genes_tab
    
    def umap_tab(self):
        """
        Function that builds the qt container for the umap
        """
        gene_label = widgets.Label(value="Choose gene")
        gene = widgets.LineEdit(value=self.select_gene.value)

        tissues_label = widgets.Label(value="Display tissues umap")
        tissues = widgets.CheckBox(value=False)

        variable_genes_label = widgets.Label(value="Take only variable genes")
        variable_genes = widgets.CheckBox(value=True)

        stats_label = widgets.Label(value="Statistic for\nchoosing distributions")
        stats = widgets.RadioButtons(
            choices=["Standard Deviation", "Mean", "Median"],
            value="Standard Deviation",
        )
        self.umap_selec = UmapSelection(
            self.viewer,
            self.embryo,
            gene,
            tissues,
            stats,
            variable_genes,
            self.tissue_color_map,
        )
        umap_run = widgets.FunctionGui(
            self.umap_selec.run, call_button="Show gene on Umap", name=""
        )

        gene_container = widgets.Container(
            widgets=[gene_label, gene], labels=False, layout="horizontal"
        )
        variable_genes_container = widgets.Container(
            widgets=[variable_genes_label, variable_genes],
            labels=False,
            layout="horizontal",
        )
        tissues_container = widgets.Container(
            widgets=[tissues_label, tissues], labels=False, layout="horizontal"
        )
        stats_container = widgets.Container(
            widgets=[stats_label, stats], labels=False, layout="horizontal"
        )
        umap_container = widgets.Container(
            widgets=[
                gene_container,
                tissues_container,
                variable_genes_container,
                stats_container,
                umap_run,
            ],
            labels=False,
        )
        umap_tab = QTabWidget()
        umap_tab.addTab(umap_container.native, "Umap")
        return umap_tab
    
    def Moran_tab(self):
        """
        Function that builds the qt container for the Moran's I
        """
        adata = self.selected_adata
        viewer = self.viewer
        def compute_moran():
            sq.gr.spatial_neighbors(adata, spatial_key=self.embryo.coordinate_3d_key,key_added='spatial')
            Moranres = sq.gr.spatial_autocorr(adata, mode="moran", copy = True)
            moran_25_idx = Moranres["I"][:25].index.tolist()
            moran_25_values = Moranres["I"][:25].values.tolist()
            gene_exp = safe_toarray(adata[:, moran_25_idx].X)
            gene_count_cell = np.count_nonzero(gene_exp,axis=0)
            choice = [f"{gene} ({value})[count:{cell}]" for gene, value,cell in zip(moran_25_idx, moran_25_values,gene_count_cell)]
            self.moran_gene.choices = choice
        
        def plot_moran():
            if isinstance(self.moran_gene.value, list) and len(self.moran_gene.value) > 0:
                gene_str = self.moran_gene.value[0]
            else:
                gene_str = self.moran_gene.value
            gene = gene_str.split(' (')[0]
            gene_exp = safe_toarray(adata[:, gene].X)[:,0]
            min_exp, max_exp = gene_exp.min(), gene_exp.max()
            gene_exp_norm = (gene_exp - min_exp) / (max_exp - min_exp)
            features = {}
            features['gene_exp'] = gene_exp
            viewer.add_points(
                adata.obsm[self.embryo.coordinate_3d_key],
                size=10,
                face_color = cc.cm["CET_L4"](gene_exp_norm / 2),
                features= features,
                name=f'gene_{gene}',
            )
            self.cell_gene_color = {
                i: color
                for i, color in zip(list(range(len(gene_exp))), cc.cm["CET_L4"](gene_exp_norm / 2))
            }
        def show_moran():
            run_moran = widgets.FunctionGui(
                compute_moran,
                call_button="Show top 10 Moran's I",
                layout="vertical",
            )
            self.moran_gene = widgets.Select(
                choices=[''],
                value='',
                label="Select a gene to see exp",
            )
            run_show_moran = widgets.FunctionGui(
                plot_moran,
                call_button="Show top 10 gene exp",
                layout="vertical",
            )
            moran_container = widgets.Container(
                widgets=[
                    run_moran,
                    self.moran_gene,
                    run_show_moran,
                ],
                layout="vertical",
            )
            return moran_container

        moran_tab = QTabWidget()
        moran_tab.addTab(show_moran().native, "Moran's I")

        return moran_tab
    
    def diff_exp_tab(self):
        """
        Function that builds the qt container for the differential expression
        """
        adata = self.selected_adata
        viewer = self.viewer
        def compute_diff_exp():
            diff_tissue = self.diff_tissue.value
            if diff_tissue[0] in adata.obs[self.embryo.tissue_name].unique():
                diff_adata = adata[adata.obs[self.embryo.tissue_name] == diff_tissue[0]]
            else:
                logging.warning(f"Tissue {diff_tissue} not found in adata.obs[{self.embryo.tissue_name}].")
                return
            diff_gene_avg = safe_toarray(diff_adata.X).mean(axis=0)
            all_gene_avg = safe_toarray(adata.X).mean(axis=0)
            epsilon = 0.0001
            SES = np.log2((diff_gene_avg + epsilon) / (all_gene_avg + epsilon))
            SES_dict = {
                adata.var_names[i]: SES[i]
                for i in range(len(adata.var_names))
            }
            sorted_genes = sorted(SES_dict.items(), key=lambda x: x[1], reverse=True)
            diff_25_idx = [gene for gene, _ in sorted_genes[:25]]
            diff_25_values = [SES_dict[gene] for gene in diff_25_idx]
            gene_count_cell = np.count_nonzero(safe_toarray(adata[:, diff_25_idx].X), axis=0)
            self.diff_exp_gene.choices = [f"{gene} ({value})[count:{cell}]" for gene, value,cell in zip(diff_25_idx, diff_25_values,gene_count_cell)]
        def plot_diff_exp():
            if isinstance(self.diff_exp_gene.value, list) and len(self.diff_exp_gene.value) > 0:
                gene_str = self.diff_exp_gene.value[0]
            else:
                gene_str = self.diff_exp_gene.value
            gene = gene_str.split(' (')[0]
            gene_exp = safe_toarray(adata[:, gene].X)[:,0]
            min_exp, max_exp = gene_exp.min(), gene_exp.max()
            gene_exp_norm = (gene_exp - min_exp) / (max_exp - min_exp)
            features = {}
            features['gene_exp'] = gene_exp
            viewer.add_points(
                adata.obsm[self.embryo.coordinate_3d_key],
                size=10,
                face_color = cc.cm["CET_L4"](gene_exp_norm / 2),
                features= features,
                name=f'gene_{gene}',
            )
            self.cell_gene_color = {
                i: color
                for i, color in zip(list(range(len(gene_exp))), cc.cm["CET_L4"](gene_exp_norm / 2))
            }
        
        def show_diff_exp():
            self.diff_tissue = widgets.Select(
                choices=self.embryo.all_tissues,
                value=self.embryo.all_tissues[0],
                label="Select tissue for differential expression",
                allow_multiple=False,
            )
            run_diff_exp = widgets.FunctionGui(
                compute_diff_exp,
                call_button="Show top 10 Differential Expression",
                layout="vertical",
            )
            self.diff_exp_gene = widgets.Select(
                choices=[''],
                value='',
                label="Select a gene to see exp",
            )
            run_show_diff_exp = widgets.FunctionGui(
                plot_diff_exp,
                call_button="Show top 10 gene exp",
                layout="vertical",
            )
            diff_exp_container = widgets.Container(
                widgets=[
                    self.diff_tissue,
                    run_diff_exp,
                    self.diff_exp_gene,
                    run_show_diff_exp,
                ],
                layout="vertical",
            )
            return diff_exp_container

        diff_exp_tab = QTabWidget()
        diff_exp_tab.addTab(show_diff_exp().native, "Diff Exp")

        return diff_exp_tab
    def create_widget(self):
        '''
        Create UI
        '''
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        main_tab = QTabWidget()

        tab_1 = QTabWidget()
        tab_1.addTab(self.legend_tab(), "Legend")
        tab_1.addTab(self.tissue_tab(), "Tissue")
        tab_1.addTab(self.slice_tab(), "Slice")
        tab_1.addTab(self.germ_layer_tab(), "Germ Layer")
        tab_1.addTab(self.selectXY_tab(), "Select XY")
        tab_1.addTab(self.surface_tab(), "Surface")
        tab_1.addTab(self.annotate_tab(), "Annotate")
        main_tab.addTab(tab_1, "Visualization")

        tab_2 = QTabWidget()
        tab_2.addTab(self.one_gene_tab(), "1 Gene")
        tab_2.addTab(self.two_genes_tab(), "2 Genes")
        tab_2.addTab(self.three_genes_tab(), "3 Genes")
        # tab_2.addTab(self.umap_tab(), "UMAP")
        tab_2.addTab(self.Moran_tab(), "Moran's I")
        tab_2.addTab(self.diff_exp_tab(), "Diff Exp")
        main_tab.addTab(tab_2, "Analysis")

        # tab_3 = QTabWidget()
        # main_tab.addTab(tab_3, "Scanpy plots")

        layout.addWidget(main_tab)
        self.viewer.window.add_dock_widget(
            container, name="Embryo Display", area="right"
        )

    def display(self):
        """
        Display the Embryo data in the napari viewer.
        """
        adata = self.selected_adata
        tissue_types = adata.obs[self.embryo.tissue_name].astype(str).values
        self.viewer.dims.ndisplay = 3
        position = adata.obsm[self.embryo.coordinate_3d_key]
        self.viewer.add_points(
            position,
            size=10,
            face_color=[self.tissue_color_map[t] for t in tissue_types],
            features=adata.obs,
        )

    def __init__(self, viewer, embryo):
        """
        Initialize the DisplayEmbryo object with the given parameters.

        Args:
            embryo (Embryo): An instance of the Embryo class.
            viewer (napari.Viewer): The napari viewer instance.
        """
        self.embryo = embryo
        self.viewer = viewer
        self.selected_adata = self.embryo.adata
    
        if self.selected_adata.obsm[self.embryo.coordinate_3d_key].shape[1] != 3 and 'z' in self.selected_adata.obs.columns:
            self.selected_adata.obsm[self.embryo.coordinate_3d_key] = np.column_stack(
                [
                    self.selected_adata.obsm[self.embryo.coordinate_3d_key][:, 0],
                    self.selected_adata.obsm[self.embryo.coordinate_3d_key][:, 1],
                    self.selected_adata.obs['z'].values,
                ]
            )
        self.create_widget()
        self.display()