"""
This file is subject to the terms and conditions defined in
file 'LICENCE', which is part of this source code package.
Author: Leo Guignard (leo.guignard...@AT@...univ-amu.fr)
"""
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget,QMessageBox,QFileDialog
from magicgui import widgets
from ._umap_selection import UmapSelection
from ._utils import error_points_selection, safe_toarray
from napari.utils.colormaps import ALL_COLORMAPS, Colormap
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib import cm, colors
import numpy as np
from random import sample
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
import os
from matplotlib import pyplot as plt

import scanpy as sc
import squidpy as sq
from scipy.spatial import distance
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


class DisplayEmbryo:
    """
    A class to build the plugin to display spatial transcriptomics

    Within this plugin, it is important to understand the way the data
    is stored. The way the data is stored is a mix of historical reason and
    a vague effort to make the whole plugin somewhat optimized in time and
    space (?).
    Note that it is simpler by the author definition. This definition
    will likely not be shared by all.

    The structure is the following:
        - the point layer is the one from napari usually accessed with:
            `points = self.viewer.layers.selection.active`
        - there is some metadata information:
            points.metadata['gene']: gene currently shown, `None` if none is shown
            points.metadata['2genes']: 2 genes and the parameters of visualization
                for the 2 genes currently shown, `None` if 2 genes are not shown
            points.metadata['gene_min_max']: min and max values for the gene shown
                if a single gene is shown
            points.metadata['2genes_params']: computed parameters for showing
                the coexpression of two genes
        - there is a new feature:
            points.features['current_view']: refers to the set of cells in the
                current view, whether they are shown or not. It is important
                when computing local gene expression

    """

    def disp_legend(self):
        """
        Display the legend for the colors displayed
        """
        # Get the points and make sure they are correctly selected
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Not ideally build a matplotlib figure to show the legend
        # For different mode, different figure type.
        # Not elegant, not efficient, not explained :/
        with plt.style.context("dark_background"):
            static_canvas = FigureCanvas()
            fig = static_canvas.figure
            ax = fig.add_subplot()
            if (
                points.metadata["gene"] is None
                and points.metadata["2genes"] is None
            ):
                tissues = set(
                    [
                        self.embryo.tissue[c]
                        for c in points.properties["cells"][points.shown]
                    ]
                )
                for t in tissues:
                    ax.plot(
                        [],
                        "o",
                        c=self.color_map_tissues[t],
                        label=self.embryo.corres_tissue.get(t, f"{t}"),
                    )
                ax.legend()
                ax.set_axis_off()
            elif points.metadata["2genes"] is None:
                if points.face_contrast_limits is None:
                    m, M = 0, 1
                else:
                    m, M = points.face_contrast_limits
                if points.face_colormap.name in plt.colormaps() or isinstance(
                    points.face_colormap, Colormap
                ):
                    if points.face_colormap.name in plt.colormaps():
                        cmap = points.face_colormap.name
                    else:
                        cmap = points.mplcmap
                    fig.colorbar(
                        cm.ScalarMappable(
                            norm=colors.Normalize(m, M),
                            cmap=cmap,
                        ),
                        label=points.metadata["gene"] + ", normalized values",
                        ax=ax,
                    )
                    min_, max_ = points.metadata["gene_min_max"]
                    min_ = (max_ - min_) * m + min_
                    max_ = (max_ - min_) * M + min_
                    fig.colorbar(
                        cm.ScalarMappable(
                            norm=colors.Normalize(min_, max_),
                            cmap=cmap,
                        ),
                        label=points.metadata["gene"] + ", original values",
                        ax=ax,
                    )
                else:
                    fig.text(
                        0,
                        0,
                        (
                            "Could not find the colormap "
                            f"`{points.face_colormap.name}` "
                            "to plot the legend"
                        ),
                    )
                ax.set_axis_off()
            else:
                scale_square = np.zeros((256, 256, 3))
                max_g1, max_g2, norm, on_channel = points.metadata[
                    "2genes_params"
                ]
                V1 = np.linspace(0, max_g1, 256)
                V2 = np.linspace(0, max_g2, 256)
                VS = np.array([V1, V2])
                VS = norm(VS)
                VS[VS < 0] = 0
                VS[1 < VS] = 1
                scale_square[..., np.where(1 - on_channel)[0][0]] = VS[0]
                for axes in np.where(on_channel)[0]:
                    scale_square[..., axes] = VS[1].reshape(-1, 1)
                ax.imshow(scale_square.swapaxes(1, 0), origin="lower")
                recap_g1 = lambda x: x * 255 / max_g1
                recap_g2 = lambda x: x * 255 / max_g2
                vals_g1 = np.arange(np.floor(max_g1) + 1, dtype=int)
                vals_g2 = np.arange(np.floor(max_g2) + 1, dtype=int)
                ax.set_xticks(recap_g1(vals_g1))
                ax.set_yticks(recap_g2(vals_g2))
                ax.set_xticklabels(vals_g1)
                ax.set_yticklabels(vals_g2)
                ax.set_xlabel(points.metadata["2genes"][1])
                ax.set_ylabel(points.metadata["2genes"][0])
            
            fig.tight_layout()
            # if self.show:
            #     plt.show()
            static_canvas.toolbar = NavigationToolbar(
                static_canvas, static_canvas.parent()
            )
            fig_can = self.viewer.window.add_dock_widget(
                static_canvas, name="Legend"
            )
            V_box = QWidget()
            V_box.setLayout(QVBoxLayout())
            V_box.layout().addWidget(fig_can)
            V_box.layout().addWidget(static_canvas.toolbar)
            self.tab1.removeTab(self.tab1.nb_tabs + 1)
            self.tab1.addTab(V_box, "Legend")

    def show_tissues(self):
        """
        Color cells according to the tissue they belong to
        """
        # Get the points and make sure they are correctly selected
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # If necessary, change the color of the cells
        if (
            points.metadata["gene"] is not None
            or points.metadata["2genes"] is not None
            or self.color_map_tissues != self.original_color_map_tissues
        ):
            self.color_map_tissues = self.original_color_map_tissues.copy()
            points.face_color = [
                self.color_map_tissues[self.embryo.tissue[c]]
                for c in points.properties["cells"]
            ]
            points.face_color_mode = "direct"
            points.metadata["gene"] = None
            points.metadata["2genes"] = None
            points.metadata["3genes"] = None
        points.refresh()

    def recolor_tissues(self):
        # Get the points and make sure they are correctly selected
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Change the color of the cells
        tissues = set(
            [
                self.embryo.tissue[c]
                for c in points.properties["cells"][points.shown]
            ]
        )
        nb_tissues = len(tissues)+1
        subset_map = {t: i+1 for i, t in enumerate(sample(tissues, len(tissues)))}
        self.color_map_tissues = {
            t: cm.tab20(subset_map.get(t, 0) / nb_tissues)
            for t in self.embryo.all_tissues
        }

        points.face_color = [
            self.color_map_tissues[self.embryo.tissue[c]]
            for c in points.properties["cells"]
        ]
        points.face_color_mode = "direct"
        points.metadata["gene"] = None
        points.metadata["2genes"] = None
        points.metadata["3genes"] = None
        points.refresh()

    def add_flattend(self):


        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        cells = sorted(self.embryo.all_cells)
        y_values = self.embryo.anndata.obs['y_flatten'].values

        XY_flatten = np.column_stack((self.embryo.anndata.obs['x_flatten'], y_values))
        positions = [XY_flatten[c] for c in cells]
        shown = points.features["current_view"]
        if not any(shown):
            shown = [True] * len(cells)
        properties = {"cells": cells}

        properties["gene"] = [0 for _ in cells]
        # if 0 < len(self.embryo.all_tissues.difference(self.color_map_tissues)):
        #     nb_tissues = len(self.embryo.all_tissues)
        #     self.color_map_tissues = {
        #         v: cm.tab20(i / nb_tissues)
        #         for i, v in enumerate(self.embryo.all_tissues)
        #     }
        #

        self.original_color_map_tissues = self.color_map_tissues.copy()
        colors_rgb = [
            self.color_map_tissues.get(self.embryo.tissue[c], [0, 0, 0])
            for c in cells
        ]

        self.viewer.dims.ndisplay = 2
        points = self.viewer.add_points(
            positions,
            face_color=colors_rgb,
            properties=properties,
            metadata={"gene": None, "2genes": None,"3genes": None},
            shown=shown,
            size=15,
        )

    def select_tissues(self):
        """
        Display a set of tissues according to user selection
        """
        # Get the points and make sure they are correctly selected
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Get the cells that belong to the tissue selected and display them
        # The cells from the selected tissue define the `current_view`
        # selected_layers = self.select_layers_choices.value
        # selected_tissues = self.select_tissues_choices.value
        # selected_germ_layers = self.select_germ_layers_choices.value
        #
        # tissues = []
        # for item in selected_tissues:
        #     if item in selected_layers and item in selected_germ_layers:
        #         tissues.append(item)
        #
        # tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}
        # tissues_to_plot = []
        # for t in tissues:
        #     if t in tissue_to_num:
        #         tissues_to_plot.append(tissue_to_num[t])
        #     else:
        #         tissues_to_plot.append(int(t))
        # shown = [
        #     self.embryo.tissue[c] in tissues_to_plot
        #     for c in points.properties["cells"]
        # ]
        #
        # points.shown = shown
        selected_layers = self.select_layers_choices.value
        selected_tissues = self.select_tissues_choices.value
        selected_germ_layers = self.select_germ_layers_choices.value



        tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}

        tissues_to_plot = []
        for t in selected_tissues:
            if t in tissue_to_num:
                tissues_to_plot.append(tissue_to_num[t])
            else:
                tissues_to_plot.append(int(t))
        for t in selected_layers:
            tissues_to_plot.append(t)
        for t in selected_germ_layers:
            tissues_to_plot.append(t)
        # tissus_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}

        shown1 = [
            self.embryo.anndata.obs['germ_layer'][c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        shown2 = [
            self.embryo.anndata.obs['orig.ident'][c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        shown3 = [
            self.embryo.tissue[c] in tissues_to_plot
            for c in points.properties["cells"]
        ]


        shown_filtered = [a and b and c for a, b, c in zip(shown1, shown2, shown3)]



        index_list = [index for index, value in enumerate(shown_filtered) if value]



        shown_XY =  np.array([False] * self.embryo.anndata.shape[0])

        # 对于每个细胞，检查其'x'和'y'值是否在self.min_X和self.max_X之间，以及是否在self.min_Y和self.max_Y之间
        for i in range(self.embryo.anndata.shape[0]):
            if self.min_X.value <= self.embryo.anndata.obs['x'][i] <= self.max_X.value and self.min_Y.value <= self.embryo.anndata.obs['y'][i] <= self.max_Y.value:
                shown_XY[i] = True

        selected_adata = self.embryo.anndata[shown_filtered]
        global current_adata
        current_adata = selected_adata
        # sq.gr.spatial_neighbors(selected_adata, coord_type='generic', spatial_key='X_spatial')
        # # 获取坐标数据
        # coordinates = selected_adata.obsm['X_spatial']
        #
        # # 计算欧式距离矩阵
        # distance_matrix = distance.cdist(coordinates, coordinates, 'euclidean')
        #
        # # 计算每个细胞到中心的平均距离
        # mean_distances = np.mean(distance_matrix, axis=1)
        #
        # # 找到远离"大部队"的点的索引
        # far_away_cells = np.argsort(mean_distances)[::-1][:self.min_edges.value]
        if 'z'  in selected_adata.obs:
            z_values = selected_adata.obs['z'].values
            #    将 'adata.obsm['X_spatial']'  'z_values' 沿第二个轴（列）合并
            combined_data = np.column_stack((selected_adata.obsm['X_spatial'], z_values*self.embryo.z_resolusion))
        # 创建新的 AnnData 对象，将 'combined_data' 分配给 'adata.obsm['X_spatial']
            selected_adata.obsm['X_spatial'] = combined_data
        coordinates = selected_adata.obsm['X_spatial']

        print(coordinates)
        # Apply the threshold to discard some cells
        # if self.choose_how2cal.value != 0:
        #     if self.choose_how2cal.value == "High distance to center of mass":
        #         center_of_mass = np.mean(coordinates, axis=0)
        #         mean_distances = np.linalg.norm(coordinates - center_of_mass, axis=1)
        #     else:
        #         node_ids = list(range(len(coordinates)))
        #         gg = self.embryo.build_gabriel_graph(
        #             node_ids, coordinates, data_struct="adj-mat", dist=True
        #         )
        #         mean_distances = gg.mean(axis=0)
        #
        #     far_away_cells = np.argsort(mean_distances)[::-1][:self.min_edges.value]
        #
        #
        # # 创建布尔列表，初始化为True
        # mask = np.array([True] * len(coordinates))
        #
        #     # 将度数最小的m个细胞置为False
        # mask[far_away_cells] = False

        if self.choose_how2cal.value != 0:
            if self.choose_how2cal.value == "High distance to center of mass":
                center_of_mass = np.mean(coordinates, axis=0)
                mean_distances = np.linalg.norm(coordinates - center_of_mass, axis=1)
                mask = np.array([True] * len(coordinates))
                far_away_cells = np.argsort(mean_distances)[::-1][:self.min_edges.value]
                    # 将度数最小的m个细胞置为False
                mask[far_away_cells] = False

            else:
                node_ids = list(range(len(coordinates)))

                # 创建布尔列表，初始化为True
                mask = np.array([True] * len(coordinates))

                while self.min_edges.value > 0:
                    gg = self.embryo.build_gabriel_graph(
                        node_ids, coordinates, data_struct="adj-mat", dist=True
                    )
                    mean_distances = gg.mean(axis=0)

                    # 找到剩余细胞中mean_distances最大的细胞
                    max_distance_cell = np.argmax(mean_distances * mask)

                    # 如果没有更多的细胞可以剪枝了，退出循环
                    if mean_distances[max_distance_cell] == 0:
                        break

                    # 将当前最大的细胞置为False
                    mask[max_distance_cell] = False
                    self.min_edges.value -= 1

                    # 重新构建坐标
                    coordinates = np.delete(coordinates, max_distance_cell, axis=0)
                    node_ids = np.delete(node_ids, max_distance_cell)

        # filterbyedgeslist = find_and_mask_low_degree_cells(selected_adata,self.min_edges.value)
        for i in range(len(index_list)):
            shown_filtered[index_list[i]] = mask[i]

        shown = [a and b for a, b in zip(shown_filtered , shown_XY)]

        points.shown = shown
        points.features["current_view"] = shown

        # Rerun the correct display function with the new set of cells
        if (
            points.metadata["gene"] is None
            and points.metadata["2genes"] is None
        ):
            self.show_tissues()
        elif points.metadata["2genes"] is None:
            self.show_gene()
        elif points.metadata["3genes"] is None:
            self.show_two_genes()
        else:
            self.show_three_genes()

    def select_layers(self):
        """
        Display a set of layers according to user selection
        """
        # Get the points and make sure they are correctly selected
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Get the cells that belong to the tissue selected and display them
        # The cells from the selected tissue define the `current_view`
        # selected_layers = self.select_layers_choices.value
        # selected_tissues = self.select_tissues_choices.value
        # selected_germ_layers = self.select_germ_layers_choices.value
        #
        # tissues = []
        # for item in selected_tissues:
        #     if item in selected_layers and item in selected_germ_layers:
        #         tissues.append(item)
        # # tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}
        # tissues_to_plot = []
        # for t in tissues:
        #         tissues_to_plot.append(t)
        # shown = [
        #     self.embryo.anndata.obs['orig.ident'][c] in tissues_to_plot
        #     for c in points.properties["cells"]
        # ]
        #
        #
        # points.shown = shown

        selected_layers = self.select_layers_choices.value
        selected_tissues = self.select_tissues_choices.value
        selected_germ_layers = self.select_germ_layers_choices.value



        tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}

        tissues_to_plot = []
        for t in selected_tissues:
            if t in tissue_to_num:
                tissues_to_plot.append(tissue_to_num[t])
            else:
                tissues_to_plot.append(int(t))
        for t in selected_layers:
            tissues_to_plot.append(t)
        for t in selected_germ_layers:
            tissues_to_plot.append(t)
        # tissus_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}

        shown1 = [
            self.embryo.anndata.obs['germ_layer'][c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        shown2 = [
            self.embryo.anndata.obs['orig.ident'][c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        shown3 = [
            self.embryo.tissue[c] in tissues_to_plot
            for c in points.properties["cells"]
        ]

        # sc.pl.highest_expr_genes(selected_adata, n_top=self.n_top.value)  # 每一个基因在所有细胞中的平均表达量（这里计算了百分比含量）





        shown = [a and b and c for a, b, c in zip(shown1, shown2, shown3)]

        points.shown = shown
        # points.features["current_view"] = shown
        points.features["current_view"] = points.shown.copy()

        # Rerun the correct display function with the new set of cells
        if (
            points.metadata["gene"] is None
            and points.metadata["2genes"] is None
        ):
            self.show_tissues()
        elif points.metadata["2genes"] is None:
            self.show_gene()
        elif points.metadata["3genes"] is None:
            self.show_two_genes()
        else:
            self.show_three_genes()

    def select_Germ_layers(self):
        """
        Display a set of layers according to user selection
        """
        # Get the points and make sure they are correctly selected
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Get the cells that belong to the tissue selected and display them
        # The cells from the selected tissue define the `current_view`
        selected_layers = self.select_layers_choices.value
        selected_tissues = self.select_tissues_choices.value
        selected_germ_layers = self.select_germ_layers_choices.value



        tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}

        tissues_to_plot = []
        for t in selected_tissues:
            if t in tissue_to_num:
                tissues_to_plot.append(tissue_to_num[t])
            else:
                tissues_to_plot.append(int(t))
        for t in selected_layers:
            tissues_to_plot.append(t)
        for t in selected_germ_layers:
            tissues_to_plot.append(t)
        # tissus_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}

        shown1 = [
            self.embryo.anndata.obs['germ_layer'][c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        shown2 = [
            self.embryo.anndata.obs['orig.ident'][c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        shown3 = [
            self.embryo.tissue[c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        # sc.pl.highest_expr_genes(selected_adata, n_top=self.n_top.value)  # 每一个基因在所有细胞中的平均表达量（这里计算了百分比含量）



        shown = [a and b and c for a, b, c in zip(shown1, shown2, shown3)]


        points.shown = shown
        points.features["current_view"] = shown

        # Rerun the correct display function with the new set of cells
        if (
                points.metadata["gene"] is None
                and points.metadata["2genes"] is None
        ):
            self.show_tissues()
        elif points.metadata["2genes"] is None:
            self.show_gene()
        elif points.metadata["3genes"] is None:
            self.show_two_genes()
        else:
            self.show_three_genes()

    def show_surf(self):
        """
        Compute and show the surface of a given tissue
        """
        # Get the points and make sure they exist
        curr_layer = self.viewer.layers.selection.active
        if (
            curr_layer is None
            or curr_layer.as_layer_data_tuple()[-1] != "points"
        ):
            error_points_selection(show=self.show)
            return

        # Makes sure to not recompute surfaces
        tissue = self.select_surf.value
        for l in self.viewer.layers:
            if l.name == f"{tissue}-{self.surf_threshold.value:.0f}":
                return
            if tissue in l.name:
                self.viewer.layers.remove(l)

        # Get the 3D position of the cells of the tissue
        tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}
        if tissue in tissue_to_num:
            t_id = tissue_to_num[tissue]
        elif not isinstance(tissue, int):
            t_id = int(tissue)
        else:
            t_id = tissue
        points = [
            self.embryo.pos_3D[c] for c in self.embryo.cells_from_tissue[t_id]
        ]
        points = np.array(points)
        print(points)
        # Apply the threshold to discard some cells
        if self.surf_threshold.value != 0:
            if self.surf_method.value == "High distance to center of mass":
                center_of_mass = np.mean(points, axis=0)
                dist = np.linalg.norm(points - center_of_mass, axis=1)
            else:
                node_ids = list(range(len(points)))
                gg = self.embryo.build_gabriel_graph(
                    node_ids, points, data_struct="adj-mat", dist=True
                )
                dist = gg.mean(axis=0)
            threshold = np.percentile(dist, 100 - self.surf_threshold.value)
            points = points[dist < threshold]

        # Build and display the surface
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
            colormap=(self.color_map_tissues.get(t_id, [0, 0, 0]),),
            name=f"{tissue}-{self.surf_threshold.value:.0f}",
            opacity=0.6,
        )
        self.viewer.layers.selection.select_only(curr_layer)

    def show_gene(self):
        """
        Colour cells according to their gene expression
        """
        # Get the points and check that we actually got them
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            self.gene_output.value = "Wrong point selection"
            return

        # Get the cells, the different parameters and makes sure that they
        # make sense
        # cell_list = list(self.embryo.all_cells)
        metric = self.metric.value
        gene = self.gene.value
        is_metric = metric in self.embryo.anndata.obs.columns
        if is_metric:
            gene = metric
        if (
            not gene in self.embryo.anndata.obs.columns
            and not gene in self.embryo.anndata.raw.var_names
        ):
            self.gene_output.value = f"Gene '{gene}' not found"
            return

        # Makes sure that we are not recomputing already computed datas
        if gene != points.metadata["gene"]:
            if "current_view" in points.features:
                mask = points.features["current_view"]
            else:
                mask = points.shown

            # Try to build the colors from the quantitative data asked
            if is_metric:
                colors = self.embryo.anndata.obs[metric].to_numpy()
                try:
                    mask &= ~np.isnan(colors)
                except Exception as e:
                    print(colors.dtype)
                    return "Failed"
                points.shown = mask
            else:
                colors = safe_toarray(self.embryo.anndata.raw[:, gene].X)[:, 0]

            # Normalise the data
            min_c, max_c = colors[mask].min(), colors[mask].max()
            if abs(max_c - min_c) < 1e-6:  # 设置一个小的阈值，比如1e-6，来检查min_c和max_c是否接近
                colors = np.ones_like(colors)  # 如果差距太小，将colors全部置为1
            else:
                colors = (colors - min_c) / (max_c - min_c)
            print("min_c=" + str(min_c))
            print("max_c=" + str(max_c))
            colors[~mask] = 0
            points.features["gene"] = colors
            points.metadata["gene_min_max"] = min_c, max_c
            points.metadata["gene"] = gene
        points.metadata["2genes"] = None
        points.metadata["3genes"] = None
        points.edge_color = "black"
        points.face_color = "gene"


        points.face_color_mode = "colormap"
        # points.face_color_mode = cmap


        # points.face_color = "colormap"
        # points.face_color_mode = "direct"

        points.face_contrast_limits = (0, 1)
        points.refresh()
        return f"{points.metadata['gene']} displayed"

    def threshold(self):
        """
        Remove from the view the cells below and above a low and high threshold
        """
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Store the current view for rapid switch between thresholded and not
        if not hasattr(points.features, "current_view"):
            points.features["current_view"] = points.shown.copy()

        # Compute and apply the threshold
        min = self.threshold_low.value
        max = self.threshold_high.value
        if max < min:
            max, min = min, max
        mask = (
            points.features["current_view"]
            & (min <= points.features["gene"])
            & (points.features["gene"] <= max)
        )
        points.shown = mask
        points.refresh()
        nb_selected = np.sum(mask)
        overall = np.sum(points.features["current_view"])
        self.threshold_output.value = (
            f"{nb_selected} cells "
            f"({100*nb_selected/overall:.1f}% of the initial)"
        )
        return

    def threshold_2g(self):
        """
        Remove from the view the cells below and above a low and high threshold
        """
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Store the current view for rapid switch between thresholded and not
        if not hasattr(points.features, "current_view"):
            points.features["current_view"] = points.shown.copy()

        # Compute and apply the threshold

        mask = (
            points.features["current_view"]
            & (0 < points.features["2genes"])
        )
        points.shown = mask
        points.refresh()
        nb_selected = np.sum(mask)
        overall = np.sum(points.features["current_view"])
        self.threshold_output.value = (
            f"{nb_selected} cells "
            f"({100*nb_selected/overall:.1f}% of the initial)"
        )
        return

    def threshold_3g(self):
        """
        Remove from the view the cells below and above a low and high threshold
        """
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Store the current view for rapid switch between thresholded and not
        if not hasattr(points.features, "current_view"):
            points.features["current_view"] = points.shown.copy()

        # Compute and apply the threshold

        mask = (
            points.features["current_view"]
            & (0 < points.features["3genes"])
        )
        points.shown = mask
        points.refresh()
        nb_selected = np.sum(mask)
        overall = np.sum(points.features["current_view"])
        self.threshold_output.value = (
            f"{nb_selected} cells "
            f"({100*nb_selected/overall:.1f}% of the initial)"
        )
        return

    def adj_int(self):
        """
        Adjust the intensity for gene expression colouring
        """
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return
        if points.face_color_mode.upper() != "COLORMAP":
            return
        min = self.adj_int_low.value
        max = self.adj_int_high.value
        if max < min:
            max, min = min, max
        points.face_contrast_limits = (min, max)
        points.refresh()

    def apply_cmap(self):
        """
        Apply a color map to cells
        """
        # Pretty straight forward (?)
        points = self.viewer.layers.selection.active
        if (
                points is None
                or points.as_layer_data_tuple()[-1] != "points"
                or len(points.properties) == 0
        ):
            error_points_selection(show=self.show)
            return
        if points.face_color_mode.lower() != "colormap":
            points.face_color = "gene"
            points.face_color_mode = "colormap"
        if not self.cmap_check.value:
            points.face_colormap = self.cmap.value
            points.mplcmap = None
        else:
            init_value = self.grey.value
            cmap_mpl = {
                "red": [[0.0, init_value, init_value], [1.0, 0.0, 0.0]],
                "blue": [[0.0, init_value, init_value], [1.0, 0.0, 0.0]],
                "green": [[0.0, init_value, init_value], [1.0, 0.0, 0.0]],
            }
            cmap_mpl[self.manual_color.value.lower()] = [
                [0.0, init_value, init_value],
                [1.0, 1.0, 1.0],
            ]
            if self.manual_color.value == "Red":
                color = 0
            elif self.manual_color.value == "Green":
                color = 1
            else:
                color = 2
            cmap_val = [
                [init_value, init_value, init_value, 1],
                [0, 0, 0, 1],
            ]
            cmap_val[1][color] = 1
            cmap = Colormap(cmap_val)
            mplcmap = colors.LinearSegmentedColormap("Manual cmap", cmap_mpl)
            points.mplcmap = mplcmap
            points.face_colormap = cmap
        points.refresh()






    def show_two_genes(self):
        """
        Function that show two genes
        """
        # Get the layer with the points, makes sure it exists and is one
        # Point layer indeed
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Get the list of cells (it is initially a set)
        # and all the parameters and makes sure that they make sense
        # cell_list = list(self.embryo.all_cells)
        gene1 = self.gene1.value
        gene2 = self.gene2.value
        low_th = self.threhold_low_2g.value
        high_th = self.threhold_high_2g.value
        main_bi_color = self.main_bi_color.value

        if not gene1 in self.embryo.anndata.raw.var_names:
            self.metric_2g_output.value = f"'{gene1}' not found"
            return
        if not gene2 in self.embryo.anndata.raw.var_names:
            self.metric_2g_output.value = f"'{gene2}' not found"
            return

        # Makes sure not to reprocess already processed data and process them
        # if necessary
        if (
            not points.metadata["2genes"]
            or (gene1, gene2, low_th, high_th, main_bi_color)
            != points.metadata["2genes"]
        ):
            if "current_view" in points.features:
                mask = points.features["current_view"]
            else:
                mask = points.shown

            # Gets the values for the 1st and 2nd genes as ndarrays
            colors1 = safe_toarray(self.embryo.anndata.raw[:, gene1].X)[:, 0]
            colors2 = safe_toarray(self.embryo.anndata.raw[:, gene2].X)[:, 0]
            C = np.array([colors1, colors2])

            # Get the threshold value for the gene activities
            min_g1 = np.percentile(C[0][mask], low_th)
            min_g2 = np.percentile(C[1][mask], low_th)
            max_g1 = np.percentile(C[0][mask], high_th)
            max_g2 = np.percentile(C[1][mask], high_th)

            # Normalize and threshold the genes from 0 to 1
            norm = lambda C: (C - [[min_g1], [min_g2]]) / [
                [max_g1 - min_g1],
                [max_g2 - min_g2],
            ]
            V = norm(C)
            V[V < 0] = 0
            V[1 < V] = 1
            print(V)
            # Build the RGB array
            final_C = np.zeros((len(colors1), 3))
            on_channel = (
                np.array(["Red", "Green", "Blue"]) != main_bi_color
            ).astype(int)
            final_C[:, 0] = V[on_channel[0]]
            final_C[:, 1] = V[on_channel[1]]
            final_C[:, 2] = V[on_channel[2]]

            # Assign the color to the cells
            points.face_color = final_C
            points.face_color_mode = "direct"
            points.features["2genes"] = final_C.sum(axis=1)
            print(points.features["2genes"])
            points.metadata["2genes"] = (
                gene1,
                gene2,
                low_th,
                high_th,
                main_bi_color,
            )
            points.metadata["2genes_params"] = (
                max_g1,
                max_g2,
                norm,
                on_channel,
            )
            points.metadata["gene"] = None
            points.edge_color = "black"
        self.metric_2g_output.value = "Showing " + ", ".join(
            points.metadata["2genes"][:2]
        )
        return

    def show_three_genes(self):
        """
        Function that shows three genes
        """
        # Get the layer with the points, make sure it exists and is one
        # Point layer indeed
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        # Get the list of cells (it is initially a set)
        # and all the parameters and make sure that they make sense
        gene3 = self.gene3.value
        gene4= self.gene4.value
        gene5 = self.gene5.value
        low_th_3g = self.threhold_low_3g.value
        high_th_3g = self.threhold_high_3g.value
        low_th_4g = self.threhold_low_4g.value
        high_th_4g = self.threhold_high_4g.value
        low_th_5g = self.threhold_low_5g.value
        high_th_5g = self.threhold_high_5g.value
        main_bi_color = self.main_bi_color.value

        genes = [gene3, gene4, gene5]

        for gene in genes:
            if not gene in self.embryo.anndata.raw.var_names:
                self.metric_3g_output.value = f"'{gene}' not found"
                return

        # Makes sure not to reprocess already processed data and process them
        # if necessary
        if (
                not points.metadata["3genes"]
                or (genes, low_th_3g, high_th_3g,low_th_4g, high_th_4g, low_th_5g, high_th_5g, main_bi_color)
                != points.metadata["3genes"]
        ):
            if "current_view" in points.features:
                mask = points.features["current_view"]
            else:
                mask = points.shown

            colors1 = safe_toarray(self.embryo.anndata.raw[:, gene3].X)[:, 0]
            colors2 = safe_toarray(self.embryo.anndata.raw[:, gene4].X)[:, 0]
            colors3 = safe_toarray(self.embryo.anndata.raw[:, gene5].X)[:, 0]
            C = np.array([colors1, colors2,colors3])

            # Get the threshold value for the gene activities
            min_g1 = np.percentile(C[0][mask], low_th_3g)
            min_g2 = np.percentile(C[1][mask], low_th_4g)
            max_g1 = np.percentile(C[0][mask], high_th_3g)
            max_g2 = np.percentile(C[1][mask], high_th_4g)
            min_g3 = np.percentile(C[2][mask], low_th_5g)
            max_g3 = np.percentile(C[2][mask], high_th_5g)

            # Normalize and threshold the genes from 0 to 1
            norm = lambda C: (C - [[min_g1], [min_g2],[min_g3]]) / [
                [max_g1 - min_g1],
                [max_g2 - min_g2],
                [max_g3 - min_g3],
            ]
            V = norm(C)
            V[V < 0] = 0
            V[1 < V] = 1

            # Build the RGB array
            final_C = np.zeros((len(colors1), 3))
            on_channel = (
                    np.array(["Red", "Green", "Blue"]) != main_bi_color
            ).astype(int)
            on_channel[2] = 2
            final_C[:, 0] = V[on_channel[0]]
            final_C[:, 1] = V[on_channel[1]]
            final_C[:, 2] = V[on_channel[2]]

            # Assign the color to the cells
            points.face_color = final_C
            points.face_color_mode = "direct"
            points.features["3genes"] = final_C.sum(axis=1)
            points.metadata["3genes"] = (
                gene3,
                gene4,
                gene5,
                low_th_3g,
                low_th_4g,
                low_th_5g,
                high_th_3g,
                high_th_4g,
                high_th_5g,
                main_bi_color,
            )
            points.metadata["3genes_params"] = (
                max_g1,
                max_g2,
                max_g3,
                norm,
                on_channel,
            )
            points.metadata["gene"] = None
            points.edge_color = "black"
        self.metric_3g_output.value = "Showing " + ", ".join(
            points.metadata["3genes"][:3]
        )
        return

    def build_tissue_selection(self):
        """
        Function that builds the qt container for the selection of the tissues
        """
        # Selecting tissues
        self.select_tissues_choices = widgets.Select(
            choices=self.all_tissues,
            value=[
                self.embryo.corres_tissue.get(t, f"{t}")
                for t in self.tissues_to_plot
            ],
        )
        run_select = widgets.FunctionGui(
            self.select_tissues, call_button="Select Tissues"
        )
        run_tissues = widgets.FunctionGui(
            self.show_tissues, call_button="Cell type colouring"
        )

        recolor_tissues = widgets.FunctionGui(
            self.recolor_tissues, call_button="Recolour tissues"
        )
        self.choose_how2cal = widgets.RadioButtons(
            choices=[
                "High distance to center of mass",
                "High distance to neighbor",
            ],
            value="High distance to center of mass",
        )
        # Coloring by tissues
        run_legend = widgets.FunctionGui(
            self.disp_legend, call_button="Display legend"
        )

        select_container = widgets.Container(
            widgets=[self.select_tissues_choices, run_select], labels=False
        )
        display_container = widgets.Container(
            widgets=[run_tissues, run_legend],
            layout="horizontal",
            labels=False,
        )

        min_edges_label = widgets.Label(
            value="删除n个独立细胞"
        )
        self.min_edges = widgets.IntText(value=0,max = 20000)
        display_flatten = widgets.FunctionGui(
            self.add_flattend, call_button="show flatten"
        )
        display_container.native.layout().addStretch(1)
        tissue_container = widgets.Container(
            widgets=[min_edges_label,self.min_edges, self.choose_how2cal,select_container,recolor_tissues, display_container,display_flatten], labels=False
        )
        tissue_container.native.layout().addStretch(1)
        return tissue_container

    def build_layer_container(self):
        """
        Function that builds the qt container for the selection of the tissues
        """
        # Selecting tissues
        self.select_layers_choices = widgets.Select(
            choices=self.all_layers,
            value=[
                        t
                        for t in self.all_layers
                    ],
        )
        run_select = widgets.FunctionGui(
            self.select_layers, call_button="Select Layers"
        )
        run_tissues = widgets.FunctionGui(
            self.show_tissues, call_button="Cell type colouring"
        )

        recolor_tissues = widgets.FunctionGui(
            self.recolor_tissues, call_button="Recolour tissues"
        )

        # Coloring by tissues
        run_legend = widgets.FunctionGui(
            self.disp_legend, call_button="Display legend"
        )

        select_container = widgets.Container(
            widgets=[self.select_layers_choices, run_select], labels=False
        )
        display_container = widgets.Container(
            widgets=[run_tissues, run_legend],
            layout="horizontal",
            labels=False,
        )
        display_container.native.layout().addStretch(1)
        tissue_container = widgets.Container(
            widgets=[select_container, recolor_tissues, display_container], labels=False
        )
        tissue_container.native.layout().addStretch(1)
        return tissue_container

    def build_germ_layer_container(self):
        """
        Function that builds the qt container for the selection of the tissues
        """
        # Selecting tissues
        self.select_germ_layers_choices = widgets.Select(
            choices=self.all_germ_layers,
            value=[
                        t
                        for t in self.all_germ_layers
                    ],
        )
        run_select = widgets.FunctionGui(
            self.select_Germ_layers, call_button="Select Germ Layers"
        )
        run_tissues = widgets.FunctionGui(
            self.show_tissues, call_button="Cell type colouring"
        )

        recolor_tissues = widgets.FunctionGui(
            self.recolor_tissues, call_button="Recolour tissues"
        )

        # Coloring by tissues
        run_legend = widgets.FunctionGui(
            self.disp_legend, call_button="Display legend"
        )

        select_container = widgets.Container(
            widgets=[self.select_germ_layers_choices, run_select], labels=False
        )
        display_container = widgets.Container(
            widgets=[run_tissues, run_legend],
            layout="horizontal",
            labels=False,
        )
        display_container.native.layout().addStretch(1)
        tissue_container = widgets.Container(
            widgets=[select_container, recolor_tissues, display_container], labels=False
        )
        tissue_container.native.layout().addStretch(1)
        return tissue_container

    def selectXY_container(self):

        self.min_X = widgets.FloatSpinBox(value=min(self.embryo.anndata.obs['x']), step=0.00000001, max=max(self.embryo.anndata.obs['x'])+0.0000001, min=min(self.embryo.anndata.obs['x'])-0.0000001)
        self.max_X = widgets.FloatSpinBox(value=max(self.embryo.anndata.obs['x']), step=0.00000001, max=max(self.embryo.anndata.obs['x'])+0.0000001, min=min(self.embryo.anndata.obs['x'])-0.0000001)
        self.min_Y = widgets.FloatSpinBox(value=min(self.embryo.anndata.obs['y']), step=0.00000001, max=max(self.embryo.anndata.obs['y'])+0.0000001, min=min(self.embryo.anndata.obs['y'])-0.0000001)
        self.max_Y = widgets.FloatSpinBox(value=max(self.embryo.anndata.obs['y']), step=0.00000001, max=max(self.embryo.anndata.obs['y'])+0.0000001, min=min(self.embryo.anndata.obs['y'])-0.0000001)

        min_X_label = widgets.Label(
            value="min_X"
        )
        max_X_label = widgets.Label(
            value="max_X"
        )
        min_Y_label = widgets.Label(
            value="min_Y"
        )
        max_Y_label = widgets.Label(
            value="max_Y"
        )



        diff_expr_container = widgets.Container(
            widgets=[
                min_X_label,
                self.min_X,
                max_X_label,
                self.max_X,
                min_Y_label,
                self.min_Y,
                max_Y_label,
                self.max_Y,
            ],
            labels=False,
        )
        diff_expr_container.native.layout().addStretch(1)
        return diff_expr_container

    def build_surf_container(self):
        """
        Function that builds the qt container to build the surfaces
        """

        # Check whether pyvista is installed
        if pyvista:
            # Tissue choice
            surf_label = widgets.Label(value="Choose tissue")
            self.select_surf = widgets.ComboBox(
                choices=self.all_tissues, value=self.all_tissues[0]
            )
            select_surf_label = widgets.Container(
                widgets=[surf_label, self.select_surf], labels=False
            )

            # Choice for the pruning method and its parameter
            self.surf_method = widgets.RadioButtons(
                choices=[
                    "High distance to center of mass",
                    "High distance to neighbor",
                ],
                value="High distance to center of mass",
            )
            surf_threshold_label = widgets.Label(
                value="Choose the percent of points to remove"
            )
            self.surf_threshold = widgets.FloatSlider(min=0, max=100, value=0)
            surf_run = widgets.FunctionGui(
                self.show_surf, call_button="Compute and show surface"
            )

            # Building the container
            surf_container = widgets.Container(
                widgets=[
                    select_surf_label,
                    self.surf_method,
                    surf_threshold_label,
                    self.surf_threshold,
                    surf_run,
                ],
                labels=False,
            )
            surf_container.native.layout().addStretch(1)
        else:
            surf_container = widgets.Label(
                value=(
                    "\tPlease install pyvista to compute tissue surfaces\n"
                    "\tYou can run:\n"
                    "\t\t- `pip install pyvista` or\n\t\t- `conda install pyvista`\n"
                    "\tto install it."
                )
            )
        return surf_container

    def compute_feature(self):

        adata = self.embryo.anndata
        if 'z' in adata.obs:
            z_values = adata.obs['z'].values
            combined_data = np.column_stack((adata.obsm['X_spatial'], z_values))
            adata.obsm['X_spatial'] = combined_data


        geneslist, num2 = sc.pp.filter_genes(adata, min_cells=self.min_cells.value, inplace=False)

        selected_adata = adata[:, geneslist]

        def process_data(adata):
            vectorlist = []

            for gene_name in tqdm(adata.var_names, desc="Processing genes"):
                    gene_expression = adata[:, gene_name].X

                    if issparse(gene_expression):
                        gene_expression = gene_expression.toarray().flatten()

                    non_zero_indices = np.where(gene_expression > 0)[0]

                    if len(non_zero_indices) > 0:
                        templist = adata.obsm['X_spatial'][non_zero_indices]
                        tempmean = np.mean(templist, axis=0)
                        vectorlist.append(tempmean)

            return vectorlist




        vectorlist = process_data(selected_adata)


        result_df = pd.DataFrame(vectorlist, index=selected_adata.var_names)


        self.result_df = result_df

    def show_similar_genelist(self):
        df = pd.DataFrame(self.result_df)

        selected_gene = self.gene.value
        selected_vector = df.loc[selected_gene]

        # 创建一个列表存储基因名、向量和差异值的元组
        gene_diff_list = []
        def cosine_similarity(vector_a, vector_b):
            dot_product = np.dot(vector_a, vector_b)
            norm_a = np.linalg.norm(vector_a)
            norm_b = np.linalg.norm(vector_b)
            similarity = dot_product / (norm_a * norm_b)
            return similarity
        for gene, vector in zip(df.index, df.values):
            #     diff = np.linalg.norm(selected_vector - vector)
            diff = cosine_similarity(selected_vector, vector)
            #     diff = vector_similarity(selected_vector, vector)
            gene_diff_list.append((gene, vector, diff))

        # 按差异值从小到大进行排序
        sorted_gene_diff_list = sorted(gene_diff_list, key=lambda x: x[2], reverse=True)

        gene_list = [gene for gene, _, _ in sorted_gene_diff_list]

        sq.gr.spatial_neighbors(current_adata, spatial_key='X_spatial', coord_type='generic')
        Moranres = sq.gr.spatial_autocorr(current_adata, mode="moran", copy=True)

        # 过滤基因，只保留 Moran's I 大于 0.05 的基因
        filtered_genelist = Moranres[Moranres['I'] > 0.05]

        # 进一步限制 genelist 只包含在 filtered_genelist 中的基因
        genelist = list(filter(lambda x: x in filtered_genelist.index.tolist(), gene_list))

        self.similar_gene.choices = []

        self.similar_gene.choices = genelist
        
    def show_similar_gene(self):
        index = self.similar_gene.value
        self.gene.value = index
        self.show_gene()
        
    def build_metric_1g_container(self):
        """
        Function that builds the qt container to display gene expression
        """

        # Choice of the metric to display
        metric_label = widgets.Label(value="What to display:")
        self.metric = widgets.ComboBox(
            choices=(
                ["Gene"]
                + [
                    c
                    for c in list(self.embryo.anndata.obs.columns)
                    if self.embryo.anndata.obs[c].dtype in [float, int]
                ]
            ),
            value="Gene",
        )
        metric_container = widgets.Container(
            widgets=[metric_label, self.metric],
            layout="horizontal",
            labels=False,
        )

        # Choice of the gene to display
        gene_label = widgets.Label(value="Which gene (if necessary)")
        self.gene = widgets.LineEdit(value="T")
        gene_container = widgets.Container(
            widgets=[gene_label, self.gene], layout="horizontal", labels=False
        )
        metric_1g_run = widgets.FunctionGui(
            self.show_gene, call_button="Show gene/metric"
        )
        self.gene_output = widgets.Label(value="")

        # Choice of the low and high threshold
        self.threshold_low = widgets.FloatSlider(min=0, max=1, value=0)
        self.threshold_high = widgets.FloatSlider(min=0, max=1, value=1)
        threshold_run = widgets.FunctionGui(
            self.threshold, call_button="Apply threshold"
        )
        self.threshold_output = widgets.Label(value="")
        threshold = widgets.Container(
            widgets=[
                self.threshold_low,
                self.threshold_high,
                threshold_run,
                self.threshold_output,
            ],
            labels=False,
        )
        threshold.native.layout().addStretch(1)

        # Choice for the intensity thresholds
        self.adj_int_low = widgets.FloatSlider(min=0, max=1, value=0)
        self.adj_int_high = widgets.FloatSlider(min=0, max=1, value=1)
        adj_int_run = widgets.FunctionGui(
            self.adj_int, call_button="Adjust contrast"
        )
        adj_int = widgets.Container(
            widgets=[self.adj_int_low, self.adj_int_high, adj_int_run],
            labels=False,
        )
        adj_int.native.layout().addStretch(1)

        # Choice for the color map
        self.cmap = widgets.ComboBox(choices=ALL_COLORMAPS.keys())
        self.cmap.changed.connect(self.apply_cmap)
        text_manual = widgets.Label(value="Manual:")
        self.cmap_check = widgets.CheckBox(value=False)
        grey_text = widgets.Label(value="Start Grey:")
        self.grey = widgets.FloatSpinBox(value=0.2, min=0, max=1, step=0.01)
        color_text = widgets.Label(value="Main color")
        self.manual_color = widgets.ComboBox(choices=["Red", "Green", "Blue"])
        cmap_check = widgets.Container(
            widgets=[text_manual, self.cmap_check, grey_text, self.grey],
            layout="horizontal",
            labels=False,
        )
        manual_color = widgets.Container(
            widgets=[color_text, self.manual_color],
            layout="horizontal",
            labels=False,
        )
        cmap_man_run = widgets.FunctionGui(
            self.apply_cmap, call_button="Apply color map"
        )
        cmap = widgets.Container(
            widgets=[self.cmap, cmap_check, manual_color, cmap_man_run],
            labels=False,
        )
        cmap.native.layout().addStretch(1)
        cmap.native.layout().setSpacing(0)
        cmap.native.layout().setContentsMargins(1, 1, 1, 1)

        # Building the container
        tab3 = QTabWidget()
        tab3.addTab(threshold.native, "Cell Threshold")
        tab3.addTab(adj_int.native, "Contrast")
        tab3.addTab(cmap.native, "Colormap")
        tab3.native = tab3
        tab3.name = ""

        compute_feature_run = widgets.FunctionGui(
            self.compute_feature, call_button="compute_feature"
        )

        self.gene_feature_output = widgets.Label(value="")
        show_similar_genelist_run = widgets.FunctionGui(
            self.show_similar_genelist, call_button="similar genes"
        )
        similar_gene = []
        self.similar_gene = widgets.ComboBox(choices=similar_gene)
        self.similar_gene.changed.connect(self.show_similar_gene)

        self.gene_output = widgets.Label(value="")
        metric_1g_container = widgets.Container(
            widgets=[
                metric_container,
                gene_container,
                metric_1g_run,
                self.gene_output,
                tab3,
                compute_feature_run,
                self.gene_feature_output,
                show_similar_genelist_run,
                self.similar_gene,

            ],
            labels=False,
        )
        metric_1g_container.native.layout().addStretch(1)
        return metric_1g_container

    def build_metric_2g_container(self):
        """
        Function that builds the qt container to display gene co-expression
        """
        # Choice of the first gene
        self.gene1 = widgets.LineEdit(value="T")
        gene1_label = widgets.Label(value="First gene (main)")
        gene1_container = widgets.Container(
            widgets=[gene1_label, self.gene1],
            layout="horizontal",
            labels=False,
        )

        # Choice of the second gene
        self.gene2 = widgets.LineEdit(value="Sox2")
        gene2_label = widgets.Label(value="Second gene")
        gene2_container = widgets.Container(
            widgets=[gene2_label, self.gene2],
            layout="horizontal",
            labels=False,
        )

        # Choice of the value for the low threshold
        self.threhold_low_2g = widgets.Slider(value=0, min=0, max=100)
        threhold_low_2g_label = widgets.Label(value="Low threshold")
        threhold_low_2g_container = widgets.Container(
            widgets=[threhold_low_2g_label, self.threhold_low_2g],
            layout="horizontal",
            labels=False,
        )

        # Choice for the high threshold
        self.threhold_high_2g = widgets.Slider(
            value=100,
            min=0,
            max=100,
            label="High threshold",
            name="High threshold",
        )
        threhold_high_2g_label = widgets.Label(value="High threshold")
        threhold_high_2g_container = widgets.Container(
            widgets=[threhold_high_2g_label, self.threhold_high_2g],
            layout="horizontal",
            labels=False,
        )

        # Choice of the main color
        self.main_bi_color = widgets.ComboBox(
            choices=["Red", "Green", "Blue"], value="Red"
        )
        main_bi_color_label = widgets.Label(value="Main color")
        main_bi_color_container = widgets.Container(
            widgets=[main_bi_color_label, self.main_bi_color],
            layout="horizontal",
            labels=False,
        )

        # Run button
        metric_2g_run = widgets.FunctionGui(
            self.show_two_genes, call_button="Map Colors", labels=False
        )
        self.metric_2g_output = widgets.Label(value="")

        threshold_run = widgets.FunctionGui(
            self.threshold_2g, call_button="Filter"
        )
        # Build the container
        metric_2g_container = widgets.Container(
            widgets=[
                gene1_container,
                gene2_container,
                threhold_low_2g_container,
                threhold_high_2g_container,
                main_bi_color_container,
                metric_2g_run,
                self.metric_2g_output,
                threshold_run
            ],
            labels=False,
        )
        metric_2g_container.native.layout().addStretch(1)
        return metric_2g_container

    def build_metric_3g_container(self):
        """
        Function that builds the qt container to display gene co-expression for three genes
        """
        # Choice of the first gene
        self.gene3 = widgets.LineEdit(value="T")
        gene3_label = widgets.Label(value="First gene (main)")
        gene3_container = widgets.Container(
            widgets=[gene3_label, self.gene3],
            layout="horizontal",
            labels=False,
        )

        # Choice of the second gene
        self.gene4 = widgets.LineEdit(value="Sox2")
        gene4_label = widgets.Label(value="Second gene")
        gene4_container = widgets.Container(
            widgets=[gene4_label, self.gene4],
            layout="horizontal",
            labels=False,
        )

        # Choice of the third gene
        self.gene5 = widgets.LineEdit(value="Cdx1")
        gene5_label = widgets.Label(value="Third gene")
        gene5_container = widgets.Container(
            widgets=[gene5_label, self.gene5],
            layout="horizontal",
            labels=False,
        )
        self.threhold_low_3g = widgets.Slider(value=0, min=0, max=100)
        threhold_low_3g_label = widgets.Label(value="Low threshold (1st gene)")
        threhold_low_3g_container = widgets.Container(
            widgets=[threhold_low_3g_label, self.threhold_low_3g],
            layout="horizontal",
            labels=False,
        )

        # Choice for the high threshold for 2nd gene
        self.threhold_high_3g = widgets.Slider(
            value=100,
            min=0,
            max=100,
            label="High threshold (2nd gene)",
            name="High threshold (2nd gene)",
        )
        threhold_high_3g_label = widgets.Label(value="High threshold (1st gene)")
        threhold_high_3g_container = widgets.Container(
            widgets=[threhold_high_3g_label, self.threhold_high_3g],
            layout="horizontal",
            labels=False,
        )
        # Choice of the value for the low threshold for 2nd gene
        self.threhold_low_4g = widgets.Slider(value=0, min=0, max=100)
        threhold_low_4g_label = widgets.Label(value="Low threshold (2nd gene)")
        threhold_low_4g_container = widgets.Container(
            widgets=[threhold_low_4g_label, self.threhold_low_4g],
            layout="horizontal",
            labels=False,
        )

        # Choice for the high threshold for 2nd gene
        self.threhold_high_4g = widgets.Slider(
            value=100,
            min=0,
            max=100,
            label="High threshold (2nd gene)",
            name="High threshold (2nd gene)",
        )
        threhold_high_4g_label = widgets.Label(value="High threshold (2nd gene)")
        threhold_high_4g_container = widgets.Container(
            widgets=[threhold_high_4g_label, self.threhold_high_4g],
            layout="horizontal",
            labels=False,
        )

        # Choice of the value for the low threshold for 3rd gene
        self.threhold_low_5g = widgets.Slider(value=0, min=0, max=100)
        threhold_low_5g_label = widgets.Label(value="Low threshold (3rd gene)")
        threhold_low_5g_container = widgets.Container(
            widgets=[threhold_low_5g_label, self.threhold_low_5g],
            layout="horizontal",
            labels=False,
        )

        # Choice for the high threshold for 3rd gene
        self.threhold_high_5g = widgets.Slider(
            value=100,
            min=0,
            max=100,
            label="High threshold (3rd gene)",
            name="High threshold (3rd gene)",
        )
        threhold_high_5g_label = widgets.Label(value="High threshold (3rd gene)")
        threhold_high_5g_container = widgets.Container(
            widgets=[threhold_high_5g_label, self.threhold_high_5g],
            layout="horizontal",
            labels=False,
        )
        threshold_run = widgets.FunctionGui(
            self.threshold_3g, call_button="Filter"
        )
        # Choice of the main color
        self.main_bi_color = widgets.ComboBox(
            choices=["Red", "Green", "Blue"], value="Red"
        )
        main_bi_color_label = widgets.Label(value="Main color")
        main_bi_color_container = widgets.Container(
            widgets=[main_bi_color_label, self.main_bi_color],
            layout="horizontal",
            labels=False,
        )

        # Run button
        metric_3g_run = widgets.FunctionGui(
            self.show_three_genes, call_button="Map Colors", labels=False
        )
        self.metric_3g_output = widgets.Label(value="")

        # Build the container for three genes
        metric_3g_container = widgets.Container(
            widgets=[
                gene3_container,
                gene4_container,
                gene5_container,
                threhold_low_3g_container,
                threhold_high_3g_container,
                threhold_low_4g_container,
                threhold_high_4g_container,
                threhold_low_5g_container,
                threhold_high_5g_container,
                main_bi_color_container,
                metric_3g_run,
                self.metric_3g_output,
                threshold_run
            ],
            labels=False,
            scrollable = True,
        )
        metric_3g_container.native.layout().addStretch(1)
        return metric_3g_container

    def build_umap_container(self):
        """
        Function that builds the qt container for the umap
        """
        # Gene choice
        gene_label = widgets.Label(value="Choose gene")
        gene = widgets.LineEdit(value="T")

        # Whether to display the clusters
        tissues_label = widgets.Label(value="Display tissues umap")
        tissues = widgets.CheckBox(value=False)

        # Whether taking variable genes or not
        variable_genes_label = widgets.Label(value="Take only variable genes")
        variable_genes = widgets.CheckBox(value=True)

        # Which stats to display variable genes
        stats_label = widgets.Label(value="Stat for\nchoosing distributions")
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
            self.color_map_tissues,
            self.tab2,
        )
        umap_run = widgets.FunctionGui(
            self.umap_selec.run, call_button="Show gene on Umap", name=""
        )

        # Builds the containers
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
        umap_container.native.layout().addStretch(1)
        return umap_container

    def display_Moran_expressed(self):

        # points = self.viewer.layers.selection.active
        # if points is None or points.as_layer_data_tuple()[-1] != "points":
        #     error_points_selection(show=self.show)
        #     return
        # selected_layers = self.select_layers_choices.value
        # selected_tissues = self.select_tissues_choices.value
        # selected_germ_layers = self.select_germ_layers_choices.value
        #
        # tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}
        #
        # tissues_to_plot = []
        # for t in selected_tissues:
        #     if t in tissue_to_num:
        #         tissues_to_plot.append(tissue_to_num[t])
        #     else:
        #         tissues_to_plot.append(int(t))
        # for t in selected_layers:
        #     tissues_to_plot.append(t)
        # for t in selected_germ_layers:
        #     tissues_to_plot.append(t)
        # # tissus_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}
        # shown1 = [
        #     self.embryo.anndata.obs['germ_layer'][c] in tissues_to_plot
        #     for c in points.properties["cells"]
        # ]
        # shown2 = [
        #     self.embryo.anndata.obs['orig.ident'][c] in tissues_to_plot
        #     for c in points.properties["cells"]
        # ]
        # shown3 = [
        #     self.embryo.tissue[c] in tissues_to_plot
        #     for c in points.properties["cells"]
        # ]
        #
        #
        #
        #
        #
        # shown = [a and b and c for a, b, c in zip(shown1, shown2, shown3)]
        #
        # adata = self.embryo.anndata
        # print("adata_cellnum = " + str(adata.shape[0]))
        # print("adata_genenum = " + str(adata.shape[1]))
        #
        #
        # selected_adata = adata[shown]
        # geneslist1, num1 = sc.pp.filter_genes(selected_adata, min_counts=self.min_counts.value,inplace=False)
        # geneslist2, num2 = sc.pp.filter_genes(selected_adata, min_cells=self.min_cells.value, inplace=False)
        #
        # geneslist = [a and b  for a, b in zip(geneslist1, geneslist2)]
        # selected_adata = selected_adata[:, geneslist]
        #
        # print("selected_adata_cellnum = " + str(selected_adata.shape[0]))
        # print("selected_adata_genenum = " + str(selected_adata.shape[1]))
        #
        #
        #
        # if 'z'  in selected_adata.obs:
        #     z_values = selected_adata.obs['z'].values
        #     #    将 'adata.obsm['X_spatial']'  'z_values' 沿第二个轴（列）合并
        #     combined_data = np.column_stack((selected_adata.obsm['X_spatial'], z_values*self.embryo.z_resolusion))
        # # 创建新的 AnnData 对象，将 'combined_data' 分配给 'adata.obsm['X_spatial']
        #     selected_adata.obsm['X_spatial'] = combined_data
        # index_list = [index for index, value in enumerate(shown) if value]
        # coordinates = selected_adata.obsm['X_spatial']
        #
        # print(coordinates)
        #
        #
        # if self.choose_how2cal.value != 0:
        #     if self.choose_how2cal.value == "High distance to center of mass":
        #         center_of_mass = np.mean(coordinates, axis=0)
        #         mean_distances = np.linalg.norm(coordinates - center_of_mass, axis=1)
        #         mask = np.array([True] * len(coordinates))
        #         far_away_cells = np.argsort(mean_distances)[::-1][:self.min_edges.value]
        #             # 将度数最小的m个细胞置为False
        #         mask[far_away_cells] = False
        #
        #     else:
        #         node_ids = list(range(len(coordinates)))
        #
        #         # 创建布尔列表，初始化为True
        #         mask = np.array([True] * len(coordinates))
        #
        #         while self.min_edges.value > 0:
        #             gg = self.embryo.build_gabriel_graph(
        #                 node_ids, coordinates, data_struct="adj-mat", dist=True
        #             )
        #             mean_distances = gg.mean(axis=0)
        #
        #             # 找到剩余细胞中mean_distances最大的细胞
        #             max_distance_cell = np.argmax(mean_distances * mask)
        #
        #             # 如果没有更多的细胞可以剪枝了，退出循环
        #             if mean_distances[max_distance_cell] == 0:
        #                 break
        #
        #             # 将当前最大的细胞置为False
        #             mask[max_distance_cell] = False
        #             self.min_edges.value -= 1
        #
        #             # 重新构建坐标
        #             coordinates = np.delete(coordinates, max_distance_cell, axis=0)
        #             node_ids = np.delete(node_ids, max_distance_cell)
        #
        # # filterbyedgeslist = find_and_mask_low_degree_cells(selected_adata,self.min_edges.value)
        # for i in range(len(index_list)):
        #     shown[index_list[i]] = mask[i]
        points = self.viewer.layers.selection.active
        if points is None or points.as_layer_data_tuple()[-1] != "points":
            error_points_selection(show=self.show)
            return

        selected_layers = self.select_layers_choices.value
        selected_tissues = self.select_tissues_choices.value
        selected_germ_layers = self.select_germ_layers_choices.value



        tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}

        tissues_to_plot = []
        for t in selected_tissues:
            if t in tissue_to_num:
                tissues_to_plot.append(tissue_to_num[t])
            else:
                tissues_to_plot.append(int(t))
        for t in selected_layers:
            tissues_to_plot.append(t)
        for t in selected_germ_layers:
            tissues_to_plot.append(t)
        # tissus_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}

        shown1 = [
            self.embryo.anndata.obs['germ_layer'][c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        shown2 = [
            self.embryo.anndata.obs['orig.ident'][c] in tissues_to_plot
            for c in points.properties["cells"]
        ]
        shown3 = [
            self.embryo.tissue[c] in tissues_to_plot
            for c in points.properties["cells"]
        ]

        #shown =  三个选项卡的交集
        shown = [a and b and c for a, b, c in zip(shown1, shown2, shown3)]

        adata = self.embryo.anndata
        print("adata_cellnum = " + str(adata.shape[0]))
        print("adata_genenum = " + str(adata.shape[1]))



        index_list = [index for index, value in enumerate(shown) if value]

        if 'z'  in adata.obs:
            z_values = adata.obs['z'].values
            #    将 'adata.obsm['X_spatial']'  'z_values' 沿第二个轴（列）合并
            combined_data = np.column_stack((adata.obsm['X_spatial'], z_values*self.embryo.z_resolusion))
        # 创建新的 AnnData 对象，将 'combined_data' 分配给 'adata.obsm['X_spatial']
            adata.obsm['X_spatial'] = combined_data
        coordinates = adata.obsm['X_spatial']

        if self.choose_how2cal.value != 0:
            if self.choose_how2cal.value == "High distance to center of mass":
                center_of_mass = np.mean(coordinates, axis=0)
                mean_distances = np.linalg.norm(coordinates - center_of_mass, axis=1)
                mask = np.array([True] * len(coordinates))
                far_away_cells = np.argsort(mean_distances)[::-1][:self.min_edges.value]
                    # 将度数最小的m个细胞置为False
                mask[far_away_cells] = False

            else:
                node_ids = list(range(len(coordinates)))

                # 创建布尔列表，初始化为True
                mask = np.array([True] * len(coordinates))

                while self.min_edges.value > 0:
                    gg = self.embryo.build_gabriel_graph(
                        node_ids, coordinates, data_struct="adj-mat", dist=True
                    )
                    mean_distances = gg.mean(axis=0)

                    # 找到剩余细胞中mean_distances最大的细胞
                    max_distance_cell = np.argmax(mean_distances * mask)

                    # 如果没有更多的细胞可以剪枝了，退出循环
                    if mean_distances[max_distance_cell] == 0:
                        break

                    # 将当前最大的细胞置为False
                    mask[max_distance_cell] = False
                    self.min_edges.value -= 1

                    # 重新构建坐标
                    coordinates = np.delete(coordinates, max_distance_cell, axis=0)
                    node_ids = np.delete(node_ids, max_distance_cell)

        # filterbyedgeslist = find_and_mask_low_degree_cells(selected_adata,self.min_edges.value)
        for i in range(len(index_list)):
            shown[index_list[i]] = mask[i]
        print(shown)
        selected_adata = adata[shown]
        print(selected_adata.shape)

        # geneslist1, num1 = sc.pp.filter_genes(selected_adata, min_counts=self.min_counts.value,inplace=False)
        # geneslist2, num2 = sc.pp.filter_genes(selected_adata, min_cells=self.min_cells.value, inplace=False)
        # print(geneslist2)
        # print(geneslist1)
        # geneslist = [a and b  for a, b in zip(geneslist1, geneslist2)]
        # selected_adata = selected_adata[:, geneslist]
        geneslist1, num1 = sc.pp.filter_genes(selected_adata, min_counts=self.min_counts.value,inplace=False)
        geneslist2, num2 = sc.pp.filter_genes(selected_adata, min_cells=self.min_cells.value, inplace=False)
        geneslist3, num3 = sc.pp.filter_genes(selected_adata, max_cells=self.max_cells.value, inplace=False)

        geneslist = [a and b and c for a, b, c in zip(geneslist1, geneslist2, geneslist3)]
        selected_adata = selected_adata[:, geneslist]

        print("selected_adata_cellnum = " + str(selected_adata.shape[0]))
        print("selected_adata_genenum = " + str(selected_adata.shape[1]))


        sq.gr.spatial_neighbors(selected_adata, spatial_key='X_spatial')
        Moranres = sq.gr.spatial_autocorr(selected_adata, mode="moran", copy = True)

        # top_autocorr = (
        #     adata.uns["moranI"]["I"].sort_values(ascending=False).head(num_view).index.tolist()
        # )
        top_autocorr = (
            Moranres["I"].sort_values(ascending=False).index.tolist()
        )
        top_autocorr_list = Moranres["I"].sort_values(ascending=False)

        Moranlist_top = []
        for index, value in top_autocorr_list.items():
            Moranlist_top.append(f"{index}: {value}")


        self.Moran_gene_diff.choices = []

        self.Moran_gene_diff.choices = Moranlist_top
        return selected_adata


    def show_Moran_diff_gene(self):

        colon_index = self.Moran_gene_diff.value.index(':')
        # 提取冒号之前的部分
        index = self.Moran_gene_diff.value[:colon_index].strip()
        self.gene.value = index
        self.show_gene()

    def save_moran_results(self):
        # 获取 Moran's I 计算结果
        selected_adata = self.display_Moran_expressed()

        sq.gr.spatial_neighbors(selected_adata, spatial_key='X_spatial')
        Moranres = sq.gr.spatial_autocorr(selected_adata, mode="moran", copy = True)

        # 获取文件路径
        file_path = self.save_moran_path.value

        # 确保文件夹存在
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if dir_path:
            # 构建保存的文件名，可以自定义
            output_file = f"{dir_path}/moran_results.csv"

            # 将 Moranres 的 'I' 列保存为 CSV 文件
            Moranres.to_csv(output_file, columns=['I'], header=True, index=True)

            print(f"Moran's I values have been saved to {output_file}")
        else:
            print("请先选择保存路径")


    def build_Moran_container(self):
        # tissue_label = widgets.Label(value="Choose tissue:")
        # self.tissue_diff = widgets.ComboBox(choices=self.all_tissues)
        # vol_th_label = widgets.Label(
        #     value="Minimum volume expressed [% of total]"
        # )
        # self.vol_th = widgets.FloatSlider(value=2.5, min=1, max=45, step=0.5)
        button = widgets.FunctionGui(
            self.display_Moran_expressed,
            call_button="Display differentially expressed",
        )
        Moran_gene_diff = []


        diff_gene_label_label = widgets.Label(
            value="Top genes:"
        )

        self.Moran_gene_diff = widgets.ComboBox(choices=Moran_gene_diff)

        self.Moran_gene_diff.changed.connect(self.show_Moran_diff_gene)

        min_cells_label = widgets.Label(
            value="每一个基因至少在n个细胞中表达"
        )
        self.min_cells = widgets.IntText(value=0,max=20000)
        min_counts_label = widgets.Label(
            value="每一个基因的最少表达量"
        )

        self.max_cells = widgets.IntText(value=self.embryo.anndata.shape[0],max=self.embryo.anndata.shape[0])
        max_cells_label = widgets.Label(
            value="每一个基因至多在n个细胞中表达"
        )
        self.min_counts = widgets.FloatSpinBox(value=0,step=0.00000001,max=1000,min = -1000)
        save_moran_path_label = widgets.Label(
            value="Save Path:"
        )
        self.save_moran_path = widgets.LineEdit(value="./")
        # Save button to save Moran results to CSV
        save_button_run = widgets.FunctionGui(
            self.save_moran_results, call_button="save", labels=False
        )
        # Create a container for the UI elements
        diff_expr_container = widgets.Container(
            widgets=[
                min_counts_label,
                self.min_counts,
                min_cells_label,
                self.min_cells,
                max_cells_label,
                self.max_cells,
                button,
                diff_gene_label_label,
                self.Moran_gene_diff,
                save_moran_path_label,
                self.save_moran_path,
                save_button_run
            ],
            labels=False,
        )

        diff_expr_container.native.layout().addStretch(1)
        return diff_expr_container

    def display_diff_expressed(self):
        tissue_to_num = {v: k for k, v in self.embryo.corres_tissue.items()}
        tissue_to_plot = tissue_to_num[self.tissue_diff.value]
        diff_expr = self.embryo.get_3D_differential_expression(
            [tissue_to_plot], self.vol_th.value / 100, all_genes=True
        )[tissue_to_plot]
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots()
            self.embryo.plot_volume_vs_neighbs(
                tissue_to_plot, print_top=10, ax=ax
            )
            fig.show()
        self.gene_diff.choices = diff_expr.sort_values(
            "Localization score", ascending=False
        )[:10]["Gene names"].values

    def show_diff_gene(self):
        self.gene.value = self.gene_diff.value
        self.show_gene()

    def build_diff_expr_container(self):
        tissue_label = widgets.Label(value="Choose tissue:")
        self.tissue_diff = widgets.ComboBox(choices=self.all_tissues)
        vol_th_label = widgets.Label(
            value="Minimum volume expressed [% of total]"
        )
        self.vol_th = widgets.FloatSlider(value=2.5, min=1, max=45, step=0.5)
        button = widgets.FunctionGui(
            self.display_diff_expressed,
            call_button="Display differentially expressed",
        )
        gene_diff = []
        diff_gene_label_label = widgets.Label(
            value="Top 10 differentially expressed genes:"
        )
        self.gene_diff = widgets.ComboBox(choices=gene_diff)
        self.gene_diff.changed.connect(self.show_diff_gene)
        diff_expr_container = widgets.Container(
            widgets=[
                tissue_label,
                self.tissue_diff,
                vol_th_label,
                self.vol_th,
                button,
                diff_gene_label_label,
                self.gene_diff,
            ],
            labels=False,
        )
        diff_expr_container.native.layout().addStretch(1)
        return diff_expr_container

    def __init__(self, viewer, embryo, *, show=False):
        """
        Initialise the plugin.
        Takes as an input a napari viewer and a sc3D embryo

        Args:
            viewer (napari.Viewer): the viewer for the plugin
            embryo (sc3D.Embryo): the embryo to display
            show (bool): an argument to practically run the tests
        """
        from qtpy import QtCore, QtWidgets
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

        
        self.viewer = viewer
        self.embryo = embryo


        # 配置日志记录
        logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

        # 在需要记录的地方添加日志语句
        try:
            if 'germ_layer' not in self.embryo.anndata.obs:
                # iltered_obs.uns['spatial']
                self.embryo.anndata.obs['germ_layer'] = 'N/A'

    

            self.color_map_tissues = {
                # # 1: [0,0,0],
                # 11: [0, 0, 0],
                # # 26: [0, 0, 0],
                # # 5: [0.7411764705882353, 0.803921568627451, 1.0],
                # # 6: [0.19607843137254902, 0.35294117647058826, 0.6078431372549019],
                # # 7: [0.996078431372549, 0.6862745098039216, 0.08627450980392157],
                # 9: [0.7686274509803922, 0.27058823529411763, 0.10980392156862745],
                # 10: [0.10980392156862745, 1.0, 0.807843137254902],
                # 12: [0.7529411764705882, 0.4588235294117647, 0.6509803921568628],
                # 13: [0.9647058823529412, 0.13333333333333333, 0.1803921568627451],
                # 14: [0.7411764705882353, 0.43529411764705883, 0.6705882352941176],
                # 15: [0.9686274509803922, 0.8823529411764706, 0.6274509803921569],
                # 16: [1.0, 0.9803921568627451, 0.9803921568627451],
                # 18: [0.47058823529411764, 0.16470588235294117, 0.7137254901960784],
                # 20: [0.7098039215686275, 0.11372549019607843, 0.5529411764705883],
                # 21: [0.19607843137254902, 0.5137254901960784, 0.996078431372549],
                # 22: [0.5098039215686274, 0.1803921568627451, 0.10980392156862745],
                # 23: [0.5215686274509804, 0.4, 0.050980392156862744],
                # 24: [0.803921568627451, 0.1607843137254902, 0.5647058823529412],
                # 27: [0.6588235294117647, 0.6588235294117647, 0.6588235294117647],
                # 29: [0.0, 0.0, 0.5450980392156862],
                # 30: [0.5450980392156862, 0.2784313725490196, 0.36470588235294116],
                # 31: [1.0, 0.7568627450980392, 0.1450980392156863],
                # 32: [0.8705882352941177, 0.6274509803921569, 0.9921568627450981],
                # 33: [0.9803921568627451, 0.0, 0.5294117647058824],
                # 34: [0.9725490196078431, 0.6313725490196078, 0.6235294117647059],
                # 35: [0.7098039215686275, 0.9372549019607843, 0.7098039215686275],
                # 36: [0.1803921568627451, 0.8509803921568627, 1.0],
                # 39: [0.10980392156862745, 0.5137254901960784, 0.33725490196078434],
                # 40: [1.0, 0.6470588235294118, 0.30980392156862746],
                # 8: [0,0,0],

                0: [0.38823529411764707, 0.3333333333333333, 0.2784313725490196],
                1: [0.8549019607843137, 0.7450980392156863, 0.6],
                2: [0.596078431372549, 0.596078431372549, 0.596078431372549],
                3: [0.9647058823529412, 0.7490196078431373, 0.796078431372549],
                4: [0.4980392156862745, 0.40784313725490196, 0.4549019607843137],
                5: [0.7725490196078432, 0.5803921568627451, 0.7490196078431373],
                6: [0.396078431372549, 0.6588235294117647, 0.24313725490196078],
                7: [0.788235294117647, 0.6627450980392157, 0.592156862745098],
                8: [0.8745098039215686, 0.803921568627451, 0.8941176470588236],
                9: [0.5333333333333333, 0.4392156862745098, 0.6784313725490196],
                10: [0.07450980392156863, 0.6, 0.5725490196078431],
                11: [0.788235294117647, 0.9215686274509803, 0.984313725490196],
                12: [0.6196078431372549, 0.403921568627451, 0.3843137254901961],
                13: [0.9803921568627451, 0.796078431372549, 0.07058823529411765],
                14: [0.8, 0.47058823529411764, 0.09411764705882353],
                15: [0.984313725490196, 0.7450980392156863, 0.5725490196078431],
                16: [0.9764705882352941, 0.8705882352941177, 0.8117647058823529],
                17: [0.9686274509803922, 0.9686274509803922, 0.6196078431372549],
                18: [0.9372549019607843, 0.35294117647058826, 0.615686274509804],
                19: [0.5529411764705883, 0.7098039215686275, 0.807843137254902],
                20: [0.20784313725490197, 0.3058823529411765, 0.13725490196078433],
                21: [0.058823529411764705, 0.2901960784313726, 0.611764705882353],
                22: [0.0, 0.3333333333333333, 0.4745098039215686],
                23: [0.24705882352941178, 0.5176470588235295, 0.6666666666666666],
                24: [0.7803921568627451, 0.13333333333333333, 0.1568627450980392],
                25: [0.9529411764705882, 0.592156862745098, 0.7529411764705882],
                26: [0.10196078431372549, 0.10196078431372549, 0.10196078431372549],
                27: [0.3254901960784314, 0.17254901960784313, 0.5411764705882353],
                28: [0.7568627450980392, 0.6235294117647059, 0.4392156862745098],
                29: [1.0, 0.5372549019607843, 0.10980392156862745],
                30: [0.39215686274509803, 0.47843137254901963, 0.30980392156862746],
                31: [0.803921568627451, 0.8784313725490196, 0.5333333333333333],
                32: [0.7098039215686275, 0.11372549019607843, 0.5529411764705883],
                33: [0.9686274509803922, 0.5647058823529412, 0.5137254901960784],
                34: [0.5568627450980392, 0.7803921568627451, 0.5725490196078431],
                35: [0.9372549019607843, 0.3058823529411765, 0.13333333333333333],
                36: [0.7647058823529411, 0.7647058823529411, 0.5333333333333333],

                37: [0.12549019607843137, 0.27450980392156865, 0.8392156862745098],
                38: [0.9137254901960784, 0.43529411764705883, 0.28627450980392155],
                39: [0.3843137254901961, 0.8431372549019608, 0.5372549019607843],
                40: [0.6196078431372549, 0.09803921568627451, 0.8196078431372549],
                41: [0.20784313725490197, 0.8470588235294118, 0.9450980392156862],
                42: [0.9333333333333333, 0.27450980392156865, 0.8509803921568627],
                43: [0.5490196078431373, 0.803921568627451, 0.1607843137254902],
                44: [0.8392156862745098, 0.09411764705882353, 0.6274509803921569],
                45: [0.1568627450980392, 0.5764705882352941, 0.3411764705882353],
                46: [0.8392156862745098, 0.48627450980392156, 0.08235294117647059],
                47: [0.43529411764705883, 0.25098039215686274, 0.9176470588235294],
                48: [0.9764705882352941, 0.6431372549019608, 0.1568627450980392],
                49: [0.30980392156862746, 0.6745098039215687, 0.8274509803921568],
                50: [0.7529411764705882, 0.06666666666666667, 0.3843137254901961],
                51: [0.12549019607843137, 0.8392156862745098, 0.7411764705882353],
                52: [0.9058823529411765, 0.20784313725490197, 0.13725490196078433],
                53: [0.3607843137254902, 0.1411764705882353, 0.9098039215686274],
                54: [0.8431372549019608, 0.396078431372549, 0.6627450980392157],
                55: [0.21568627450980393, 0.8784313725490196, 0.23921568627450981],
                56: [0.9333333333333333, 0.14901960784313725, 0.7568627450980392],
                57: [0.4666666666666667, 0.7607843137254902, 0.13725490196078433],
                58: [0.9137254901960784, 0.3215686274509804, 0.6745098039215687],
                59: [0.30980392156862746, 0.8588235294117647, 0.9764705882352941],
                60: [0.8156862745098039, 0.09019607843137255, 0.4980392156862745],
                61: [0.5843137254901961, 0.9137254901960784, 0.2980392156862745],
                62: [0.10980392156862745, 0.2823529411764706, 0.9098039215686274],
                63: [0.9019607843137255, 0.49019607843137253, 0.12549019607843137],
                64: [0.2784313725490196, 0.9058823529411765, 0.6627450980392157],
                65: [0.7490196078431373, 0.08235294117647059, 0.2901960784313726],
                66: [0.09411764705882353, 0.6274509803921569, 0.8941176470588236],
                67: [0.8941176470588236, 0.33725490196078434, 0.5803921568627451],
                68: [0.5372549019607843, 0.9137254901960784, 0.12941176470588237],
                69: [0.2823529411764706, 0.1411764705882353, 0.9176470588235294],
                70: [0.8745098039215686, 0.6392156862745098, 0.1450980392156863],
                71: [0.21568627450980393, 0.8235294117647058, 0.9450980392156862],
                72: [0.9176470588235294, 0.30196078431372547, 0.1568627450980392],
                73: [0.48627450980392156, 0.22745098039215686, 0.8901960784313725],
                74: [0.8980392156862745, 0.5137254901960784, 0.09411764705882353],
                75: [0.1607843137254902, 0.6627450980392157, 0.9137254901960784],
                76: [0.8784313725490196, 0.3803921568627451, 0.6509803921568628],
                77: [0.6549019607843137, 0.8901960784313725, 0.1568627450980392],
                78: [0.28627450980392155, 0.12549019607843137, 0.9098039215686274],
                79: [0.8980392156862745, 0.6509803921568628, 0.10588235294117647],
                80: [0.23921568627450981, 0.6784313725490196, 0.8549019607843137],
                81: [0.8352941176470589, 0.2196078431372549, 0.1411764705882353],
                82: [0.5764705882352941, 0.8549019607843137, 0.33725490196078434]
                # 0: [0.8, 0.47058823529411764, 0.09411764705882353],  # Mesenchyme
                # 1: [0.5568627450980392, 0.7803921568627451, 0.5725490196078431],  # NMP
                # 2: [1.0, 0.5372549019607843, 0.10980392156862745],  # Endothelium
                # 3: [0.4980392156862745, 0.40784313725490196, 0.4549019607843137],  # ExE endoderm
                # 4: [0.9686274509803922, 0.9686274509803922, 0.6196078431372549],  # Surface ectoderm
                # 5: [0.07450980392156863, 0.6, 0.5725490196078431],  # Intermediate mesoderm
                # 6: [0.9372549019607843, 0.35294117647058826, 0.615686274509804],  # Gut
                # 7: [0.5333333333333333, 0.4392156862745098, 0.6784313725490196],  # ExE mesoderm
                # 8: [0.10196078431372549, 0.10196078431372549, 0.10196078431372549],  # Parietal endoderm
                # 9: [0.6196078431372549, 0.403921568627451, 0.3843137254901961],  # Caudal epiblast
                # 10: [0.24705882352941178, 0.5176470588235295, 0.6666666666666666],  # Caudal Mesoderm
                # 11: [0.3254901960784314, 0.17254901960784313, 0.5411764705882353],  # Allantois
                # 12: [0.396078431372549, 0.6588235294117647, 0.24313725490196078],  # Rostral neurectoderm
                # 13: [0.5529411764705883, 0.7098039215686275, 0.807843137254902],  # Paraxial mesoderm
                # 14: [0.788235294117647, 0.9215686274509803, 0.984313725490196],  # Pharyngeal mesoderm
                # 15: [0.9529411764705882, 0.592156862745098, 0.7529411764705882],  # Def endoderm
                # 16: [0.39215686274509803, 0.47843137254901963, 0.30980392156862746],  # Forebrain Midbrain Hindbrain
                # 17: [0.20784313725490197, 0.3058823529411765, 0.13725490196078433],  # Caudal neurectoderm
                # 18: [0.7098039215686275, 0.11372549019607843, 0.5529411764705883],  # Cardiomyocytes
                # 19: [0.0, 0.3333333333333333, 0.4745098039215686],  # Somitic mesoderm
                # 20: [0.058823529411764705, 0.2901960784313726, 0.611764705882353],  # Notochord
                # 21: [0.9647058823529412, 0.7490196078431373, 0.796078431372549],  # Visceral endoderm
            }
            self.tissues_to_plot = [20]
            self.tissues_to_plot = [
                t for t in self.tissues_to_plot if t in embryo.all_tissues
            ]
            if len(self.tissues_to_plot) < 1:
                self.tissues_to_plot = list(self.embryo.all_tissues)
            cells = sorted(self.embryo.all_cells)
            positions = [self.embryo.pos_3D[c] for c in cells]
            shown = [self.embryo.tissue[c] in self.tissues_to_plot for c in cells]
            if not any(shown):
                shown = [True] * len(cells)
            properties = {"cells": cells}

            properties["gene"] = [0 for _ in cells]
            # if 0 < len(self.embryo.all_tissues.difference(self.color_map_tissues)):
            #     nb_tissues = len(self.embryo.all_tissues)
            #     self.color_map_tissues = {
            #         v: cm.tab20(i / nb_tissues)
            #         for i, v in enumerate(self.embryo.all_tissues)
            #     }
            #

            self.original_color_map_tissues = self.color_map_tissues.copy()
            colors_rgb = [
                self.color_map_tissues.get(self.embryo.tissue[c], [0, 0, 0])
                for c in cells
            ]

            self.viewer.dims.ndisplay = 3
            points = self.viewer.add_points(
                positions,
                face_color=colors_rgb,
                properties=properties,
                metadata={"gene": None, "2genes": None,"3genes": None},
                shown=shown,
                size=15,
            )

            self.all_tissues = [
                self.embryo.corres_tissue.get(t, f"{t}")
                for t in self.embryo.all_tissues
            ]
            self.all_tissues = sorted(self.all_tissues)


            self.tab1 = QTabWidget()
            if 'orig.ident' in self.embryo.anndata.obs:

                self.all_layers = [
                    t
                    for t in self.embryo.anndata.obs['orig.ident'].unique().tolist()
                    # for t in self.embryo.embryo.anndata.obs.get_loc('orig.ident').unique()
                ]
                self.all_layers = sorted(self.all_layers)
                layer_container = self.build_layer_container()
                self.tab1.addTab(layer_container.native, "Slices")
            if 'germ_layer' in self.embryo.anndata.obs:

                self.all_germ_layers = [
                    t
                    for t in self.embryo.anndata.obs['germ_layer'].unique().tolist()
                    # for t in self.embryo.embryo.anndata.obs.get_loc('orig.ident').unique()
                ]
                self.all_germ_layers = sorted(self.all_germ_layers)
                Germ_layer_container = self.build_germ_layer_container()
                self.tab1.addTab(Germ_layer_container.native, "Germ Layers")
            tissue_container = self.build_tissue_selection()
            SelectXY_container = self.selectXY_container()


            self.tab1.addTab(tissue_container.native, "Tissues")
            surf_container = self.build_surf_container()
            self.tab1.addTab(SelectXY_container.native,"SelectXY")
            last_tab = self.tab1.addTab(surf_container.native, "Surfaces")
            self.tab1.nb_tabs = last_tab

            self.tab2 = QTabWidget()
            metric_1g_container = self.build_metric_1g_container()
            metric_2g_container = self.build_metric_2g_container()
            metric_3g_container = self.build_metric_3g_container()
            umap_container = self.build_umap_container()
            diff_expr_container = self.build_diff_expr_container()
            Moran_container = self.build_Moran_container()
            self.tab2.addTab(metric_1g_container.native, "Single Metric")
            self.tab2.addTab(metric_2g_container.native, "2 Genes")
            self.tab2.addTab(metric_3g_container.native, "3 Genes")
            self.tab2.addTab(umap_container.native, "umap")
            self.tab2.addTab(Moran_container.native, "Moran")
            last_tab = self.tab2.addTab(diff_expr_container.native, "Diff Expr")
            self.tab2.nb_tabs = last_tab

            self.viewer.window.add_dock_widget(
                self.tab1, name="Tissue visualization"
            )
            self.viewer.window.add_dock_widget(
                self.tab2, name="Metric visualization"
            )
            self.show = show
        except Exception as e:
            # 捕获异常并记录日志
            logging.exception("An exception occurred: %s", str(e))
