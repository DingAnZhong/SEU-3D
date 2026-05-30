import json
from .embryo import Embryo
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton,QLabel
from magicgui import widgets
from .display import DisplayEmbryo
from pathlib import Path
import anndata as ad

class ReadAdata(QWidget):
    """
    Widget to read anndata files and display their contents.
    """
    def show_components(self):
        try:
            file_path = Path(self.h5ad_file.value)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            adata = ad.read_h5ad(file_path)
            self.adata = adata
        except Exception as e:
            error_label = QLabel(f"错误: {str(e)}")
            self.layout.addWidget(error_label)
            return

        if not hasattr(self, 'tab_widget'):
            self.tab_widget = QTabWidget()
            self.layout.addWidget(self.tab_widget)
        else:

            while self.tab_widget.count() > 0:
                self.tab_widget.removeTab(0)

        tab_info = QWidget()
        tab_info.setLayout(QVBoxLayout())
        adata_info = widgets.TextEdit(
            value=adata,
            tooltip="Anndata object information",
        )
        tab_info.layout().addWidget(adata_info.native)
        self.tab_widget.addTab(tab_info, "Anndata Info")

        self.tissue = widgets.ComboBox(
            label="Tissue",
            choices=list(adata.obs.columns),
            value=adata.obs.columns[0] if not adata.obs.columns.empty else None,
        )

        self.spatial_coord = widgets.ComboBox(
            label="Spatial Coordinates",
            choices=list(adata.obsm.keys()),
            value=list(adata.obsm.keys())[0] if adata.obsm else None,
        )
        self.load_button_2 = QPushButton("Load Embryo")
        self.load_button_2.clicked.connect(self.load_embryo)
        selec_params = widgets.Container(
            widgets=[self.tissue, self.spatial_coord],
            layout="vertical",
        )
        tab_info.layout().addWidget(selec_params.native)
        tab_info.layout().addWidget(self.load_button_2)

    def load_embryo(self):
        """
        Load the embryo data and display it.
        """
        self.embryo = Embryo(
            data_path=self.h5ad_file.value,
            tissue_name=self.tissue.value,
            coordinate_3d_key=self.spatial_coord.value,
            adata = self.adata,
        )
        DisplayEmbryo(self.viewer, self.embryo)

    def __init__(self,napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.h5ad_file = widgets.FileEdit(label="H5AD file")
        self.layout.addWidget(self.h5ad_file.native)

        self.load_button_1 = QPushButton("Read anndata then show components")
        self.load_button_1.clicked.connect(self.show_components)
        self.layout.addWidget(self.load_button_1)
