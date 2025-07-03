# SEU-3D
基于napari的空间转录组可视化插件，也包括一些分析方法

## Quickstart
在终端中
### 创建环境
```python
conda env create -f env.yml
```
note!if adata.uns['rank_genes_groups']:numpy==1.26.4 (uns中存在这一列的话numpy需退回1.xx.xx版本)
### 安装包
```python
pip install --upgrade seu-3d
```
### 使用
```python
napari
```
#### 打开文件目录选择h5ad文件
![image](https://github.com/user-attachments/assets/47d8d99f-e2c7-41e1-be29-6671beef2d8d)
#### 加载文件可看到组件，在下拉框中选择展示的组织或细胞类型，以及空间坐标
![image](https://github.com/user-attachments/assets/4f94de15-365a-44a8-b796-a33ec60b950b)
#### 结果示例
![64ba8e66-9666-4ce1-9cac-01a49f5a549a](https://github.com/user-attachments/assets/0c6432a4-b6a1-4736-a0fb-4c4e0b8d498e)
