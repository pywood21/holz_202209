# Flexural behavior of wood in transverse direction investigated using novel computer vision and machine learning approach

The final aim of this repository is to submit current work to international journal: Construction and building materials.

### Generation introduction of this repository:

As a natural cellular material, wood has complex structure with
different cell types (anatomical features) acting together to serve the needs
of living tree. Those features have large variation with different wood
species that highly influence the mechanical properties of wood . 

To unveil the relationship between anatomical features and mechanical behavior of wood, the quantitative and accurate analysis of local deformation of anatomical features during the mechanical test is an important subject. However, due to its technical difficulty, only a few studies around this topic have been conducted.

On the other hand, with the development of computer vision in the
field of artificial intelligence, the semantic segmentation has been proposed
as a promising approach to label each pixel of an image with a corresponding
class of what is represented. If such approach can be applied into the field of
wood science, it provides a possibility to simultaneously analyze almost all
local changes in anatomical features and their interaction during the mechanical
test.

Therefore, as a first try in this study, the semantic
segmentation model has been built to conduct partition of anatomical
features, and their local deformation during the micro three-point bending test
were precisely analyzed.

### What is in this repository ?

The repository contains the necessary codes that used for analyzing the intensity of cell wall deformation of wood specimen during micro three-point bending test. They are two jupyter notebooks:

```
1. cell_wall_deformation_analysis.ipynb
2. kmeans_clustering.ipynb
```

The first one is for the geometry parameters extraction and calculation of deformation intensity. The second is the procedure for performing k-means clustering of deformation patterns and their relationship with stress-strain curve.
