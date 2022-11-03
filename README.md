## Cell segmentation and tracking from time-series sequential images

#### General introduction

As a natural cellular material, wood has complex structure with different cell types (anatomical features) acting together to serve the needs of living tree. Those features have large variation with different wood species that highly influences the mechanical properties of wood . 

To unveil the relationship between anatomical features and mechanical behavior of wood, the quantitative and accurate analysis of local deformation of anatomical features during the mechanical test is an important subject. However, due to its technical difficulty, only a few studies around this topic have been conducted.

On the other hand, with the development of computer vision in the field of artificial intelligence, the semantic segmentation has been proposed as a promising approach to label each pixel of an image with a corresponding class of what is represented. If such approach can be applied into the field of wood science, it provides a possibility to simultaneously analyze almost all local changes in anatomical features and their interaction during the mechanical test.

Therefore, as the first try in this study, the semantic segmentation model has been built to conduct partition of anatomical features, and their local deformation during the micro three-point bending test were precisely analyzed.

You can find more infomation in :

> Chen Shuoye, Awano Tatsuya, Yoshinaga Arata and Sugiyama Junji. "Flexural behavior of wood in the transverse direction investigated using novel computer vision and machine learning approach" *Holzforschung*, vol. 76, no. 10, 2022, pp. 875-885. https://doi.org/10.1515/hf-2022-0096



#### What is in this repository ?

The repository contains the necessary codes that used for analyzing the intensity of cell wall deformation of wood specimen during the micro three-point bending test. They are two jupyter notebooks:

```python
1. cell_wall_deformation_analysis.ipynb
2. kmeans_clustering.ipynb
```

The first one is for the geometrical parameters extraction and calculation of deformation intensity. The second is the procedure for performing k-means clustering of deformation patterns and their relationship with stress-strain curve.

Another example of k-means clustering is published in:

> Imai Makiko, Mihashi Asako, Imai Tomoya, Kimura Satoshi,  Matsuzawa Tomohiko, Yaoi Katsuro, Shibata Nozomu, Kakeshita Hiroshi, Igarashi Kazuaki, Kobayashi Yoshinori and Sugiyama Junji. Selective fluorescence labeling: time-lapse enzyme visualization during sugarcane hydrolysis. *J Wood Sci* **65**, 17, 2019. https://doi.org/10.1186/s10086-019-1798-0

