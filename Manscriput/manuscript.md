# Title:

# Flexural behavior of wood in transverse direction studied by computer vision and machine learning

---

Authors: Shuoye Chen, XXXX, Junji Sugiyama

Affiliations: Graduate School of Agriculture, Kyoto University, 

E-mail: chenshuoye@gmail.com

---

## Abstract

A deep-learning based semantic segmentation approach (U-Net) was used to partition anatomical features in cross section of hinoki during micro three-point bending test. With the help of Crocker-Grier linking algorithm, thousands of cells were successfully extracted. Then, several parameters (area, eccentricity, major/minor axis length, vertical/horizontal bounding box length) were used to evaluate the intensity of their deformation. Finally, 2D mapping of a deformation intensity distribution was successfully built. The novel approach developed in this study showed the great possibility for understanding the relationship between anatomical features and mechanical behavior of wood.

## Keywords:

Flexural behavior, Cell wall deformation, Semantic segmentation, Individual cell tracking, Computer vision, Deep learning

---

## 1. introduction

Wood is a natural cellular material, it has complex structure with different cell types (anatomical features) acting together to serve the needs of living tree [1]. Also, as an anisotropic material,  wood has excellent mechanical properties parallel to the grain ( longitudinal direction), while its mechanical properties perpendicular to the grain (transverse direction) are relatively weak [2] and varied among different wood species with relation to their unique anatomical features [1].

From ancient time, human started to use wood as a construction materials for building civilization considering the microstructure of wood in transverse direction. For instance, a traditional roofing method called kokerabuki in Japanese.

To completely unveil the relationship between anatomical features and mechanical behavior of wood, the quantitative and accurate analysis of local deformation of anatomical features during the mechanical test is an important subject. Up to now, wood scientists developed several approaches from two perspectives for understanding how anatomical features affect the mechanical behavior of wood in transverse direction.

The first one is **top-down perspective** that is direct microscopic observation of deformation of anatomical features during or after the mechanical test. Ando and Onda (1999) [3] used wet-type scanning electron microscope (SEM) to observe the compression of wood cell wall. Combing with image analysis, it was found that the first fracture of cell wall occurred in one tangential row of earlywood tracheids just after the load displacement curve exceeded the proportional limit. Müller et al. (2003) [4] observed cell deformation of both softwood (spruce) and hardwood (oak and beech) at different yielding stage of compression test by using both SEM and light microscope for concluding different fracture pattern of anatomical features in those species. Hwang et al. (2021) [5] used the replica method to intermittently analyze the cell wall deformation of flat-swan, quarter-swan and rift-swan in transverse direction of wood due to three point bending test. The rift-swan of softwood exhibited a unique shear deformation of earlywood cell wall contributing for the extremely large flexural deformation. Such direct microscopic observation methods provides important information to understand the in-situ deformation of wood microstructures.

The second one is **bottom-up perspective** that is the mechanical simulation of wood properties considering its hierarchical structure.  Watanabe et al. [6-8] firstly used fast Fourier transform (FFT) to extract the characteristics such as axial length of tangential and radial cell wall, cell wall thickness, etc. of several conifer wood species for simulating the tangential Young's modulus by cell wall model. Ando and Onda [9] used generalized cell wall model to successfully simulated the first buckling mechanism of conifer wood cell wall under radial compression. Holmberg et al. [10] used finite element method (FEM) to simulate the nonlinear mechanical behavior considering the irregular cell shape, anisotrpic layer structure of the cell walls and the periodic variations in density of wood. And their simulated deformation and fracture of wood was similar to the those found in the refining process of wood. De Magistris and Salmén (2008) [11] investigate the compression and in combined shear and compression deformation of cell wall with anisotropic one-layer cell walls and orthotropic multi-layer cell wall models by finite element method (FEM) . Their results indicates the cell structures is the key factors influencing the deformation pattern.  Recently the multi-scale FEM is adopted to simulate wood compression behavior under both axis and transverse loading [12]. it was found that transverse deformation of wood is gradual and uniform,  while the loading velocity greatly affects wood microstructure failure modes in axial loading. Those developed approaches are quite useful and powerful for providing comprehensive explanation of mechanical behavior of  wood.

On the other hand, in the field of computer vision, the semantic segmentation has been proposed as a important approach to label each pixel of an image with a corresponding class of what is represented. With the development of artificial intelligence, the deep-learning based semantic segmentation model such as U-Net[13], LinkNet [14], Feature Pyramid Networks [15], and Pyramid Scene Parsing Network [16] has been developed and those technology that has large field of application and already started to be applied into the field autonomous vehicles [17] and  analysis of biomedical image for medical diagnosis [18]. If the approach of semantic segmentation can also be adapted for analyzing the cell wall deformation, it provides a great possibility to simultaneously analyze almost all local changes in anatomical features and their interaction during the mechanical test. Furthermore, the observed information provides more accurate and quantitative image analysis and the collected cell wall geometry can also be used for more realistic mechanical simulation for optimizing developed both top-down and bottom-up approaches.

Therefore, in this study, the semantic segmentation model has been built for the partitioning of anatomical features of hinoki wood, and their local deformation during the micro three-point bending test were precisely analyzed with the help of individual cell tracking algorithm.

## 2. materials and method

#### 2.1 specimen preparation

Hinoki (*Chamaecyparis obtusa*) was used in this study. Three types of (flat-swan, rift-swan and quarter-swan) samples were firstly prepared considering their orientation of annual ring by visual confirmation. The annual ring aligned at horizontal direction and vertical direction were 0° and 90°, respectively. The sample with the angle of annual ring of 0° to 30° was defined as flat-swan, 30° to 60° was defined as rift-swan and 60° to 90° were defined as quarter-swan. After that, the 5 specimens of flat-swan, quarter-swan and rift-swan, respectively, were prepared with the dimension of 10 mm (longitudinal) x 20 mm (width) x 1.5 mm (thickness). Then, the cross section of all specimens were smoothed by a sliding microtome (TU-213, Yamato kohki industrial Co., Ltd., Japan). All specimens were then  conditioned at a plastic glove box at 60% relative humidity (RH) and 25°C by using sodium bromide solution for more than two weeks.

#### 2.2 micro three-point bending test

After the conditioning, all specimens were subjected to the micro three-point bending test. The customized metal jig (Fig. X (a)) was used for the test. A motor (BLM230P-GFV2, ORIENTAL MOTOR Co.,Ltd., Japan) with test speed of 1mm/min was used to horizontally bend the specimen. And a 200N load cell (LUR-A-200NSA1, Kyowa Electronic Instruments Co., Ltd., Japan) with sensor interface (PCD-320A, Kyowa Electronic Instruments Co., Ltd., Japan) was used to record the force, the sampling speed is 1Hz. During the test, a stereo-microscope (Leica DMS300, Leica Camera AG, Germany) was used to record the deformation of cell wall by video mode with 30 fps. The resolution was 1080p and the length of one pixel is equal to about 2.09 *µ*m. All experiment was conducted at 60% RH and 25°C.

<img title="" src="../Figures/01_apparatus.png" alt="appratus.png" data-align="inline" width="417">

Fig. 1 The illustration of micro three-point bending test. (a) The illustrated apparatus used for the test. (b) Cross section of wood specimen observed by stereo-microscope.

#### 2.3 Deep learning based semantic segmentation model

For preparation of training dataset and model training, after the video taking during the bending test, the first image at every second of the video was captured for preparing the image sequence. The 12 original images with 256 pixels x 256 pixels were cropped from the image sequence recorded by. The watershed segmentation was firstly applied for label the boundary of wood cell wall. The unlabeled part was manually modified to make their corresponding ground truth masks with cell wall boundary labeled in white and background labeled in black were manually prepared. Finally, The 12 sets of original image and corresponding ground truth mask were used for building semantic segmentation model. And the asymmetric U-net architecture was used for the model training. The binary cross entropy was used as the loss function, and adam was used as the optimizer. The learning rate was 0.0001. During the model training, the augmentation of image by image generator was applied (Fig.2).

<img title="" src="../Figures/02_mask_preparation.png" alt="02_mask_preparation.png" data-align="inline" width="569">

Fig. 2 Preparation of data set for semantic segmentation model training. (a) Cropped patch of cross section of wood; (b) cell wall boundary labeled mask by watershed segmentation algorithm (c) manually corrected image mask. The scale bar indicates length of 100 *μ*m

#### 2.4 Metrics for model evaluation

Four metrics were used for evaluating the trained model. They were accuracy, recall, precision, and f1-score. Those metrics were calculated from true positive (TP), false positive (FP), true negative (TN) and false negative (FN) obtained from the confusion matrix for the binary classification of cell boundary and background. The formula for the calculation were showed as below:

<img title="" src="../Figures/03_metrics.png" alt="metrics.png" width="271" data-align="inline">

#### 2.5 image prediction and individual cell tracking

After model training, the trained model combined with the patch blending algorithm implemented by Vooban [19] were used to conduct partition of all potential cells in the image sequence with 1920 pixels x 1080 pixels. After predicting all image sequence, watershed segmentation was applied again to achieve the instance segmentation of all cells. Finally, the coordinates of centriod of all cells were collected and a tracking algorithm (Crocker-Grier linking algorithm) [20] implemented by trackpy [21] was used to link the same cell walls exist in each image.

<img title="" src="../Figures/04_particle_linking.png" alt="particle_linking.png" width="568" data-align="inline">

Fig.3 tracking the cell wall deformation during mechanical test. (a) watershed segmentation of predicted image by trained U-net model to achieve instance segmentation; (b) The coordinates of centriods of each cell wall were exacted as the features for particle linking; (c)  trajectories was found by Crocker-Grier linking algorithm.

#### 2.6 parameters measurement for cell wall deformation analysis

Finally, after the tracking of individual cells existing at every image sequence, the area, eccentricity, length of major and minor axis of fitted ellipse, and length of vertical and horizontal length of bounding box for each cell wall were measured (Fig.X). Those measurements were implemented by python package: scikit-image[]. Furthermore, the fitted ellipse aspect ratio and the aspect ratio of vertical and bounding box aspect ratio were calculated. For evaluating the intensity of cell wall deformation, the changes in area, eccentricity, fitted ellipse aspect ratio and bounding box aspect ratio were calculated based on the following equation:

<img title="" src="../Figures/06_parameters_calculation.png" alt="parameters_calculation.png" data-align="inline" width="429">

the n indicates the order of the observed image sequence. The i indicates the measured parameters showed on Fig.X.

<img title="" src="../Figures/05_parameters_measurment.png" alt="04_parameters_measurment.png" width="443">

Fig.4 the measurement parameters to evaluate the intensity of deformation of cell wall

## 3. results and discussion

#### 3.1 flexural behavior of flat-sawn, quarter-sawn and rift-sawn in transverse direction

<img title="" src="../Figures/07_Hinoki_dis_MOE.png" alt="06_Hinoki_dis_MOE.png" width="460">

Fig.5 mechanical properties of flat-swan, quarter-swan and rift-swan of hinoki specimens in transverse direction. (a) load and displacement of three types of hinoki specimens during micro three-point test. (b) MOE (modulus of elasticity) and MOR (modulus of rupture) of three types of hinoki specimen; the error bars indicate the standard deviation.

The Fig.5 shows the difference in the mechanical properties of flat-, quarter-, and rift- sawn in transverse direction. During the micro three-point bending test, the rift-sawn specimen showed smallest load with largest displacement at around 3.3 mm, resulting the smallest modulus of elasticity (MOE) and modulus of rupture (MOR) (Fig. X (a)). If we assume the linear stage of load-displacement as elastic region and nonlinear as plastic region. The rift-sawn specimen showed the largest plastic region. In contrast,  the quarter-sawn showed the largest MOE and MOR (Fig.X (b)) with the smallest plastic region.

Those results agree with the previous study [5], which suggests the orientation of annual ring plays an important role on the flexural behavior of wood in transverse direction. It also demonstrates that built micro three bending test system in this study is reliable for discussing the mechanical properties of wood.

#### 3.2 Validation of U-Net model and large image prediction.

With the development of artificial intelligence, the fully convolutional networks (FCN) was proposed for conducting the semantic segmentation []. Furthermore, as an improvement of FCN, the U-Net architecture was proposed by Ronneberger et al.[] , which is designed to allow fewer training samples for model training. It is a U-shape architecture consisting of encoder blocks, decoder blocks and skip connections. It has became one of the most popular approaches for any semantic segmentation tasks. Recently, the U-net model has been applied for the segmentation of plant tissues[] and xylem vessels in stained cross-section of wood with excellent accuracy. Therefore, in this study, the U-Net has been selected for building model.

The Fig.X (a) shows the evolution of binary cross entropy loss during 100 epochs training with U-Net architecture. After about 40 epochs training, the validation loss tended to became almost constant, while the loss continue to decreasing to about 0.1.  Four metrics were used for evaluating the trained model. The Table 1 showed the averaged value of those four metrics with the standard deviation. 

The Fig.X (b) shows a example of input original image and Fig.X (c) is the predicted image of original image through the trained model. The combination of patch blending algorithm [] and the model worked well to predict large image. The most of trachied cells seemed to be well segmented, while the partition of latewood trachied cells was not well predicted. It is known that the latewood trachied cell has quite small cell area and cell lumen makes it difficult to prepare the accurate masks from the image taken by the stereo microscope. To overcome the problem, the improvement of  image resolution will be needed by optimizing the methodology of microscopic observation. 

To further confirm the accuracy of the segmentation, the geometry parameters of a flat-swan specimen were measured. The vertical bounding box and horizontal bounding box were regraded as cell radial diameter and cell tangential diameter. The Fig.X showed the distribution of typical parameters measured from the segmented cells. The averaged cell area, cell eccentricity, cell radial diameter and cell tangential diameter were 955 *μ*m<sup>2</sup> (306), 0.596 (0.146), 37.5 *μ*m (7.63) and 34.8 *μ*m (6.62), respectively. Those measured geometry values agree with the previous research [] suggesting a accurate segmentation model has been built.

Table. 1 the evaluated metrics for predicted images by trained U-Net model. The values in parentheses indicate the standard deviation.

| recall       | precision    | F1           | accuracy     |
|:------------:|:------------:|:------------:|:------------:|
| 0.82 (0.019) | 0.82 (0.017) | 0.82 (0.017) | 0.92 (0.006) |

<img title="" src="../Figures/08_large_img_predicted.png" alt="large_img_predicted.png" width="475">

Fig. 6 cell wall boundary prediction by trained U-net model. (a) binary cross entropy loss plotted against the training epochs; (b) input original image; (c) predicted image. The scale bar indicates length of 400 *μ*m.

<img title="" src="../Figures/09_parameters_distribution.png" alt="08_parameters_distribution.png" data-align="inline">

Fig.X the distribution of cell area (a), cell eccentricity (b), cell tangential diameter (c) and radial diameter measured from one specimen before.

#### 3.3 Typical deformation patterns of tracheid cell wall in three types of specimens

The Fig.7 showed a typical deformation pattern of tracheid earlywood cell wall located at the compression part and tension part of three types of specimens. The changes in shape of single cell wall located at both compression part and tension part of specimen during the mechanical test were intermittently extracted. And our results agree with the previous study that use replica method to intermittently observe cell wall deformation of rift-swan specimen [].

For flat-sawn specimen, a uniaxial compression and tension of tangential cell wall were possibly occurred at the compression and tension part of specimen, respectively. And because of the orthogonal orientation of cell wall in quarter-swan, the deformation of radial cell wall was observed. As quarter-swan was fractured when displacement reach to around only 1 mm, the dimensional changes of cell wall were relatively smaller than that of flat-swan. 

Different with flat- and quarter-sawn specimen, the cell wall in rift-sawn seemed to show a different deformation pattern.  The shear deformation of cell wall along the vertical and horizontal direction was observed at compression part and tension part, respectively. Furthermore, such orientation of tracheid cells was quite similar to the uniaxial loading of honeycombs in the in-plane off-axial direction. Li et al.[] have simulated the in-plane yield strengths of the square honeycombs in different direction under the compression by theoretical approach and FEM method. They have concluded the square honeycombs show a strong anisotropy when loaded in different orientations. And the numerical simulation indicates that the axial yield strength of the square honeycomb have minimum values at the angle of orientation with 37 degree to 38 degree, which corresponding to the orientation of annual ring for rift-swan. Therefore,  we suppose such shear deformation induced by the off-axis loading of trachied cell is responsible for the large displacement, relatively low MOE and MOR of rift-swan specimen.

<img title="" src="../Figures/10_partial_deformation.png" alt="partial_deformation.png" width="473" data-align="inline">

Fig.7 Typical deformation of wood cell wall for three types of hinoki specimens during micro three-point bending test. (a) cell wall deformation of flat-swan specimen, upper: compression part, lower: tension part; (b) cell wall deformation of quarter-swan specimen, upper: compression part, lower: tension part; (c) cell wall deformation of rift-swan specimen, upper: compression part, lower: tension part. The scale bar indicates 50 μm

#### 3.4 Visualization of the distribution of trachied cell wall deformation

With the benefit of Grocker-Grier linking algorithm, the coordinates of centriods for each common tracheid cell existed at each frame were successfully linked. Therefore, thousands of common cells were extracted. Finally, after evaluating the intensity of the deformation by described in the section of materials and method, the 2d mapping of the intensity of deformation for three types of specimen was successfully built.  At the elastic region, all specimens showed relative slight and varied deformation for all parameters. When entering the plastic region, the cell wall deformation distribution differed. And the suitable parameter for the evaluation of deformation for those types of specimen were differed.

For flat-swan, the area seems to be the most suitable parameter for the deformation evaluation. As shown in Fig.X, the area of cell increased in tension part and decreased in compression part of specimen. And the intensity the changes of compression part is smaller than that of tension part. It might due to the existence of the latewood at compression part contributing the restriction of deformation of cells. Both significant increase and decrease in eccentricity were observed in tension part of specimen. It is because such changes highly depends on original shape of the cell. In other words. the horizontal tension of wood cell wall induces both increase in eccentricity for circle-shape cell while it might induce the decrease in eccentricity for vertical ellipse-shaped cell. The similar result was also observed in the case of fitted ellipse aspect ratio. For the changes in bounding box aspect ratio,  the cells located at central part of specimen showed reasonable results. The increase and decrease in bounding box aspect ratio were observed in compression part and tension part, respectively. However, the bending test also caused the curve of the specimen that also change in the orientation of cell located at the surrounding part of the specimen and influence the reliability of their measured bounding box aspect ratio. 

For quarter-swan specimen shown smallest plastic region in three types of specimen, the bounding box seems to be a promising parameter for the deformation evaluation. 



For rift-swan, the fitted ellipse aspect ratio is the most robust parameter for its deformation evaluation.

the concentrated deformation was observed at the innermost part of the compression part and outermost of the tension part for rift-sawn specimen. And such deformation reaches the maximum before the fracture of the specimen. Such large deformation is involved by the shear deformation of the cell wall along with the radial file unit.  the large deformation is contributed to the concentrated deformation of cell wall. And it makes it easier to predict the starting point of the fracture.

The ray might play an important role on to restrict the cell wall along the radial file, 

![flat-sawn_map.png](../Figures/11_flat-sawn_map.png)

Fig. 8 Intensity of cell wall deformation of flat-sawn specimen during micro three-point bending test evaluated by.

![quarter-sawn_map.png](../Figures/12_quarter-sawn_map.png)

Fig. 9 Intensity of cell wall deformation of quarter-sawn specimen during micro three-point bending test evaluated by.

![rift-sawn_map.png](../Figures/13_rift-sawn_map.png)

Fig. 10 Intensity of cell wall deformation of rift-sawn specimen during micro three-point bending test evaluated by.

<img title="" src="../Figures/14_Fracture_pattern.png" alt="Fracture_pattern.png" width="257" data-align="center">

Fig.X the fracture of the flat-, quarter-, and rift- swan specimen taken after micro three-point bending test. The scale bar indicates 400 $μ$m.

#### 3.5 Relationship between changes in intensity of cell wall deformation and stress-strain curve

As desribed above, the changes in area, changes in bounding box aspect ratio , and change in fitted ellipse aspect ratio have been concluded as suitable parameters discussing the deformation of cell for flat-sawn, quarter-sawn and rift-sawn, respectively. 

Therefore, the k-means clustering algorithm was applied to summarize the deformation pattern. And their relationship between strain and stress of the specimen were showed at Fig.11.

![kmeans_clustering_pattern.png](../Figures/15_kmeans_clustering_pattern.png)

Fig.11

## 4. Conclusion

A  deep-learning based semantic segmentation approach (U-Net) was used to partition anatomical features in cross section of hinoki during the micro three-point bending test. With the help of Crocker-Grier linking algorithm, thousands of cells were successfully extracted. Then, several parameters (area, eccentricity, major/minor axis length, vertical/horizontal bounding box length) were used to evaluate the intensity of
their deformation. Finally, 2D mapping of a deformation intensity distribution
was successfully built. The novel approach developed in this study showed the great possibility for understanding the relationship between anatomical features and the mechanical behavior of wood.

## 5. Reference

> 1. Forest Products Laboratory (1999) Wood Handbook-Wood as an Engineering Material; General Technical Report FPL-GTR-113; U.S. Department of Agriculture, Forest Service, Forest Products Laboratory: Madison, WI, USA.

> 2. Gibson LJ, Ashby MF (1998) Cellular Solids: Structure and Properties; Pergamon Press: New York, NY, USA.

> 3. Ando K and  Onda H (1999) Mechanism for deformation of wood as a honeycomb structure I: effect of anatomy on the initial deformation process during radial compression. J Wood Sci 45: 120-125.

> 4. Müller U, Gindl W, Teischinger A (2003) Effects of cell anatomy on the plastic and elastic behaviour of different wood species loaded perpendicular to grain. IAWA J 24: 117–128.

> 5. Hwang S, Isoda H, Nakagaw T, Sugiyama J (2021) Flexural anisotropy of rift-sawn softwood boards induced by the end-grain orientation. J Wood Sci 67: 14.

> 6. Watanabe  U,  Norimoto M,  Ohgama T,  Fujita  M (1999)  Tangential Young’s modulus of coniferous early wood investigated using cell models. Holzforschung 53: 209–214.

> 7. Watanabe U,  Norimoto  M,  Morooka T (2000) Cell wall thickness and tangential Young’s modulus in coniferous early wood.  J Wood Sci: 46, 109–114.

> 8. Watanabe U,  Fujita M,  Norimoto M (2002) Transverse Young’s moduli and cell shapes in coniferous early wood. Holzforschung 56: 1–6.

> 9. Ando K and Onda H (1999) Mechanism for deformation of wood as a honeycomb structure II: First buckling mechanism of cell walls under radial compression using the generalized cell model. J Wood Sci 45:250-253.

> Gibson LJ (2012)  The hierarchical structure and mechanics of plant materials. J R Soc Interface 9: 2749-2766.

> 10. Holmberg S, Persson K, Petersson H (1999) Nonlinear mechanical behaviour and analysis of wood and fibre materials. Comput Struct 72: 459-480.

> 11. De Magistris, F, Salmén L (2008) Finite Element modelling of wood cell deformation transverse to the fibre axis. Nord Pulp Pap Res J23: 240–246.

> 12. Zhong W, Zhang Z, Chen X, Wei Q, Chen G, Huang X (2021) Multi-scale finite element simulation on large deformation behavior of wood under axial and transverse compression conditions. Acta Mech Sin 37: 1136-1151.

> 13. Ronneberger O, Fischer P, Brox T (2015) U-Net: convolutional networks for
>     biomedical image segmentation, Lect. Notes Comput. Sci. (including Subser Lect Notes Artif Intell Lect Notes Bioinformatics) 9351:234–241.

> 14. Chaurasia A,  Culurciello E (2017) LinkNet: Exploiting encoder representations for efficient semantic segmentation. In Proceedings of the 2017 IEEE Visual Communications and Image Processing (VCIP), St. Petersburg, FL, USA, 1–4.

> 15. Lin T, Dollár P, Girshick R, He K, Hariharan B, Belongie S (2017) Feature pyramid networks for object detection. In: IEEE conference on computer vision and pattern recognition : 936–944.

> 16. Zhao H, Shi J, Qi X, Wang X, Jia J (2017) Pyramid scene pars-ing network. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2017, pp. 2881–2890 .

> 17. Janai J, Güney F, Behl A,Geiger A (2020) Computer Vision for Autonomous Vehicles: Problems, Datasets and State of the Art. Found. Trends Comput Graph Vis 12, 85. ???

> 18. Müller D, Kramer F. (2021) MIScnn: a framework for medical image segmentation with convolutional neural networks and deep learning. BMC Med Imaging 21: 12.

> 19. https://github.com/Vooban/Smoothly-Blend-Image-Patches

> 20. Crocker, J C, Grier, D G (1996) Methods of digital video microscopy for colloidal studies. J Colloid Interf Sci, 179: 298–310.

> 21. Allan, D. B., Caswell, T., Keim, N. C. & van der Wel, C. M. trackpy: Trackpy v0.5.0. (Zenodo, 2018). doi:10.5281/zenodo.1226458

> 22. Garcia-Pedrero, A, García-Cervigón I A, Olano J M , García-Hidalgo M, Lillo-Saavedra M, Gonzalo-Martín C, Caetano C, Calderón-Ramírez S (2020) Convolutional neural networks for segmenting xylem vessels in stained cross-sectional images. Neural Comput Appl 32: 17927-17939

> 23. Wolny A, Cerrone L, Vijayan A, Tofanelli R, Barro AV, Louveaux M, Wenzl C, Strauss S, Wilson-Sánchez D, Lymbouridou R, Steigleder S S, Pape C, Bailoni A, Duran-Nebreda S, Bassel GW, Lohman JU, Tsiantis M, Hamprecht FA, Scheitz K, Maizel A, Kreshuk A (2020) Accurate and versatile 3D segmentation of  plants tissues at cellular resolution, elife 29.

> Li X, Lu Z, Yang Z, Yang C (2018) Anisotropic in-plane mechanical behavior of square honeycombs under off-axis loading. Mater Des 158: 88-97.

## 6. Acknowledgement
