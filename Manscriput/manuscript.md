# Title:

# Deformation of wood cell wall during three-point bending test analyzed by computer vision and machine learning

---

Authors: Shuoye Chen, XXXX, Junji Sugiyama

Affiliations: Graduate School of Agriculture, Kyoto University, 

E-mail: chenshuoye@gmail.com

---

## Abstract



## Keywords:

Cell wall deformation, Mechanical property, Semantic segmentation, U-net, Computer vision, Deep learning, Particle tracking

---

## 1. introduction

    Wood is a natural cellular material, it has complex structure with different cell types (anatomical features) acting together to serve the needs of living tree []. Also, as an anisotropic material,  wood has excellent mechanical properties parallel to the grain ( longitudinal direction), while its mechanical properties perpendicular to the grain (transverse direction) are relatively weak and varied among different wood species with relation to their unique anatomical features. 

    To completely unveil the relationship between anatomical features and mechanical behavior of wood, the quantitative and accurate analysis of local deformation of anatomical features during the mechanical test is an important subject. Up to now, mainly two approaches have been developed to understand how anatomical features affect the mechanical behavior of wood in transverse direction. 

    The first one is a **top-down approach** that is direct microscopic observation of deformation of anatomical features during or after the mechanical test. Ando and Onda (1999) used wet-type scanning electron microscope (SEM) to observe the compression of wood cell wall. Combing with image analysis, it was found that the first fracture of cell wall occurred in one tangential row of earlywood tracheids just after the load displacement curve exceeded the proportional limit. Müller et al. (2003) observed cell deformation of both softwood (spruce) and hardwood (oak and beech) at different yielding stage of compression test by using both SEM and light microscope for concluding different fracture pattern of anatomical features in those species. Hwang et al. (2021) used the replica method to intermittently analyze the cell wall deformation of flat-swan, quarter-swan and rift-swan in transverse direction of wood due to three point bending test. The rift-swan of softwood exhibited a unique shear deformation of earlywood cell wall contributing for the extremely large flexural deformation. Such direction observation provides important information to understand the in-situ deformation of wood specimen.

    The second approach is a **bottom-up** approach that simulate the mechanical properties of wood considering its microstrcuture. 



    On the other hand, with the development of computer vision in the field of artificial intelligence, the semantic segmentation has been proposed as a promising approach to label each pixel of an image with a corresponding class of what is represented. Recently, the deep-learning based semantic segmentation model has been developed and those technology that has large field of application and already been adapted to for self-driving vehicles and  analysis of biomedical image for medical diagnosis. 

    If such approach can be applied for analyzing the cell wall deformation, it provides a great possibility to simultaneously analyze almost all local changes in anatomical features and their interaction during the mechanical test.

    The information in this study helps to optimizing the up-down approach and bottom up approach.

    Therefore, as a pioneering try in this study, the semantic segmentation model has been built for conducting partition of anatomical features, and their local deformation during the micro three-point bending test were precisely analyzed.

## 2. materials and method

#### 2.1 specimen preparation

   Hinoki (*Chamaecyparis obtusa*) was used in this study. Three types of (flat-swan, rift-swan and quarter-swan) samples were firstly prepared considering their orientation of annual ring by visual confirmation. The annual ring aligned at horizontal direction and vertical direction were 0° and 90°, respectively. The sample with the angle of annual ring of 0° to 30° was defined as flat-swan, 30° to 60° was defined as rift-swan and 60° to 90° were defined as quarter-swan. After that, the 5 specimens of flat-swan, quarter-swan and rift-swan, respectively, were prepared with the dimension of 10 mm (longitudinal) x 20 mm (width) x 1.5 mm (thickness). Then, the cross section of all specimens were smoothed by a sliding microtome (Yamato, co Ltd...). All specimens were conditioned at a plastic glove box at 60% relative humidity (RH) and 25°C by using sodium bromide solution for more than two weeks.

#### 2.2 micro three-point bending test

    After the conditioning, all specimens were subjected to the micro three-point bending test. The Fig. A customized metal jig was used for the test. A motor () with test speed of 1mm/min was used to horizontally bend the specimen. And a 100N load cell () was used to record the force, the sampling speed is 1Hz. During the test, a stereo-microscope (Leica DMS300, Leica Camera AG, Germany) was used to record deformation of wood cell wall by video mode with 30 fps. The resolution was 1080p and the length of one pixel is equal to about 2.09 *µ*m. All experiment was conducted at 60-65% RH and 25 to 27°C.

<img title="" src="../Figures/apparatus.png" alt="appratus.png" data-align="center" width="417">

Fig. X The illustration of micro three-point bending test. (a) The illustrated apparatus used for the test. (b) Cross section of wood specimen observed by stereo-microscope.

#### 2.3 building deep learning based semantic segmentation model

    After the video taking during the bending test, the first image at every second of tje image was captured for preparing the image sequence. The 12 original images with 256 pixels x 256 pixels were cropped from the image sequence recorded by. The watershed segmentation was firstly applied for label the boundary of wood cell wall. The unlabeled part was manually modified to make their corresponding ground truth masks with cell wall boundary labeled in white and background labeled in black were manually prepared. Finally, The 12 sets of original image and corresponding ground truth mask were used for building semantic segmentation model. And the asymmetric U-net architecture with batch normalization was used for the model training. (Fig.X )

<img title="" src="../Figures/mask_preparation.png" alt="mask_preparation.png" data-align="center" width="635">

Fig. X Preparation of data set for semantic segmentation model training. (a) Cropped patch of cross section of wood; (b) cell wall boundary labeled mask by watershed segmentation algorithm (c) manually corrected image mask. The scale bar indicates length of 100 micrometer

#### 2.4 image prediction and tracking of cell wall deformation

    After model training, the model with the patch blending algorithm implemented by Vooban () were used to partition all cell walls in the image sequence with 1920 pixels x 1080 pixels. After predicting image sequence, watershed segmentation was applied again to achieve the instance segmentation of cell walls. Finally, a tracking algorithm (Crocker-Grier linking algorithm) implemented by trackpy () was used to link the same cell walls exist in each image.

<img title="" src="../Figures/particle_linking.png" alt="particle_linking.png" width="638" data-align="center">

Fig.X tracking the cell wall deformation during mechanical test. (a) watershed segmentation of predicted image by trained U-net model to achieve instance segmentation; (b) The coordinates of centriods of each cell wall were exacted as the features for particle linking; (c)  trajectories was found by Crocker-Grier linking algorithm.

#### 2.5 parameters measurement for cell wall deformation analysis

    After the tracking of cell wall, scikit-image was used to measured the area, eccentricity, major/minor axis length, vertical/horizontal length of bounding box and maximum Feret diameter of each cell wall for analyzing their intensity of cell wall deformation (Fig. X). And their rate of change during the bending test was also calculated based on the following equation:

<img title="" src="../Figures/parameters_calculation.png" alt="parameters_calculation.png" data-align="center" width="413">

the n indicates the order of the observed image sequence. The i indicates the measured parameters.

<img title="" src="../Figures/parameters_measurment.png" alt="parameters_measurment.png" data-align="center" width="450">

Fig. X the measurement parameters to evaluate the intensity of deformation of cell wall

## 3. results and discussion

#### 3.1 mechanical properties of flat-swan, quarter-swan and rift-swan

discuss the force-displacement curve, modulus of elasticity and modulus of rupture

<img src="../Figures/dis_moe_mor.png" title="" alt="dis_moe_mor.png" data-align="center">

Fig. X mechanical properties of flat-swan, quarter-swan and rift-swan of hinoki specimens. (a) load and displacement of three types of hinoki specimens during micro three-point test. (b) MOE (modulus of elasticity) and MOR (modulus of rupture) of three types of hinoki specimen; the error bars indicate the standard deviation.

#### 3.2 Validation of U-net model and cell wall deformation tracking

discuss the training results (loss vs. epoch, accuracy/f1_score vs. epoch) 

show the predicted mask (problem: the latewood part was not well predicted)

<img title="" src="../Figures/large_img_predicted.png" alt="large_img_predicted.png" width="545" data-align="center">

Fig. X cell wall boundary prediction by trained U-net model. (a) binary cross entropy loss plotted against the training epochs; (b) input original image; (c) predicted image

Table. X the values of metrics for predicted images by trained U-net model

| accuracy     | f1_score     | recall       | precision    |
|:------------:|:------------:|:------------:|:------------:|
| 0.92 (0.006) | 0.82 (0.017) | 0.82 (0.019) | 0.82 (0.017) |

<img title="" src="../Figures/partial_deformation.png" alt="partial_deformation.png" data-align="center" width="380">

Fig.x typical deformation of wood cell wall for three types of hinoki specimens during micro three-point bending test. (a) cell wall deformation of flat-swan specimen, upper: compression part, lower: tension part; (b) cell wall deformation of quarter-swan specimen, upper: compression part, lower: tension part; (c) cell wall deformation of rift-swan specimen, upper: compression part, lower: tension part. The scale bar indicates 50 $μ$m

#### 3.3 Visualization of the intensity of cell wall deformation

show the map with several parameters as example, discuss the cell wall deformation from elastic region to plastic region

#### 3.4 Relationship between changes in intensity of cell wall deformation and Stress-strain curve

The variety of the cell wall deformation pattern and its relationship with  strain-stress curve

## 4. Conclusion

## 5. Reference

> Forest Products Laboratory (1999) Wood Handbook-Wood as an Engineering Material; General Technical Report FPL-GTR-113; U.S. Department of Agriculture, Forest Service, Forest Products Laboratory: Madison, WI, USA.

> Ando K and  Onda H (1999) Mechanism for deformation of wood as a honeycomb structure I: effect of anatomy on the initial deformation process during radial compression. J Wood Sci 45: 120-125.

> Ando K and Onda H (1999) Mechanism for deformation of wood as a honeycomb structure II: First buckling mechanism of cell walls under radial compression using the generalized cell model. J Wood Sci 45:250-253.

> Gibson LJ,  Ashby  MF (1998) Cellular Solids: Structure and Properties; Pergamon Press: New York, NY, USA.

> Müller U, Gindl W, Teischinger A (2003) Effects of cell anatomy on the plastic and elastic behaviour of different wood species loaded perpendicular to grain. IAWA J 24: 117–128.

> Watanabe  U,  Norimoto M,  Ohgama T,  Fujita  M (1999)  Tangential Young’s modulus of coniferous early wood investigated using cell models. Holzforschung 53: 209–214.

> Watanabe U,  Norimoto  M,  Morooka T (2000) Cell wall thickness and tangential Young’s modulus in coniferous early wood.  J Wood Sci: 46, 109–114.

> Watanabe U,  Fujita M,  Norimoto M (2002) Transverse Young’s moduli and cell shapes in coniferous early wood. Holzforschung 56: 1–6.

> Gibson LJ (2012)  The hierarchica structure and mechanics of plant materials. J R Soc Interface 9: 2749-2766.



## 6. Acknowledgement
