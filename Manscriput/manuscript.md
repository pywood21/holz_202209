# Title:

# Deformation of wood cell wall during three-point bending test studied by computer vision and machine learning

---

Authors: Shuoye Chen, XXXX, Junji Sugiyama

Affilliations: Graduate School of Agriculture, Kyoto University, 

E-mail: chenshuoye@gmail.com



---

## Abstract

## Keywords:

Cell wall deformation, Mechanical property, Semantic segmentation, U-net, Computer vision, Deep learning, Particle tracking

---

## 1. introduction

   As a natural cellular material, wood has complex structure with different cell types (anatomical features) acting together to serve the needs of living tree. Those features have large variation with different wood species that highly influence the mechanical properties of wood. To unveil the relationship between anatomical features and mechanical behavior of wood, the quantitative and accurate analysis of local deformation of anatomical features during the mechanical test is an important subject. However, due to its technical difficulty, only a few studies around this topic have been conducted.

   On the other hand, with the development of computer vision in the field of artificial intelligence, the semantic segmentation has been proposed as a promising approach to label each pixel of an image with a corresponding class of what is represented. If such approach can be applied into the field of wood science, it provides a possibility to simultaneously analyze almost all local changes in anatomical features and their interaction during the mechanical test.

   Therefore, as a pioneering try in this study, the semantic segmentation model has been built to conduct partitioning of anatomical features, and their local deformation during the micro three-point bending test were precisely analyzed.

---

## 2. materials and method

#### 2.1 specimen preparation

   Hinoki (*Chamaecyparis obtusa*) was used in this study. The 5 specimens of flat-swan, quarter-swan and rift-swan were prepared with the dimension of 10 mm (L) x 20 mm (width) x 1.5 mm (thickness). The cross section was smoothened by a sliding microtome (Yamato, co Ltd...). Then, all prepared specimens were conditioned at 60% RH and 25°C for more than two weeks.

#### 2.2 micro three-point bending test

   The specimen was horizontally bent by the customized metal jig, while a stereo-microscope (Lecia DMS100, ) was set vertically on the top of the specimen to record the deformation of the cross-section. The test speed was 1 mm/min and the test was conducted at 60% RH and 25°C. (Figure for illustration of appratus)

#### 2.3 building deep learning based semantic segmentation model

    The recored 30 fps video , one image per sec was extracted from the recored video from stereo-microscope. 

    The 12 original images with 256 pixels x 256 pixels were cropped from image sequence recorded by a stereo-microscope. The watershed segmentation was applied for label the boundary of wood cell wall. The unlabeled part was 

    Their corresponding ground truth masks with cell wall boundary labeled in white and background labeled in black were manually prepared. Finally, The 12 sets of original image and corresponding ground truth mask were used for building semantic segmentation model. And the asymmetric U-net architecture was used for the model training. (Figure for )

#### 2.4 image prediction and tracking of cell wall deformation

   After model training, the model with the patch blending algorithm implemented by Vooban were used to partition cell walls in the image sequence with 1920 pixels x 1080 pixels. After predicting image sequence, watershed segmentation was applied to achieve the instance segmentation of cell walls. Finally, a tracking algorithm (Crocker-Grier linking algorithm) was used to link the same cell walls exist in each image.

#### 2.5 parameters measurement for cell wall deformation analysis

    Scikit-image was used to measured the area, eccentricity, major/minor axis length, vertical/horizontal length of bounding box and maxium feret of each cell wall for analyzing their intensity of cell wall deformation. And their change rate during the bending test was also calucated based on the following equation:j

$$
Change = [(parameter_k[i] - parameter_k[0]) / parameter_k[0]] * 100
$$

---

## 3. results and discussion

#### 3.1 mechanical properties of flat-swan, quarter-swan and rift-swan



#### 3.2 Validation of U-net model and cell wall deformation tracking



#### 3.3 Visualization of cell wall deformation

#### 

#### 3.4 Relationship between changes in intensity of cell wall deformation and Stress-strain curve

---

## 4. Conclusion

---

## 5. Reference

---

## 6. Acknowledagement
