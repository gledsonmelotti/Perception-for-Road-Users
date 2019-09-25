# Perception-Sytems

# Convolutional Neural Networks (CNN)

![CNN](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/AlexNet.PNG)

# Dataset created from the KITTI Vision Benchmark Suite.

3D Point Clouds (Frame 65-KITTI Vision Benchmark Suite):
![Point Clouds](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/000065_pointclouds.png)

2D RGB image (Frame 65-KITTI Vision Benchmark Suite):

![Frame 65](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/000065.png)


Projected 3D Point clouds in the 2D image-plane (Frame 65-KITTI Vision Benchmark Suite):

![Frame 65 projected](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/000065_projected.png)

Depth-range map (Frame 65): average, bilateral filter, IDW, maximum, minimum using sliding-window (a mask) 13x13 in size for upsample the projected 3D point clouds in the 2D image-plan. 

![AVG](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/DepthMap/000065_AVG.png)
![BF](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/DepthMap/000065_BF.png)
![IDW](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/DepthMap/000065_IDW.png)
![MAX](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/DepthMap/000065_MAX.png)
![MIN](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/DepthMap/000065_MIN.png)

Reflectance map (Frame 65): average, bilateral filter, IDW, maximum, minimum using sliding-window (a mask) 13x13 in size for upsample the projected 3D point clouds in the 2D image-plan. 

![AVG](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/ReflectanceMaP/000065_AVG.png)
![BF](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/ReflectanceMaP/000065_BF.png)
![IDW](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/ReflectanceMaP/000065_IDW.png)
![MAX](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/ReflectanceMaP/000065_MAX.png)
![MIN](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/ReflectanceMaP/000065_MIN.png)


Dataset with RGB image and Depth/Range Map using bilateral filter (13x13 in size): car, pedestrian, cyclist and person sitting.

![Dataset](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/dataset.png)

# CNN-LIDAR pedestrian classification: combining range and reflectance data

Link: https://ieeexplore.ieee.org/document/8519497

The use of multiple sensors in perception systems is becoming a consensus in the automotive and robotics industries. Camera is the most popular technology, however, radar and LIDAR are increasingly being adopted more often in protection and safety systems for object/obstacle detection. In this paper, we particularly explore the LIDAR sensor as an inter-modality technology which provides two types of data, range (distance) and reflectance (intensity return), and study the influence of high-resolution distance/depth (DM) and reflectance maps (RM) on pedestrian classification using a deep Convolutional Neural Network (CNN). Pedestrian protection is critical for advanced driver assistance system (ADAS) and autonomous driving, and it has regained particular attention recently for known reasons. In this work, CNN-LIDAR based pedestrian classification is studied in three distinct cases: (i) having a single modality as input in the CNN, (ii) by combining distance and reflectance measurements at the CNN input-level (early fusion), and (iii) combining outputs scores from two single-modal CNNs (late fusion). Distance and intensity (reflectance) raw data from LIDAR are transformed to high-resolution (dense) maps which allow a direct implementation on CNNs both as single or multi-channel inputs (early fusion approach). In terms of late-fusion, the outputs from individual CNNs are combined by means of non-learning rules, such as: minimum, maximum, average, product. Pedestrian classification is evaluated on a 'binary classification' dataset created from the KITTI Vision Benchmark Suite, and results are shown for the three cases.


# Multimodal CNN Pedestrian Classification: A Study on Combining LIDAR and Camera Data

Link: https://ieeexplore.ieee.org/document/8569666

This paper presents a study on pedestrian classification based on deep learning using data from a monocular camera and a 3D LIDAR sensor, separately and in combination. Early and late multi-modal sensor fusion approaches are revisited and compared in terms of classification performance. The problem of pedestrian classification finds applications in advanced driver assistance system (ADAS) and autonomous driving, and it has regained particular attention recently because, among other reasons, safety involving self-driving vehicles. Convolutional Neural Networks (CNN) is used in this work as classifier in distinct situations: having a single sensor data as input, and by combining data from both sensors in the CNN input layer. Range (distance) and intensity (reflectance) data from LIDAR are considered as separate channels, where data from the LIDAR sensor is feed to the CNN in the form of dense maps, as the result of sensor coordinate transformation and spatial filtering; this allows a direct implementation of the same CNN-based approach on both sensors data. In terms of late-fusion, the outputs from individual CNNs are combined by means of learning and non-learning approaches. Pedestrian classification is evaluated on a 'binary classification' dataset created from the KITTI Vision Benchmark Suite, and results are shown for each sensor-modality individually, and for the fusion strategies.



