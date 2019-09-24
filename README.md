# Perception-sytems

# CNN-LIDAR pedestrian classification: combining range and reflectance data

The use of multiple sensors in perception systems is becoming a consensus in the automotive and robotics industries. Camera is the most popular technology, however, radar and LIDAR are increasingly being adopted more often in protection and safety systems for object/obstacle detection. In this paper, we particularly explore the LIDAR sensor as an inter-modality technology which provides two types of data, range (distance) and reflectance (intensity return), and study the influence of high-resolution distance/depth (DM) and reflectance maps (RM) on pedestrian classification using a deep Convolutional Neural Network (CNN). Pedestrian protection is critical for advanced driver assistance system (ADAS) and autonomous driving, and it has regained particular attention recently for known reasons. In this work, CNN-LIDAR based pedestrian classification is studied in three distinct cases: (i) having a single modality as input in the CNN, (ii) by combining distance and reflectance measurements at the CNN input-level (early fusion), and (iii) combining outputs scores from two single-modal CNNs (late fusion). Distance and intensity (reflectance) raw data from LIDAR are transformed to high-resolution (dense) maps which allow a direct implementation on CNNs both as single or multi-channel inputs (early fusion approach). In terms of late-fusion, the outputs from individual CNNs are combined by means of non-learning rules, such as: minimum, maximum, average, product. Pedestrian classification is evaluated on a ‘binary classification’ dataset created from the KITTI Vision Benchmark Suite, and results are shown for the three cases.



![Bilateral filter](https://github.com/gledsonmelotti/Perception-sytems/blob/master/Images/000000.png)


