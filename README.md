# 3D-Image-Classification-of-Lung-CT-Based-on-PaddlePaddle

# Introduction
Construct a 3D Convolutional Neural Network (CNN) to predict the presence of viral pneumonia in computed tomography (CT) scans. 2D CNN is typically used to process RGB images (3 channels). 3D CNN is only a 3D equivalent: it requires input of 3D graphics or 2D frame sequences (such as slices in CT scans), and 3D CNN is a powerful model for learning and representing volume data.
# Motivation
* The binary classification problem for 3D-CT images
* High accuracy of prediction results
* Convenient environment construction
# Configuration and usage
* Environment 1: Paddle2.1 + CUDA 11.2 + 8G graphics memory 3070
* Environment 2: Paddle2.1 + CUDA 11.2 + 8G graphics memory 1080
# Notes
* Database：
1. Data Preprocessing: Using Functions ` def multiply_ Nrrd_ Files (image_folder, save_folder) ` Merge image and label into new input data. For data annotation, the dataset of CT-0 and CT-1, as well as the training and validation sets, are directly distinguished through Excel.
2. Data input: Import the package of pynrrd and redefine the threshold of CT scan specifications for nrrd data
3. Other processes such as normalization, adjusting width, height, and depth, and modifying corresponding functions based on the pynrrd package.
* Model：
  For ` epoch_ Num ',' batch '_ Size `, ` batch_ Size_ Valid ` for debugging;
  Using ` paddle.set_ Device 'calls local GPU for training.
# Framework
![图片](https://github.com/YaoOOoooolin/3D-Image-Classification-of-Lung-CT-Based-on-PaddlePaddle/assets/108522692/e96b01d4-deb3-4820-ba40-658ec375cf31)
