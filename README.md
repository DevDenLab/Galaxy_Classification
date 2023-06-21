# Galaxy Image Classification Project

This project aims to classify galaxy images into different classes using various deep learning models. The goal is to achieve high accuracy in predicting the correct class for a given galaxy image.

## Project Overview

Galaxy classification is an important task in astronomy as it helps astronomers understand the different types and properties of galaxies. Traditional classification methods require manual analysis by experts, which can be time-consuming and subjective. By using deep learning techniques, we can automate the process and potentially improve the accuracy of classification.

The purpose of this project is to develop and compare different deep learning models for galaxy image classification. By training models on a labeled dataset of galaxy images, we can evaluate their performance and identify the most effective model for this task.

## Dataset

The dataset used in this project consists of a collection of galaxy images. Each image is labeled with a corresponding class, indicating the type of galaxy it represents. The dataset is divided into three subsets: training, validation, and testing.

The dataset is located in the `data/` directory 
<!-- and is organized as follows:

- `data/train/`: Contains the training images.
- `data/validation/`: Contains the validation images.
- `data/test/`: Contains the testing images. -->

<!-- The dataset is balanced, meaning that it contains an equal number of images for each galaxy class. This ensures that the models are trained and evaluated on a representative distribution of classes. -->
## Model Performace 
![Alt Text](path/to/perform_table.jpg)

## Models Used

This project utilizes several deep learning models for galaxy image classification:

1. CNN_Model-2.1:
   - Architecture: Custom CNN model with multiple convolutional and pooling layers.
   - Training Loss: 0.0396
   - Training Accuracy: 0.9862
   - Validation Loss: 0.3149
   - Validation Accuracy: 0.9156
   - Test Loss: 0.0912
   - Test Accuracy: 0.9730

2. Inception_V3:
   - Architecture: Pretrained InceptionV3 model with weights initialized from ImageNet.
   - Training Loss: 0.0298
   - Training Accuracy: 0.9899
   - Validation Loss: 0.2171
   - Validation Accuracy: 0.9534
   - Test Loss: 0.7215
   - Test Accuracy: 0.8210

3. VGG16:
   - Architecture: Pretrained VGG16 model with weights initialized from ImageNet.
   - Training Loss: 0.0324
   - Training Accuracy: 0.9884
   - Validation Loss: 1.1052
   - Validation Accuracy: 0.8306
   - Test Loss: 0.3729
   - Test Accuracy: 0.9320

4. ResNet50:
   - Architecture: Pretrained ResNet50 model with weights initialized from ImageNet.
   - Training Loss: 0.0745
   - Training Accuracy: 0.9731
   - Validation Loss: 0.1239
   - Validation Accuracy: 0.9578
   - Test Loss: 0.0900
   - Test Accuracy: 0.9650

5. Xception:
   - Architecture: Pretrained Xception model with weights initialized from ImageNet.
   - Training Loss: 0.2763
   - Training Accuracy: 0.8885
   - Validation Loss: 0.1924
   - Validation Accuracy: 0.9369
   - Test Loss: 0.2838
   - Test Accuracy: 0.8910

## Getting Started

To run this project on your local machine, follow these steps:

1. Clone the repository: `git clone https://github.com/your_username/galaxy-classification.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the dataset and place it in the `data/` directory.
<!-- 4. Run the desired model script, such as `python cnn_model.py` or `python inception_v3.py`.
 -->
Feel free to explore and modify the code to suit your needs. Happy classifying!

