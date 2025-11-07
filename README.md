# 🚀 Drone Detection with YOLO11

Welcome to the Drone Detection project using YOLO11! This project aims to detect objects such as Airplanes, Drones, and Helicopters in images. We use a YOLO11 model trained on a NVIDIA GeForce RTX 4060.

## 📋 Table of Contents
- [Project Description](#-project-description)
- [Files](#-files)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)

## 📝 Project Description
This project focuses on detecting various types of aerial vehicles (Airplanes, Drones, Helicopters) using the YOLO11 model. The dataset used for training and testing can be found [here](https://www.kaggle.com/datasets/cybersimar08/drone-detection/data).

## 📂 Files

### 🔧 Configuration
- **configuration.py**: This file contains all the necessary configurations for the project, including model parameters, paths to data files, and other settings required for the smooth operation of the project.

### 📊 Dataset Preparation
- **make_dataset.py**: This script is responsible for preparing and generating the dataset for training and testing. It includes functions to load, preprocess, and split the dataset into training and testing sets.

### 🔄 Training
- **train_model.py**: This script is used to train the YOLO11 model on the prepared dataset. It includes functions to load the dataset, define the model architecture, train the model, and save the trained model to disk.

### 📈 Analysis
- **analyze_results.py**: This script is used to analyze the results of the model. It includes functions to load the trained model, make predictions on the test set, and calculate metrics such as precision, recall, and dice score.

### 🛠️ Utilities
- **tools.py**: This file contains utility functions that are used across different scripts. These functions include data loading, data preprocessing, model evaluation, and visualization functions.

### 🔮 Prediction
- **predict.py**: This script is used to make predictions on new images using the trained YOLO11 model. It includes functions to load the trained model, preprocess new images, make predictions, and visualize the results.

## 🛠️ Installation
To get started with this project, follow these steps:

1. **Clone the Repository:**
   Clone the repository to your local machine using the following command:
   ```
   git clone https://github.com/BatBapt/aerial_detection
   ```

2. **Install Dependencies:**
   Install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/cybersimar08/drone-detection/data) and place it in the appropriate directory within the project.

4. **Set Up Configuration:**
   Update the `configuration.py` file with the appropriate paths to the dataset and any other necessary configurations.

## 🚀 Usage

1. **Prepare the Dataset:**
   Run the `make_dataset.py` script to prepare the dataset for training and testing:
   ```
   python make_dataset.py
   ```

2. **Train the Model:**
   Use the `train_model.py` script to train the YOLO11 model on the prepared dataset:
   ```
   python train_model.py
   ```
   
3. **Make Predictions:**
   Use the `predict.py` script to make predictions on validation dataset:
   ```
   python predict.py
   ```

4. **Analyze Results:**
   After training, use the `analyze_results.py` script to evaluate the model's performance:
   ```
   python analyze_results.py
   ```

## 📊 Training plots
Here you will find training plots such as loss curves, precision-recall curves, and other relevant visualizations generated during the training process.

### Training Results for the Best Model
![](runs/detect/bs8_100/results.png "Training results for the best model")

### Training Results for the BS4_100 Model
![](runs/detect/bs4_100/results.png "Training results for the BS4_100 model")

### Training Results for the BS4_200 Model
![](runs/detect/bs4_200/results.png "Training results for the BS4_200 model")

## 📊 Results
Here you will find the results of the model training and evaluation. The user will add plots for training time, precision, recall, dice score, etc.

| Metric       | Value  |
|--------------|--------|
| Precision    | -      |
| Recall       | -      |
| Dice Score   | -      |
| Training Time| -      |


---

Stay tuned for more updates and happy detecting! 🚁✈️🛩️
