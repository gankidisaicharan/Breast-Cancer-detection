# 🧠 Breast Cancer Detection using Deep Learning (CNN + Transfer Learning + Ensemble)

This project presents a robust deep learning pipeline to automatically detect breast cancer using ultrasound images. It leverages **Convolutional Neural Networks (CNNs)**, **Transfer Learning** with pre-trained models like VGG16, and **Ensemble Methods** to classify images into three categories: **Benign**, **Malignant**, or **Normal**. The model is deployed via a **Gradio web app** for interactive predictions.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Key Features](#key-features)
- [Performance](#performance)
- [Getting Started](#getting-started)
- [Project Files](#project-files)
- [Screenshots](#screenshots)
- [Future Work](#future-work)
- [License](#license)

---

## 🧐 Overview

Breast cancer is one of the most common types of cancer affecting women globally. Early detection significantly improves treatment outcomes. This project aims to assist radiologists by offering an **AI-powered solution** that can:

- Automatically analyze ultrasound breast scans
- Classify them into **normal**, **benign**, or **malignant**
- Deploy the solution in a **user-friendly web interface**

---

## 🧪 Tech Stack

| Tool               | Usage                           |
|--------------------|----------------------------------|
| Python             | Core programming language        |
| TensorFlow / Keras | Model building and training      |
| OpenCV             | Image preprocessing              |
| Scikit-learn       | Data handling & utilities        |
| Matplotlib         | Visualization                    |
| Gradio             | Web deployment of ML model       |
| Google Colab       | Development and training platform|

---

## 📊 Dataset

- **Source**: [Kaggle - Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Images**: 780 ultrasound images with mask annotations
- **Classes**:
  - Normal
  - Benign
  - Malignant
- **Image Format**: PNG (500x500 px average)
- **Augmentation**: Rotation, shift, shear, brightness, flips (horizontal/vertical)

---

## 🧠 Model Architecture

### ✅ Custom CNN
- 7 convolutional layers with ReLU and MaxPooling
- Dense layers with dropout to reduce overfitting
- Trained from scratch on the dataset

### ✅ Transfer Learning (VGG16)
- Pretrained on ImageNet
- Top layer removed, custom dense layers added
- Fine-tuned on the ultrasound images

### ✅ Ensemble Learning
- Multiple CNNs trained with different seeds
- Final prediction obtained via averaging (bagging)

---

## 🎯 Key Features

- 🚀 High accuracy model using **CNN + Transfer Learning**
- 🧪 Evaluation using **validation accuracy**, **loss curves**
- 🧰 Integrated **data augmentation**
- 🧠 Ensemble for boosting generalization
- 🌐 Live web app with **Gradio** for interactive diagnosis
- 💡 Focus on explainability, early detection & clinical impact

---

## 📈 Performance

| Model               | Validation Accuracy |
|--------------------|---------------------|
| CNN (custom)        | ~88%                |
| VGG16 Transfer Learning | ~91.5%            |
| Ensemble CNN        | **93%+**             |

(Note: Results may vary based on dataset split and training parameters.)

---

