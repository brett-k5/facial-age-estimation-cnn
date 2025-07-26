🧑‍💻 Facial Age Estimation CNN
Welcome to the Facial Age Estimation project! This repo contains a convolutional neural network (CNN) model designed to predict the age of a person from facial images. 🎉

🚀 Project Overview
This model leverages a ResNet50 backbone as the first layer for feature extraction, followed by:

A convolutional layer

Global Average Pooling to reduce spatial dimensions

A fully connected dense layer with a single output neuron using ReLU activation to predict age

We experimented with regularization techniques like Dropout to prevent overfitting, but these did not improve the model's performance significantly.

📊 Performance
Mean Absolute Error (MAE) on test set: 8.5 years

This metric reflects the average absolute difference between predicted and true ages.

⚙️ Repository Structure
Everything lives in the root directory:

```
├── data/                  # (Optional) Folder for datasets
├── model.py               # CNN model architecture definition
├── train.py               # Training script
├── requirements.txt       # Required Python packages with pinned versions
├── README.md              # This file
└── notebooks/             # Jupyter notebooks for experiments & visualization
