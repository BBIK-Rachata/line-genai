# Training Model Guide

## Overview
This guide explains how to train a model in `.pt` and `.onnx` formats using datasets from Roboflow and training via Ultralytics Hub.

## Prerequisites
- Roboflow account to access datasets
- Ultralytics Hub account for training
- Colab notebook
- Internet connection

## Dataset Sources
We use datasets from Roboflow, accessible via the following links:

Dataset 1: Classification between Car and Motorcycle
(https://universe.roboflow.com/project-tdxxb/cctv_car_bike_detection-fhqk8)

Dataset 2: Car Brand Detection
(https://universe.roboflow.com/abc-pjocx/car-models-ves3u/dataset/2)

Dataset 3: License Plate Detection
(https://universe.roboflow.com/au-parking-ffplf/license-plate-eql7j)

Dataset 4: Thai License Plate Character Recognition
(https://universe.roboflow.com/meenyossakorn/thai-license-plate-character-recognition)

Dataset 5: Car Damage Detection
(https://universe.roboflow.com/gp-gja5p/gp-car-damage)

## Training Process
### 1. Download Dataset from Roboflow
1. Visit the dataset links provided.
2. Export the dataset in YOLO11 format.
3. Download the dataset as a ZIP file.

### 2. Upload Dataset to Ultralytics Hub
1. Go to Ultralytics Hub
  (https://ultralytics.com/hub).
2. Create a new project and upload the dataset.
3. Configure training settings such as model type (YOLOv11, etc.), epochs, and batch size.

### 3. Train the Model
1. Start the training process via Ultralytics Hub using Colab notebook.
2. After training completes, download the trained model in `.pt` and `.onnx` formats.

## Exporting the Model
- `.pt` format: Suitable for PyTorch-based applications.
- `.onnx` format: Ideal for deployment in various environments including mobile and cloud.
