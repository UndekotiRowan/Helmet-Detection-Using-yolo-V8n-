# Helmet and Face Detection using YOLOv8
This repository contains a computer vision project capable of detecting human faces and helmets in images and video streams. The project utilizes the Ultralytics YOLOv8 architecture and is fine-tuned on a custom dataset hosted by Roboflow.

📌 Project Overview
Ensuring safety compliance (such as wearing helmets) is critical in construction, mining, and traffic management. This model is trained to identify two specific classes:

Human-face

Helmet

It leverages transfer learning using the YOLOv8-Nano (yolov8n) model for efficient and fast detection.

🛠️ Installation & Requirements
To run this project, you will need Python installed along with the following libraries:

Bash

pip install ultralytics roboflow
If you wish to run inference on local webcam feeds or process images using OpenCV, ensure you also have:

Bash

pip install opencv-python-headless numpy
📂 Dataset
The dataset used for training is acquired from the Roboflow Universe.

Workspace: infernal-3whye

Project: full_dataset-vwxcb

Version: 1

The notebook automatically handles downloading and formatting the dataset for YOLOv8.

🚀 Training
The model was trained using the following configuration:

Base Model: YOLOv8 Nano (yolov8n.pt)

Epochs: 50

Image Size: 640x640

Classes: 2 (Human-face, helmet)

Training Script
You can reproduce the training process with the following python script:

Python

from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
   data='/path/to/dataset/data.yaml',
   epochs=50,
   imgsz=640,
   name='face_helmet_detector'
)
💻 Usage
1. Load Trained Weights
After training, the best weights are saved in the runs directory (e.g., runs/detect/face_helmet_detector/weights/best.pt).

2. Run Inference on Images
Python

from ultralytics import YOLO
import cv2

# Load your custom model
model = YOLO('path/to/best.pt')

# Run prediction
image_path = 'path/to/your/image.jpg'
results = model.predict(image_path)

# Display results
result_img = results[0].plot()
cv2.imshow("Detection", result_img)
cv2.waitKey(0)
3. Webcam Inference
The repository includes logic to capture images from a webcam (via JavaScript for Colab/Jupyter environments) and run detections on them.

📊 Results
During training, the model achieved high accuracy metrics (around epoch 29):

mAP@50: ~0.94

Precision (Box): ~0.90

Recall (Box): ~0.89

🤖 Technologies Used
YOLOv8 by Ultralytics - State-of-the-art object detection.

Roboflow - Dataset management and preprocessing.

OpenCV - Image processing.

Google Colab - Training environment with GPU acceleration (Tesla T4).

📜 License
MIT
