# Vehicle License Plate Recognition and Information Retrieval

A project to detect and extract vehicle license plate information from images or real-time camera feeds, providing details such as the vehicle's brand, model, and specifications using an API. This project showcases computer vision and AI skills, including object detection, OCR.

# Overview

Goal: Develop a system to detect license plates in images or live camera feeds, extract the plate text using OCR, and retrieve vehicle details (e.g., brand, model, engine specs) via an API, with real-time inference capabilities on mobile devices.
Use Case: Enable users to point a phone camera at a vehicle's license plate and instantly view its specifications.

Tools:
Python: Core programming language.
YOLOv11: For license plate detection.
EasyOCR: For text extraction from cropped plates.
API: External API for retrieving vehicle details (e.g., brand, model, motor).


# Features

License Plate Detection: Accurately localizes license plates in images using YOLOv11.
Text Extraction: Extracts text from detected plates with EasyOCR, optimized for license plate fonts.
Vehicle Information Retrieval: Queries an external API to fetch vehicle details based on the extracted plate number.

Real-Time Inference (Future Work): Optimized for mobile devices to enable live camera-based recognition.

# Roadmap
1. Dataset Collection & Preparation

Source images from open datasets.
Annotate license plates with bounding boxes or use pre-annotated datasets.
Apply data augmentation (brightness, blur, rotation, angle distortion) to improve robustness.

2. License Plate Detection

Use YOLOv11 for real-time license plate localization.
Fine-tune on annotated datasets to improve detection accuracy.
Output cropped plate regions for OCR processing.

3. OCR Recognition

Feed cropped plates into EasyOCR for text extraction.
Fine-tune OCR model on license plate-specific fonts and formats.
Output raw text string of the license plate.

4. Evaluation & Deployment

Metrics:
Detection Accuracy (IoU).
OCR Accuracy.
End-to-End ALPR Accuracy.


# Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs, feature requests, or improvements.
License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments

YOLOv11: Ultralytics for the object detection framework.
EasyOCR: For robust OCR capabilities.
