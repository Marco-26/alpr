#Vehicle License Plate Recognition and Information Retrieval

A project to detect and extract vehicle license plate information from images or real-time camera feeds, providing details such as the vehicle's brand, model, and specifications using an API. This project showcases computer vision and AI skills, including object detection, OCR, and real-time inference on mobile devices.
Overview

Goal: Develop a system to detect license plates in images or live camera feeds, extract the plate text using OCR, and retrieve vehicle details (e.g., brand, model, engine specs) via an API, with real-time inference capabilities on mobile devices.
Use Case: Enable users to point a phone camera at a vehicle's license plate and instantly view its specifications.
Tools:
Python: Core programming language.
YOLOv8: For license plate detection.
EasyOCR: For text extraction from cropped plates.
OpenCV: For image preprocessing.Ascendancy
API: External API for retrieving vehicle details (e.g., brand, model, motor).


Dataset:
Car License Plate Detection Dataset (Dataset Ninja, 433 images, CC0 1.0) or UFPR-ALPR (1,500 images, academic use).


Timeline: 1-2 weeks (~10-15 hours part-time).
Deliverables:
GitHub repository with source code.
Comprehensive README.
Sample output images.
Demo video showcasing real-time inference.

# Features

License Plate Detection: Accurately localizes license plates in images or video frames using YOLOv8.
Text Extraction: Extracts text from detected plates with EasyOCR, optimized for license plate fonts.
Vehicle Information Retrieval: Queries an external API to fetch vehicle details based on the extracted plate number.
Real-Time Inference: Optimized for mobile devices to enable live camera-based recognition.
Post-Processing: Applies regex and country-specific rules to correct OCR errors (e.g., O vs. 0, B vs. 8) and ensure reliable output.

# Installation

Clone the repository:git clone https://github.com/your-username/vehicle-license-plate-recognition.git
cd vehicle-license-plate-recognition


Install dependencies:pip install -r requirements.txt


Download pre-trained YOLOv8 weights and EasyOCR models (links provided in models/ directory).
(Optional) Set up an API key for vehicle information retrieval (see config/ for instructions).

# Usage

Run on a single image:
python main.py --image path/to/image.jpg

Outputs the detected license plate text and vehicle details in the console and saves cropped plate images to outputs/.

Real-time camera inference:
python main.py --camera

Opens the default camera feed for live license plate detection and vehicle info display.

Sample outputs:

View sample results in the outputs/ directory.
Watch the demo video (demo.mp4) for a live example.



# Project Structure
vehicle-license-plate-recognition/
├── config/                 # API keys and configuration files
├── data/                   # Datasets and annotations
├── models/                 # Pre-trained YOLOv8 and OCR models
├── outputs/                # Cropped plates and sample results
├── src/                    # Source code
│   ├── detection.py        # YOLOv8 license plate detection
│   ├── ocr.py              # EasyOCR text extraction
│   ├── postprocess.py      # Text validation and correction
│   ├── vehicle_info.py     # API integration for vehicle details
│   └── main.py             # Main script for inference
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── demo.mp4                # Demo video

# Roadmap
1. Dataset Collection & Preparation

Source images from open datasets (e.g., Dataset Ninja, UFPR-ALPR).
Annotate license plates with bounding boxes or use pre-annotated datasets.
Apply data augmentation (brightness, blur, rotation, angle distortion) to improve robustness.

2. License Plate Detection

Use YOLOv8 for real-time license plate localization.
Fine-tune on annotated datasets to improve detection accuracy.
Output cropped plate regions for OCR processing.

3. OCR Recognition

Feed cropped plates into EasyOCR for text extraction.
Fine-tune OCR model on license plate-specific fonts and formats.
Output raw text string of the license plate.

4. Post-Processing & Validation

Apply regex and country-specific rules (e.g., XX-00-YY format for Portugal).
Correct common OCR errors (e.g., O vs. 0, B vs. 8).
Implement confidence scoring to reject low-quality reads.

5. Evaluation & Deployment

Metrics:
Detection Accuracy (IoU).
OCR Accuracy.
End-to-End ALPR Accuracy.


Optimize for real-time performance using ONNX, TensorRT, or quantization.
Deploy on edge devices (e.g., mobile phones, Raspberry Pi, Jetson) or as a server API.

# Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs, feature requests, or improvements.
License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments

YOLOv8: Ultralytics for the object detection framework.
EasyOCR: For robust OCR capabilities.
Datasets: Dataset Ninja and UFPR-ALPR for providing open license plate datasets.
