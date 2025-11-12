# ALPR — Simple Automatic License Plate Recognition

A compact Automatic License Plate Recognition (ALPR) system that detects, reads, and cleans license plate text from images. Designed for clarity and experimentation, it demonstrates how detection, OCR, and post-processing connect into a full pipeline.

## Overview
The system processes an image in three stages:
1. **Detection**: A YOLO model localizes the license plate.
2. **OCR**: The cropped plate is read using fast_plate_ocr (European plates model).
3. **Post-Processing**: The recognized text is normalized and corrected using simple, rule‑based substitutions tuned for Portuguese formats.

## Setup

Requirements
- Python 3.8+
- A virtual environment (recommended)
- Key Python packages: gradio, pillow, numpy
- Additional packages required by the detector/extractor (e.g. ultralytics, torch) — see `requirements.txt` for the project's full list.

1. Create and activate a virtual environment (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

If you already have `requirements.txt`:

```bash
pip install -r requirements.txt
```

Note: Some packages (like `torch`) have platform/GPU-specific wheels — follow their official install instructions if you need CUDA support.

## Run the demo

From the repository root run:

```bash
python src/interface.py
```

This will launch a Gradio web UI (a local URL will be printed). In the UI you can upload an image and the app will show:
- Original image
- Cropped plate image (detected by YOLO)
- OCR result (normalized text)

- Output components are positional. The return value of the `get_plate_flow` function must match the `outputs` components in order and type.

## Troubleshooting
- "No license plates were detected": ensure the detector model file path is correct and the input image contains a clear plate. Check logs for errors loading the model.
- Empty images in the web UI: verify the type and contents of the `file` object passed to the processing function (see debug print above). You can force `gr.File(type="filepath")` to always get a filesystem path.
- GPU issues: make sure CUDA drivers are installed and your environment has the correct `torch`/CUDA build.


## License & Contact
Pick a license for the project (e.g. MIT) and add contact or contribution instructions here.