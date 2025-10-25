# ALPR — Simple Automatic License Plate Recognition

This repository contains a small, learning‑oriented ALPR system that detects a license plate in an image, reads its text with OCR, and then cleans the result using lightweight post‑processing rules tuned for Portuguese formats. I built it for fun to understand how the detection and OCR pieces fit together into an end‑to‑end pipeline.

The flow is simple. An Ultralytics YOLO model first localizes the plate in the input image and crops the detected region. Those crops are passed to an OCR component backed by `fast_plate_ocr` using its European plates model, which returns one or more candidate strings. The candidates are normalized to uppercase alphanumeric text and then refined by a small post‑processor that corrects common confusions such as O↔0, I↔1, and S↔5 while preferring strings that match Portuguese plate patterns. Together, this gives a compact example of detection → OCR → post‑processing, and the repository includes an evaluation script that reports basic metrics over a small validation set.

The code is intentionally compact and easy to read. The CLI entry point in `src/main.py` runs single‑image inference. The detector in `src/detect.py` wraps the YOLO model (with a configurable confidence threshold at `src/detect.py:17`) and includes a convenience method to fine‑tune a YOLO variation if you want to experiment. The OCR wrapper and normalization live in `src/extract.py`, which uses the model named in code ("european-plates-mobile-vit-v2-model" at `src/extract.py:17`). The Portuguese‑specific validator and position‑aware substitutions are in `src/post_processor.py`. Configuration such as the weights path, dataset locations, patterns, and confusion pairs is centralized in `src/constants.py`; by default the detector reads weights from `outputs/runs/detect/train/weights/best.pt` as referenced at `src/constants.py:5`. Sample images and labels are provided under `validation_images/` to make experiments easy.

To get started, install Python 3.10 or newer and set up a virtual environment. On Unix‑like systems, create and activate a venv with:

```
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate with:

```
.\.venv\Scripts\activate
```

With the environment active, install dependencies using:

```
pip install -r requirements.txt
```

The first YOLO run may download additional assets. If your environment struggles with the extras syntax for the OCR dependency in `requirements.txt`, you can install it directly with:

```
pip install fast-plate-ocr[onnx]
```

After installation, you can run single‑image inference by executing:

```
python src/main.py path/to/image.jpg
```

This will open the image, detect and crop the plate, run OCR, apply post‑processing, and log the candidate plate texts it finds. To evaluate the full pipeline on the included validation set, run:

```
python src/evaluate.py
```

The evaluation prints a short summary that includes detector recall, OCR accuracy, how often the post‑processor corrected the OCR to a match, and an overall end‑to‑end score. If you want to fine‑tune a detector, the project points to the included weights at `outputs/runs/detect/train/weights/best.pt` and exposes a minimal training helper via `Detector.finetune(...)` in `src/detect.py`; alternatively, you can use the Ultralytics CLI (`yolo detect train ...`) with your own data configuration.

Configuration is centralized so tweaks are easy: change the YOLO weights path in `src/constants.py`, adjust the detection threshold in `src/detect.py`, and extend or modify the Portuguese plate patterns and confusion pairs in `src/constants.py` if your data differs. The current approach has some limitations worth noting. The post‑processing logic is tuned for Portuguese formats and may not generalize to other countries without changes. The OCR model is European‑centric, so plates from other regions may require a different model. The interface targets single images from the command line rather than video streams or a real‑time UI.

This project relies on Ultralytics YOLO for detection and on `fast_plate_ocr` for OCR. No license file is currently included; if you plan to reuse or distribute the code, consider adding one (for example, MIT or Apache‑2.0) that matches your needs.

