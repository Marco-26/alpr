# ALPR — Simple Automatic License Plate Recognition

A small, learning‑oriented ALPR system that detects a license plate in an image, reads its text with OCR, and applies light post‑processing tailored to Portuguese plate formats. It was built for fun to understand the end‑to‑end pipeline.

**What it does**

- Detects plates with an Ultralytics YOLO model (weights included).
- Runs OCR using `fast_plate_ocr` (European plate model).
- Normalizes and post‑processes candidates to correct common confusions (O/0, I/1, S/5) and enforce Portuguese formats.
- Provides a simple CLI for single‑image inference and a script to evaluate on a small validation set.

**Why it’s useful**

- Clear, minimal code you can read and tweak.
- End‑to‑end example: detection → OCR → post‑processing → metrics.

**Project Structure**

- `src/main.py`: CLI for detecting and OCR’ing a single image.
- `src/detect.py`: YOLO detector and optional fine‑tuning helper.
- `src/extract.py`: OCR wrapper and normalization using `fast_plate_ocr`.
- `src/post_processor.py`: Portuguese‑specific validation and character substitutions.
- `src/constants.py`: Config paths and patterns (YOLO weights, dataset, regex).
- `validation_images/`: Sample images + `validation_images_labels.csv` for evaluation.
- `outputs/runs/detect/train/weights/best.pt`: Included trained weights.

**Prerequisites**

- Python 3.10+
- macOS/Linux/Windows with a working Python toolchain
- Optional GPU for faster YOLO inference (CPU works fine)

**Setup**

- Create a virtual environment (recommended) and install deps:
  - `python -m venv .venv && source .venv/bin/activate` (Windows: `.\.venv\\Scripts\\activate`)
  - `pip install -r requirements.txt`

Note: First YOLO run may download additional assets. If you run into issues with the quoted extras in `requirements.txt`, install OCR directly: `pip install fast-plate-ocr[onnx]`.

**Run Inference (Single Image)**

- `python src/main.py path/to/image.jpg`

The script logs candidate plate texts after detection and OCR.

**Evaluate on Validation Set**

- Place images and labels under `validation_images/` (already included as an example).
- Run: `python src/evaluate.py`
- Outputs a metrics summary (detector recall, OCR accuracy, post‑processor fixes, end‑to‑end accuracy).

**Fine‑Tuning (Optional)**

- Trained weights live at `outputs/runs/detect/train/weights/best.pt` and are referenced by default.
- To fine‑tune a YOLO variant via code, see `Detector.finetune(...)` in `src/detect.py`.
- Alternatively, use the Ultralytics CLI directly (`yolo detect train ...`).

**Configuration**

- Change YOLO weights path: `src/constants.py:5`
- Adjust detection threshold: `src/detect.py:17`
- Modify Portuguese plate patterns/confusions: `src/constants.py:10`

**Limitations**

- Post‑processing currently targets Portuguese formats and may not fit other countries.
- OCR model is European‑centric; other regions may need a different model.
- This is a simple, single‑image CLI — no streaming/video UI.

**Acknowledgments**

- Ultralytics YOLO for detection.
- `fast_plate_ocr` for the OCR model and runtime.
