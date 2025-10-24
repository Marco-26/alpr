"""Evaluate detector + OCR + post-processor on validation data."""

from post_processor import PostProcessor
from extract import Extractor
from detect import Detector
from tqdm import tqdm
from PIL import Image
import glob
import pandas as pd
import numpy as np
import os
import logging
import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

extractor = Extractor()
detector = Detector()
processor = PostProcessor()

# Evaluate the performance of model flow (extractor & detector) to performance of the processor.
# To evaluate the extractor also measure the confidence so we can see how the confidence correlates with accuracy.

validation_images_files = sorted(glob.glob(f'{constants.VALIDATION_IMAGES_PATH}/*.jpg'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
csv = pd.read_csv(constants.VALIDATION_IMAGES_LABELS_PATH)

length = len(validation_images_files)

def evaluate() -> dict:
  """Run evaluation and return a metrics dictionary."""
  # METRICS
  #Detection (yolo) metrics
  detector_images_gt = 0
  detector_missed_images = 0
  detector_missed_images_labels = []
  detector_accuracy = 0

  #Extractor (ocr) metrics
  extractor_gt = 0
  extractor_accuracy = 0
  extractor_post_processor_help = 0 # how many plates post processor helped get right
  
  for index, image_path in enumerate(tqdm(validation_images_files, desc="Evaluating")):
    with Image.open(image_path) as img:
      detected_plates_result = detector.inference(img)
      
      if not detected_plates_result or len(detected_plates_result) == 0:
        detector_missed_images += 1
        detector_missed_images_labels.append(image_path)
        continue
      
      detector_images_gt += 1
      
      for detected_plate in detected_plates_result:
        img = detected_plate.resize((500,100))
        gray = img.convert("L")
      
        result = extractor.inference(np.array(gray))
        for plate in result.plates:
          if plate != str(csv.label[index].strip()):
            refactored_plate = processor.validate(plate)
            if refactored_plate == csv.label[index]:
              extractor_post_processor_help += 1
          
          if plate == csv.label[index] or refactored_plate == csv.label[index]:
            extractor_gt += 1
            continue
          
  extractor_accuracy = (extractor_gt/length)*100 if length else 0.0
  detector_accuracy = (detector_images_gt/length)*100 if length else 0.0

  metrics = {
    "images": length,
    "detector": {
      "correct_detections": detector_images_gt,
      "missed": detector_missed_images,
      "missed_labels": detector_missed_images_labels,
      "recall": detector_accuracy,
    },
    "extractor": {
      "exact_matches": extractor_gt,
      "accuracy": extractor_accuracy,
      "fixed_by_post_processor": extractor_post_processor_help
    },
    "end_to_end_accuracy": (extractor_accuracy+detector_accuracy)/2
  }

  return metrics

if __name__ == "__main__":
  metrics = evaluate()
  logging.info(
    (
      "METRICS SUMMARY\n"
      "- Total images evaluated: %d\n"
      "- Detector: images with at least one detection: %d\n"
      "- Detector: missed images (no plate detected): %d\n"
      "- Detector: recall: %.2f%%\n"
      "- Extractor: exact matches: %d\n"
      "- Extractor: accuracy: %.2f%%\n"
      "- Extractor: fixed by post processor: %d\n"
      "- End-to-end accuracy: %.2f%%\n"
    ),
    metrics["images"],
    metrics["detector"]["correct_detections"],
    metrics["detector"]["missed"],
    metrics["detector"]["recall"],
    metrics["extractor"]["exact_matches"],
    metrics["extractor"]["accuracy"],
    metrics["extractor"]["fixed_by_post_processor"],
    metrics["end_to_end_accuracy"],
  )
