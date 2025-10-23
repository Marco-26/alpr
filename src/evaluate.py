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
  # METRICS
  #Detection (yolo) metrics
  detector_images_gt = 0
  detector_missed_images = 0
  detector_missed_images_labels = []
  detector_accuracy = 0

  #Extractor (ocr) metrics
  extractor_gt = 0
  extractor_letter_accuracy = 0
  extractor_accuracy = 0
  
  #Post processor metrics
  
  for index, image_path in enumerate(tqdm(validation_images_files, desc="Evaluating")):
    with Image.open(image_path) as img:
      detected_plates_result = detector.inference(img)
      
      if not detected_plates_result or len(detected_plates_result)==0:
        detector_missed_images += 1
        detector_missed_images_labels.append(image_path)
        continue
      
      detector_images_gt += 1
      
      for detected_plate in detected_plates_result:
        img = detected_plate.resize((200,100))
        gray = img.convert("L")
      
        result = extractor.inference(np.array(gray))
        for plate in result.plates:
          if plate != str(csv.label[index].strip()):
            plate = processor.validate(plate)
          
          if plate == csv.label[index]:
            extractor_gt +=1
            continue
          
  extractor_accuracy = (extractor_gt/length)*100 if length else 0.0
  detector_accuracy = (detector_images_gt/length)*100 if length else 0.0

  metrics = {
    "images": length,
    "detector": {
      "ground_truth": detector_images_gt,
      "missed": detector_missed_images,
      "missed_labels": detector_missed_images_labels,
      "accuracy": detector_accuracy,
    },
    "extractor": {
      "ground_truth": extractor_gt,
      "accuracy": extractor_accuracy,
    },
    "overall_accuracy": (detector_accuracy+extractor_accuracy)/2
  }

  return metrics

if __name__ == "__main__":
  metrics = evaluate()
  logging.info(
    (
      "METRICS SUMMARY\n"
      "- Total images evaluated: %d\n"
      "- Detector: correct detections (match ground truth): %d\n"
      "- Detector: missed images (no plate detected): %d\n"
      "- Detector: accuracy (%%): %.2f\n"
      "- Extractor: correct recognitions: %d\n"
      "- Extractor: accuracy (%%): %.2f\n"
      "- Overall accuracy (%%): %.2f" 
    ),
    metrics["images"],
    metrics["detector"]["ground_truth"],
    metrics["detector"]["missed"],
    metrics["detector"]["accuracy"],
    metrics["extractor"]["ground_truth"],
    metrics["extractor"]["accuracy"],
    metrics["overall_accuracy"],
  )
