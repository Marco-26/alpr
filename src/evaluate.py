from post_processor import PostProcessor
from extract import Extractor
from detect import Detector
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

# METRICS
right_choices = []
accuracy = 0
length = len(validation_images_files)
# also save any missed plates

def evaluate():
  for index, image_path in enumerate(validation_images_files):
    with Image.open(image_path) as img:
      detected_plates_result = detector.inference(img)
      for detected_plate in detected_plates_result:
        img = detected_plate.resize((200,100))
        gray = img.convert("L")
      
        result = extractor.inference(np.array(gray))
        for plate in result.plates:
          if plate != str(csv.label[index].strip()):
            plate = processor.validate(plate)
          
          if plate == csv.label[index]:
            right_choices.append({
              "plate": plate,
              "index": index
            })
          
  accuracy = (len(right_choices)/length)*100
  logging.info(f"CORRECT: {len(right_choices)}/{length}, ACCURACY: {accuracy:.2f}%") 

if __name__ == "__main__":
  evaluate()