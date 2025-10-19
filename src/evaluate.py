from post_processor import PostProcessor
from extract import Extractor
from detect import Detector
from PIL import Image
import glob
import pandas as pd
import numpy as np
import os

extractor = Extractor()
detector = Detector()
processor = PostProcessor()

# Evaluate the performance of model flow (extractor & detector) to performance of the processor.
# Test with 100 images to calculate the accuracy of detector and accuracy of the processor.

# To evaluate the extractor also measure the confidence so we can see how the confidence correlates with accuracy.

images_dir = "/Users/mcosta/dev/alpr/outputs/data/train/images"
images_files = glob.glob(f'{images_dir}/*.jpg')

base_ocr_images = sorted(glob.glob("/Users/mcosta/dev/alpr/ocr_eval_images/*.jpg"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)
labels = "ocr_eval_labels.csv"

# MAKE FOUR TESTS:
# 1. With no algorithm and no fine-tune
# 2. With no algorithm and fine-tune
# 3. With algorithm and no fine-tune
# 4. with algorithm and fine-tune

right_choices = []

if __name__ == "__main__":
  csv = pd.read_csv(labels)
  for index, image_path in enumerate(base_ocr_images):
    if index == 30:
      break
    
    with Image.open(image_path) as img:
      img = img.resize((200,100))
      gray = img.convert("L")
      gray.save(f"normalized_ocr_inputs/{index}.jpg")
    
      results = extractor.inference(np.array(gray))
      for plate in results.plate:
        if plate == csv.label[index]:
          right_choices.append({
            "plate": plate,
            "index": index
          })
        
  print(f"Stats: {len(right_choices)}/30 correct guesses") 
    