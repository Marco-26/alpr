"""CLI: detect plates in an image and OCR them."""

import logging
import numpy as np
from PIL import Image
from detect import Detector
from extract import Extractor
from post_processor import PostProcessor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="Image path for the vehicle used to extract the license plate")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

extractor = Extractor()
detector = Detector()
processor = PostProcessor()

if __name__ == "__main__":
  args = parser.parse_args()
  image_file_path = args.image_path
  
  with Image.open(image_file_path) as img:
    results = detector.inference(img)
    if not results:
      logging.error(f"No license plates were detected in the image: {image_file_path}")
    for result in results:
      img = result.resize((200,100))
      gray = result.convert("L")
      
      result = extractor.inference(np.array(gray))
      plates = result.plates
      for plate in plates:
        logging.info("Candidate plate text: %s", plate)
