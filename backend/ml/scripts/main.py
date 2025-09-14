from detect import Detector
from extract import Extractor
from backend.utils import is_image
from PIL import Image
import os 
import sys
import numpy as np

img_path = '/Users/mcosta/dev/alpr/traffic_multiple_cars.png'
detector = Detector()
extractor = Extractor()

if __name__ == "__main__": 
  if not os.path.exists(img_path):
    print('File provided does not exists')
    sys.exit(1)

  if not is_image(img_path):
    print('File provided is not an image')
    sys.exit(1)
  
  detector = Detector()
  extractor = Extractor()
  
  cropped = detector.inference(img_path=img_path)