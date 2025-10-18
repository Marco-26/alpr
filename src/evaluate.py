from post_processor import PostProcessor
from extract import Extractor
from detect import Detector
from PIL import Image
import glob
import pandas as pd

extractor = Extractor()
detector = Detector()
processor = PostProcessor()

# Evaluate the performance of model flow (extractor & detector) to performance of the processor.
# Test with 100 images to calculate the accuracy of detector and accuracy of the processor.

# To evaluate the extractor also measure the confidence so we can see how the confidence correlates with accuracy.

images_dir = "/Users/mcosta/dev/alpr/outputs/data/train/images"
images_files = glob.glob(f'{images_dir}/*.jpg')

base_ocr_images = glob.glob("/Users/mcosta/dev/alpr/ocr_eval_images/*.jpg")
labels = "ocr_eval_labels.csv"

if __name__ == "__main__":
  csv = pd.read_csv(labels)
  for index, image_path in enumerate(base_ocr_images):
    with Image.open(image_path) as img:
      result = extractor.inference(img)
      print(result.plate)
        
    