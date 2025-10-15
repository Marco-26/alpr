from post_processor import PostProcessor
from extract import Extractor
from detect import Detector
import os

extractor = Extractor()
detector = Detector()
processor = PostProcessor()

# Evaluate the performance of model flow (extractor & detector) to performance of the processor.
# Test with 100 images to calculate the accuracy of detector and accuracy of the processor.

# To evaluate the extractor also measure the confidence so we can see how the confidence correlates with accuracy.

images_dir = "/Users/mcosta/dev/alpr/outputs/data/train/images"

if __name__ == "__main__":
  print(len(os.listdir(images_dir)))