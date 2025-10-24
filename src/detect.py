"""YOLO-based license plate detector returning cropped plate images."""

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from PIL import Image
import constants
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER.setLevel(logging.WARNING)
class Detector:
  """YOLO detector for localizing license plates."""

  def __init__(self):
    """Load weights and set a confidence threshold."""
    self.model = YOLO(constants.YOLO_MODEL_PATH, verbose=False)
    self.threshold = 0.7
  
  def finetune(self, variation, data, epochs, img_size, device):
    """Fine-tune a YOLO variation on the given dataset."""
    model = YOLO(variation)
    model.train(data=data, epochs=epochs, imgsz=img_size, device=device)
  
  def inference(self, img: Image.Image) -> list[Image.Image]:
    """Detect plates and return cropped PIL images (may be empty)."""
    plate_crops: list[Image.Image] = []

    detections = self.model(img)

    for detection in detections:
      if len(detection.boxes.conf) > 0 and max(detection.boxes.conf) >= self.threshold:
        xyxy = detection.boxes.xyxy.numpy()

        if len(xyxy) == 0:
          logging.error("Could not detect a licence plate")
          return []

        (a,b,c,d) = xyxy[0]

        resized = img.crop((a,b,c,d))
        plate_crops.append(resized)

    return plate_crops
