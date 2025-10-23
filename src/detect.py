from ultralytics import YOLO
from ultralytics.utils import LOGGER
from PIL import Image
import constants
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER.setLevel(logging.WARNING)
class Detector:
  def __init__(self):
    self.model = YOLO(constants.YOLO_MODEL_PATH, verbose=False)
    self.threshold = 0.7
  
  def finetune(self, variation, data, epochs, img_size, device):
    model = YOLO(variation)
    model.train(data=data, epochs=epochs, imgsz=img_size, device=device)
  
  def inference(self, img: Image) -> list:
    # THIS WILL NOT WORK ON MULTIPLE CARS AS THE MODEL WAS ONLY TRAINED ON SINGLE CAR IMAGES
    plate_crops = []
    
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
