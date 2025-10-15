from ultralytics import YOLO
from PIL import Image
import constants

#TODO: Use case : Multiple cars on picture.
#TODO: Train this model with bigger dataset and with images with multiple vehicles.

class Detector:
  def __init__(self):
    self.model = YOLO(constants.YOLO_MODEL_PATH)
  
  def finetune(self, variation, data, epochs, img_size, device):
    model = YOLO(variation)
    results = model.train(data=data, epochs=epochs, imgsz=img_size, device=device)
    print(results)
  
  def inference(self, img: str) -> Image:
    result = self.model(img)
    xyxy = result[0].boxes.xyxy.numpy()
    (a,b,c,d) = xyxy[0]

    resized = img.crop((a,b,c,d))
      
    return resized
