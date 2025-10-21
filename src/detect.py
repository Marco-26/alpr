from ultralytics import YOLO
from PIL import Image
import constants

class Detector:
  def __init__(self):
    self.model = YOLO(constants.YOLO_MODEL_PATH)
    self.threshold = 0.7
  
  def finetune(self, variation, data, epochs, img_size, device):
    model = YOLO(variation)
    results = model.train(data=data, epochs=epochs, imgsz=img_size, device=device)
    print(results)
  
  def inference(self, img: Image) -> Image:
    # THIS WILL NOT WORK ON MULTIPLE CARS AS THE MODEL WAS ONLY TRAINED ON SINGLE CAR IMAGES
    cropped_images = []
    
    results = self.model(img)
    
    for result in results:
      if len(result.boxes.conf) > 0 and max(result.boxes.conf) >= self.threshold:
        xyxy = result.boxes.xyxy.numpy()
    
        if len(xyxy) == 0:
          print("Could not detect a licence plate")
          return
        
        (a,b,c,d) = xyxy[0]

        resized = img.crop((a,b,c,d))
        cropped_images.append(resized)
    
    return cropped_images
