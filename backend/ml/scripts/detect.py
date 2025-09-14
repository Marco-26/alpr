from ultralytics import YOLO
from PIL import Image

#TODO: Use case : Multiple cars on picture.
#TODO: Train this model with bigger dataset and with images with multiple vehicles.

class Detector:
  def __init__(self):
    self.model = YOLO("/Users/mcosta/dev/alpr/backend/ml/runs/detect/train/weights/best.pt")
  
  def finetune(self, variation, data, epochs, img_size, device):
    model = YOLO(variation)
    results = model.train(data=data, epochs=epochs, imgsz=img_size, device=device)
    print(results)
  
  def inference(self, img_path) -> Image:
    # right now, this only works on images of one car
    result = self.model(img_path)
    xyxy = result[0].boxes.xyxy.numpy()
    (a,b,c,d) = xyxy[0]

    with Image.open(img_path) as img:
      resized = img.crop((a,b,c,d))
      
    return resized

