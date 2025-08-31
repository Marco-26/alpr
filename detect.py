from ultralytics import YOLO
from PIL import Image

class Detector:
  def __init__(self):
    self.model = YOLO("/Users/mcosta/dev/alpr/runs/detect/train/weights/best.pt")
  
  def finetune(self, variation, data, epochs, img_size, device):
    model = YOLO(variation)
    results = model.train(data=data, epochs=epochs, imgsz=img_size, device=device)
    print(results)
    
  def inference(self, img_path) -> Image:
    result = self.model(img_path)

    xyxy = result[0].boxes.xyxy.numpy()
    (a,b,c,d) = xyxy[0]

    with Image.open(img_path) as img:
      resized = img.crop((a,b,c,d))
      
    return resized

