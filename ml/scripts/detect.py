from ultralytics import YOLO
from PIL import Image

#TODO: Use case : Multiple cars on picture.
#TODO: Train this model with bigger dataset and with images with multiple vehicles.

class Detector:
  def __init__(self):
    self.model = YOLO("/Users/mcosta/dev/alpr/ml/runs/detect/train/weights/best.pt")
  
  def finetune(self, variation, data, epochs, img_size, device):
    model = YOLO(variation)
    results = model.train(data=data, epochs=epochs, imgsz=img_size, device=device)
    print(results)
    
  def inference(self, img_path) -> Image:
    results = self.model(img_path)
    print(f'Got {len(results)} results, {results}')
    for result in results:
      result[0].save()
    
    return
    xyxy = result[0].boxes.xyxy.numpy()
    (a,b,c,d) = xyxy[0]

    with Image.open(img_path) as img:
      resized = img.crop((a,b,c,d))
      
    return resized

