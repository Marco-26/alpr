from ultralytics import YOLO
from PIL import Image

model = YOLO("/Users/mcosta/dev/alpr/runs/detect/train/weights/best.pt")
model.info()

#finetuning the model to portuguese licence plates
#results = model.train(data="./data/data.yaml", epochs=20, imgsz=640, device='mps')
#print(results)

img_path = './matricula2.jpg'

result = model(img_path)
result[0].show()

xyxy = result[0].boxes.xyxy.numpy()
(a,b,c,d) = xyxy[0]

img = Image.open(img_path)
resized = img.crop((a,b,c,d))
resized.show()