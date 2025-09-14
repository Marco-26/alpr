import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

from ml.scripts.detect import Detector
from ml.scripts.extract import Extractor

from utils import is_image_obj as is_image

app = FastAPI()
detector = Detector()
extractor = Extractor()

@app.post("/inference")
async def inference(img: UploadFile = File(...)):
  img_bytes = await img.read()
  with Image.open(io.BytesIO(img_bytes)) as img:
    print(img.verify())

  # if not is_image(image):
  #   return JSONResponse(status_code=400, content={
  #     "success":"false",
  #     "data":"null",
  #     "error": "File is not an image."
  #   })
  
  # cropped = detector.inference(image)
  
  return JSONResponse(status_code=200, content={
      "success":"true",
      "data":"hello",
      "error": ""
    })
  