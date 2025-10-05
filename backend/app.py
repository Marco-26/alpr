import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

from ml.scripts.detect import Detector
from ml.scripts.extract import Extractor
import numpy as np

from pydantic import BaseModel

from utils import is_image_obj as is_image

app = FastAPI()
detector = Detector()
extractor = Extractor()

class InferenceSuccess(BaseModel):
    success: bool = True
    plate: str
    confidence: float

class InferenceFailure(BaseModel):
    success: bool = False
    error: str

@app.post("/inference")
async def inference(img: UploadFile = File(...)):
  img_bytes = await img.read()
  if not is_image(img_bytes):
    return JSONResponse(status_code=400, content={
      "success":"false",
      "data":"null",
      "error": "File is not an image."
    })
   
  with Image.open(io.BytesIO(img_bytes)) as img:
    cropped = detector.inference(img)
    
    arr_img = np.array(cropped)
    extraction = extractor.inference(arr_img)

  if extraction.succeeded:
    return InferenceSuccess(plate=extraction.plate, confidence=extraction.confidence)
  
  return InferenceFailure(error=extraction.error or "Unknown extraction failure")
  