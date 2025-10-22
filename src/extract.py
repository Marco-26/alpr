from dataclasses import dataclass
import re
from fast_plate_ocr import ONNXPlateRecognizer
@dataclass
class ExtractionResult:
  plates: list | None
  error: str | None = None

class Extractor:
  def __init__(self):
    self.model = ONNXPlateRecognizer("european-plates-mobile-vit-v2-model")
  
  def normalize(self, text:str) -> str:
    text = text.replace(" ", "")
    norm_text = re.sub(r'[^A-Za-z0-9]', '',text)
    return norm_text.upper()

  def inference(self, img) -> dict:
    normalized_plates = []
    try:
      detections = self.model.run(img)
    except Exception as e:
      return ExtractionResult(plates=[], error=f"Error detecting plate: {e}")
    
    if not detections:
      return ExtractionResult(plates=[], error="Unexpected Error occurred")
    
    for detection in detections:
      normalized_plates.append(self.normalize(detection))

    return ExtractionResult(plates=normalized_plates)
      