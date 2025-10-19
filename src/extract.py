from dataclasses import dataclass
import re
from fast_plate_ocr import ONNXPlateRecognizer
@dataclass
class ExtractionResult:
  plate: list | None
  confidence: float
  error: str | None = None
  
  @property
  def succeeded(self) -> bool:
    return self.error is None and self.plate is not None

class Extractor:
  def __init__(self):
    self.model = ONNXPlateRecognizer("european-plates-mobile-vit-v2-model")
  
  def normalize(self, text:str) -> str:
    text = text.replace(" ", "")
    norm_text = re.sub(r'[^A-Za-z0-9]', '',text)
    return norm_text.upper()

  def inference(self, img) -> dict:
    plate_normalized = []
    try:
      results = self.model.run(img)
    except Exception as e:
      return ExtractionResult(plate="", confidence=0.0, error=f"Error detecting plate: {e}")
    
    if not results:
      return ExtractionResult(plate="", confidence=0.0, error="Unexpected Error occurred")
    
    for result in results:
      plate_normalized.append(self.normalize(result))

    if len(plate_normalized) > 0:
      return ExtractionResult(plate=plate_normalized, confidence=0.0, error="")
      