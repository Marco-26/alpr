from dataclasses import dataclass
import re
from fast_plate_ocr import ONNXPlateRecognizer
@dataclass
class ExtractionResult:
  plates: list | None
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
    normalized_plates = []
    try:
      results = self.model.run(img)
    except Exception as e:
      return ExtractionResult(plates=[], confidence=0.0, error=f"Error detecting plate: {e}")
    
    if not results:
      return ExtractionResult(plates=[], confidence=0.0, error="Unexpected Error occurred")
    
    for result in results:
      normalized_plates.append(self.normalize(result))

    if len(normalized_plates) > 0:
      return ExtractionResult(plates=normalized_plates, confidence=0.0, error="")
      