import easyocr
from dataclasses import dataclass
import re
from constants import PORTUGUESE_LICENSE_PLATE_REGEX
@dataclass
class ExtractionResult:
  plate: str | None
  confidence: float
  error: str | None = None
  
  @property
  def succeeded(self) -> bool:
    return self.error is None and self.plate is not None

class Extractor:
  def __init__(self):
    self.reader = easyocr.Reader(['en'])
  
  def normalize(self, text:str) -> str:
    text = text.replace(" ", "")
    norm_text = re.sub(r'[^A-Za-z0-9]', '',text)
    return norm_text.upper()

  def inference(self, img) -> dict:
    try:
      result = self.reader.readtext(img)
    except Exception as e:
      return ExtractionResult(plate="", confidence=0.0, error=f"Error detecting plate: {e}")
    
    if not result:
      return ExtractionResult(plate="", confidence=0.0, error="Unexpected Error occurred")
    
    (_, text, conf) = result[0]

    plate_normalized = self.normalize(text)

    if plate_normalized:
      return ExtractionResult(plate=plate_normalized, confidence=conf.item(), error="")
      