import easyocr
from dataclasses import dataclass

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
    
    # Portugal case
    if len(text) == 6 or len(text) == 8:
      return text
    
    return ""

  #TODO: Use character confusion matrix and calculate confidence
  def inference(self, img) -> dict:
    result = self.reader.readtext(img)
    
    if not result:
      return {"text":"", "conf":0.0}
    
    # this model has trouble diferentiating 0 and O. Maybe finetune it aswell?
    (_, text, conf) = result[0]

    plate_normalized = self.normalize(text)
      
    if plate_normalized:
      return ExtractionResult(plate=plate_normalized, confidence=conf.item(), error="")
      