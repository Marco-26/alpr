"""OCR-based license plate text extraction and normalization."""

from dataclasses import dataclass
import re
from typing import List, Optional
from fast_plate_ocr import LicensePlateRecognizer
@dataclass
class ExtractionResult:
  """OCR result: candidate `plates` and optional `error`."""

  plates: Optional[List[str]]
  error: Optional[str] = None

class Extractor:
  def __init__(self):
    """Load the licence plate recognizer."""
    self.model = LicensePlateRecognizer("european-plates-mobile-vit-v2-model")
  
  def normalize(self, text:str) -> str:
    """Return uppercase alphanumeric string (no spaces/dashes)."""
    text = text.replace(" ", "")
    norm_text = re.sub(r'[^A-Za-z0-9]', '', text)
    return norm_text.upper()

  def inference(self, img) -> ExtractionResult:
    """Run OCR and return an `ExtractionResult` with normalized candidates."""
    normalized_plates: List[str] = []
    try:
      detections = self.model.run(img)
    except Exception as e:
      return ExtractionResult(plates=[], error=f"Error detecting plate: {e}")

    if not detections:
      return ExtractionResult(plates=[], error="Unexpected Error occurred")

    for detection in detections:
      normalized_plates.append(self.normalize(detection))

    return ExtractionResult(plates=normalized_plates)
      
