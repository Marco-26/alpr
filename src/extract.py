"""OCR-based license plate text extraction and normalization."""

from dataclasses import dataclass
import re
from typing import List, Optional
from fast_plate_ocr import LicensePlateRecognizer

class Extractor:
  def __init__(self):
    """Load the licence plate recognizer."""
    self.model = LicensePlateRecognizer("european-plates-mobile-vit-v2-model")
  
  def normalize(self, text:str) -> str:
    """Return uppercase alphanumeric string (no spaces/dashes)."""
    text = text.replace(" ", "")
    norm_text = re.sub(r'[^A-Za-z0-9]', '', text)
    return norm_text.upper()

  def inference(self, img) -> list:
    """Run OCR and return an `ExtractionResult` with normalized candidates."""
    normalized_plates: List[str] = []
    try:
      detections = self.model.run(img)
    except Exception as e:
      return []

    if not detections:
      return []

    for detection in detections:
      normalized_plates.append(self.normalize(detection))

    return normalized_plates
      
