import easyocr

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

    normalized = self.normalize(text)
    
    if normalized:
      return {"text": normalized, "conf":conf}
      