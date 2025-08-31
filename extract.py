import easyocr

class Extractor:
  def __init__(self):
    self.reader = easyocr.Reader(['en'])
  
  def inference(self, img) -> str:
    result = self.reader.readtext(img)
    
    # this model has trouble diferentiating 0 and O. Maybe finetune it aswell?
    (_, text, _) = result[0]

    text = text.replace(" ", "")
    return text