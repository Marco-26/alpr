import numpy as np
from PIL import Image
from detect import Detector
from extract import Extractor
from post_processor import PostProcessor

extractor = Extractor()
detector = Detector()
processor = PostProcessor()

if __name__ == "__main__":
  image_file_path = "/Users/mcosta/dev/alpr/outputs/data/valid/images/F18DV223_jpg.rf.2ff8bd704f355c42f820dd3ffeba280b.jpg"
  
  with Image.open(image_file_path) as img:
    results = detector.inference(img)
    for result in results:
      img = result.resize((200,100))
      gray = result.convert("L")
      
      result = extractor.inference(np.array(gray))
      plates = result.plates
      print(f"Identified {len(plates)} licence plates")
      for plate in plates:
        print(f"Identified the following licence plate: {plate}")