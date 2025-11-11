import gradio as gr
import numpy as np
from PIL import Image
from detect import Detector
from extract import Extractor

detector = Detector()
extractor = Extractor()

def get_plate_flow(file):
  with Image.open(file) as img:
    plate_crops = detector.inference(img)
    if not plate_crops:
      return None, None, "No license plates were detected in the image."
    
    # Assuming we take the first detected plate for simplicity
    plate_img = plate_crops[0].resize((200, 100))
    gray_plate = plate_crops[0].convert("L")

    ocr_result = extractor.inference(np.array(gray_plate))
    plates = ocr_result.plates
    plate_text = plates[0] if plates else "No text recognized."
    return img, plate_img, plate_text
  
  return None, None, "Error processing the image."

demo = gr.Interface(
  fn=get_plate_flow,
  inputs="file",
  outputs=["image", "image", "text"],
  title="ALPR System",
  description=(
    "Upload an image, the app will detect and crop the licence plate, "
    "upscale it for clarity, and then run OCR to show the recognized text."
  )
)

if __name__ == '__main__':
  demo.launch()