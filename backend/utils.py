from PIL import Image
import io

def is_image_path(img_path:str) -> bool:
  try:
    with Image.open(img_path) as img:
      img.verify()
      return True
  except:
    print("Error")
    return False

def is_image_obj(img_bytes: bytes) -> bool:
  try:
    with Image.open(io.BytesIO(img_bytes)) as img:
      img.verify()
      return True
  except:
    print("Error")
    return False