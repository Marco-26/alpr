from PIL import Image

def is_image_path(img_path:str) -> bool:
  try:
    with Image.open(img_path) as img:
      img.verify()
      return True
  except:
    print("Error")
    return False

def is_image_obj(img_obj: Image) -> bool:
  try:
    with img_obj.open() as img:
      img.verify()
      return True
  except:
    print("Error")
    return False