from PIL import Image

def is_image(img_path:str) -> bool:
  try:
    with Image.open(img_path) as img:
      img.verify()
      return True
  except:
    print("Error")
    return False    