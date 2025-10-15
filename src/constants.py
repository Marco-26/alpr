PORTUGUESE_LICENSE_PLATE_REGEX="(?:[A-Z]{2}[0-9]{2}[A-Z]{2}|[0-9]{2}[A-Z]{2}[0-9]{2})"

YOLO_MODEL_PATH = "/Users/mcosta/dev/alpr/outputs/runs/detect/train/weights/best.pt"

CONFUSION_PAIRS = {
  'I':'1',
  '1': 'I',
  'O':'0',
  '0':'O'
}

PATTERNS = {
  'format_2005':{
    'example': '41-BE-81',
    'regex': r'^[0-9]{2}[A-Z]{2}[0-9]{2}$',
    'letter_pos': [2,3],
    'number_pos': [0,1,4,5]
  },
  'format_2020':{
    'example': 'AA-16-AA',
    'regex': r'^[A-Z]{2}[0-9]{2}[A-Z]{2}$',
    'letter_pos': [0,1,4,5],
    'number_pos': [2,3]
  }
}