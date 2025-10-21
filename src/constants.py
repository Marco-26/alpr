PORTUGUESE_LICENSE_PLATE_REGEX="(?:[A-Z]{2}[0-9]{2}[A-Z]{2}|[0-9]{2}[A-Z]{2}[0-9]{2})"

YOLO_MODEL_PATH = "/Users/mcosta/dev/alpr/outputs/runs/detect/train/weights/best.pt"

CONFUSION_PAIRS = {
  'I':'1',
  '1': 'I',
  'O':'0',
  '0':'O',
  'S':'5',
  '5':'S'
}

PATTERNS = {
  'format_2020':{
    'example': 'AA-16-AA',
    'regex': r'^[A-Z]{2}[0-9]{2}[A-Z]{2}$',
    'letter_pos': [0,1,4,5],
    'number_pos': [2,3]
  },
  'format_2005':{
    'example': '41-BE-81',
    'regex': r'^[0-9]{2}[A-Z]{2}[0-9]{2}$',
    'letter_pos': [2,3],
    'number_pos': [0,1,4,5]
  },
  'format_1992':{
    'example': '00-00-AA',
    'regex': r'^[0-9]{2}[0-9]{2}[A-Z]{2}$',
    'letter_pos': [4,5],
    'number_pos': [0,1,2,3]
  },
  'format_1932  ':{
    'example': 'AA-00-00',
    'regex': r'^[A-Z]{2}[0-9]{2}[0-9]{2}$',
    'letter_pos': [0,1],
    'number_pos': [2,3,4,5]
  },
}