CONFUSION_PAIRS = {
  'I':'1',
  '1': 'I',
  'O':'0',
  '0':'O'
}

patterns = {
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

# returns the pattern for the licence plate in question, based on scoring
def position_scoring(normalized_plate) -> str:
  """
    normalized plates dont have dashes
  """
  scores = []
  for name, pattern in patterns.items():
    current_score = 0
    for index, letter in enumerate(normalized_plate):
      if (letter.isdigit() and index in pattern["number_pos"]) or (not letter.isdigit() and index in pattern["letter_pos"]):
        current_score +=1
    scores.append((name, current_score))
  best = max(scores, key=lambda x: x[1])
  pattern, _ = best
  return pattern  

def apply_substitutions_rules(guessed_plate, pattern) -> str:
  letter_pos = patterns[pattern]["letter_pos"]
  number_pos = patterns[pattern]["number_pos"]
  
  for index, char in enumerate(guessed_plate):
    if (char.isdigit() and index in letter_pos) or (not char.isdigit() and index in number_pos):
      guessed_plate = guessed_plate[:index] + CONFUSION_PAIRS[char] + guessed_plate[index + 1:]
   
  return guessed_plate