import re
from constants import PORTUGUESE_LICENSE_PLATE_REGEX, PATTERNS, CONFUSION_PAIRS

class PostProcessor:
  def __init__(self):
    pass
  
  def validate(self, raw_plate):
    regex_val = re.search(PORTUGUESE_LICENSE_PLATE_REGEX, raw_plate)
    if not regex_val:
    #   # If program reach here, try to apply substitution rules
      pattern = self.__position_scoring(raw_plate)
      plate_refactored = self.apply_substitutions_rules(raw_plate,pattern)
      return plate_refactored
      
  def __position_scoring(self, normalized_plate) -> str:
    """
      normalized plates dont have dashes
    """
    scores = []
    for name, pattern in PATTERNS.items():
      current_score = 0
      for index, letter in enumerate(normalized_plate):
        if (letter.isdigit() and index in pattern["number_pos"]) or (not letter.isdigit() and index in pattern["letter_pos"]):
          current_score +=1
      scores.append((name, current_score))
    best = max(scores, key=lambda x: x[1])
    pattern, _ = best
    return pattern  

  def __apply_substitutions_rules(self, guessed_plate, pattern) -> str:
    letter_pos = PATTERNS[pattern]["letter_pos"]
    number_pos = PATTERNS[pattern]["number_pos"]
    
    for index, char in enumerate(guessed_plate):
      if (char.isdigit() and index in letter_pos) or (not char.isdigit() and index in number_pos):
        guessed_plate = guessed_plate[:index] + CONFUSION_PAIRS[char] + guessed_plate[index + 1:]
    
    return guessed_plate