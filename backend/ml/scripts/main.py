from post_processor import PostProcessor

processor = PostProcessor()

if __name__ == "__main__":
  plate = processor.validate("A120CD")
  print(plate)