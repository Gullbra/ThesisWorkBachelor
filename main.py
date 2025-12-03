"""
https://stackoverflow.com/questions/918154/relative-paths-in-python
"""


from pathlib import Path
import sys

# Path configurations
BASE_DIR = Path(__file__).parent
path_to_cover_images = ""
path_to_preprocessed_images = (BASE_DIR / "images/preprocessed").resolve()
path_to_preprocessed_images.mkdir(parents=True, exist_ok=True)


# Dictionary mapping commands to functions
# commands = {
#   'greet': greet,
#   'calc': calculate,
#   'list': list_items,
#   'help': show_help
# }


def is_validate_image_set(path: str) -> bool:
  """
  Should valdate img folder:
   * Check if extensions are valid
   * If hierarchy matches what the choosen model's gonna expect
  """

  print("[Not Implemented]: Validating image set...")
  return True


def set_img_set():
  """Set path to image set"""

  while True:
    print("\nEnter the path to the desired image set (leave blank for default: \"./images/image_set/\", and enter \"cancel\" to cancel)\nPath: ")

    user_input = input().strip()

    if user_input.lower() == "cancel":
      print("Cancelled setting image set path.")

      if path_to_cover_images == "":
        print("No image set path configured. Shutting down.")
        sys.exit(0)
      return 
    
    elif user_input == "":
      global path_to_cover_images
      path_to_cover_images = (BASE_DIR / "images/image_set").resolve()
      print(f"[Success]: Using default image set path: {path_to_cover_images}")

      if not is_validate_image_set():
        print("[Error]: Invalid image set path. Please try again.")
        continue
      break

    else:
      potential_path = Path(user_input).resolve()

      if not potential_path.exists():
        print("[Error]: The provided path does not exist. Please try again.")
        continue
        
      if not potential_path.is_dir():
        print("[Error]: The provided path is not a directory. Please try again.")
        continue

      path_to_cover_images = potential_path

      if not is_validate_image_set():
        print("[Error]: Invalid image set path. Please try again.")
        continue

      print(f"[Success]: Image set path set to: {path_to_cover_images}")
      break


def preprocess_img():
  """
  Should ask if preprocing should be done and if so, do it. 
  (filetype, resize, grayscale)
  """

  print("[Not Implemented]: Preprocessing image set...")
  return True


def steganography_menu():  
  """
  Should implement(or import) a steganography menu
  """
  print("[Not Implemented]: Steganography menu...")
  pass



def steganalysis_menu():  
  """
  Should implement(or import) a steganalysis menu
  """
  print("[Not Implemented]: Steganalysis menu...")
  pass


def main():
  """Our Cli"""

  print("\n**Steganography/Steganalys CLI: under development.**")

  # input loop: set img set
  set_img_set()

  # input loop: ask if preprocing img
  #   add code for preproc
  preprocess_img()

  # input loop: Ask for Steganography method
  steganography_menu()

  # input loop: Ask for Steganalysis method
  steganalysis_menu()

  # input loop: Analysis Options
  # Something here...


  # print("Welcome! Type 'help' for available commands.")
  # while True:
  #   try:
  #     # Get user input
  #     command = input("\nEnter command: ").strip().lower()
      
  #     # Check for quit command
  #     if command == 'quit':
  #       print("Goodbye!")
  #       break
      
  #     # Execute command if it exists
  #     if command in commands:
  #       commands[command]()
  #     elif command == '':
  #       continue
  #     else:
  #       print(f"Unknown command: '{command}'. Type 'help' for available commands.")
  
  #   except KeyboardInterrupt:
  #     print("\n\nInterrupted. Type 'quit' to exit.")
  #   except EOFError:
  #     print("\n\nGoodbye!")
  #     break


if __name__ == "__main__":
  main()
