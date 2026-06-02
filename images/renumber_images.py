import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".pgm")

def renumber_images(folder_path: Path, ask_confirmation=True):
  """
  Renumbers image files in a folder to close gaps in numbering sequence.
  Starting from 1, 2, 3, etc. preserving each file's original extension.
  Supports: .png, .jpg, .jpeg, .pgm
  """

  # Get all supported image files with numeric names
  image_files = []
  for ext in SUPPORTED_EXTENSIONS:
    for file in folder_path.glob(f"*{ext}"):
      try:
        image_files.append(file)
      except ValueError:
        continue

  image_files.sort(key=lambda f: int(f.stem))

  if not image_files:
    print("No numbered image files found in the folder.")
    return

  if ask_confirmation:
    print("\nRenaming plan:")

  rename_plan = []
  for new_num, file in enumerate(image_files, start=1):
    new_name = folder_path / f"{new_num}{file.suffix}"
    if file != new_name:
      rename_plan.append((file, new_name))
      if ask_confirmation:
        print(f" {file.name} -> {new_name.name}")

  if not rename_plan:
    print("No renaming needed - files are already numbered sequentially!")
    return

  if ask_confirmation:
    response = input("\nProceed with renaming? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
      print("Renaming cancelled.")
      return

  # Collect names that are final destinations to detect conflicts
  final_names = {new_name for _, new_name in rename_plan}

  temp_files = []
  for file, new_name in rename_plan:
    if file in final_names:
      temp_name = folder_path / f"temp_{file.name}"
      file.rename(temp_name)
      temp_files.append((temp_name, new_name))
    else:
      file.rename(new_name)

  for temp_file, new_name in temp_files:
    temp_file.rename(new_name)

  print(f"\nSuccessfully renumbered {len(rename_plan)} files!")


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python renumber_images.py <folder_path> <ask_confirmation>")
    sys.exit(1)

  folder_path = Path(sys.argv[1])

  if not folder_path.exists():
    print(f"Error: Folder '{folder_path}' does not exist.")
    sys.exit(1)

  renumber_images(folder_path, len(sys.argv) >= 3)
