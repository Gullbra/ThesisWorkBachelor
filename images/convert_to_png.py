import sys
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_SIZE = (256, 256)


def convert_image(pgm_path: Path) -> Path:
  png_path = pgm_path.with_suffix(".png")
  img = Image.open(pgm_path)
  
  if img.size != TARGET_SIZE:
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
  
  img.save(png_path)
  pgm_path.unlink()
  return png_path


def main():
  if len(sys.argv) != 2:
    print("Usage: python convert.py <folder_path>")
    sys.exit(1)

  folder = Path(sys.argv[1])
  image_files = [f for ext in ("*.pgm", "*.jpeg", "*.jpg") for f in folder.glob(ext)]
  total = len(image_files)

  if total == 0:
    print("No .pgm, .jpg, or .jpeg files found.")
    sys.exit(1)

  print(f"Converting {total} images...")

  completed = 0
  with ThreadPoolExecutor() as executor:
    futures = {executor.submit(convert_image, p): p for p in image_files}
    for future in as_completed(futures):
      future.result()  # re-raises any exception
      completed += 1
      print(f"{completed}/{total}", end="\r")

  print(f"\nDone! Converted {total} images.")

if __name__ == "__main__":
  main()
