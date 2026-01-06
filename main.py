"""
https://stackoverflow.com/questions/918154/relative-paths-in-python
"""


from glob import glob
from pathlib import Path
# import sys


# Path configurations
BASE_DIR = Path(__file__).parent

image_paths = {
  'test': (BASE_DIR / "images/BOSSbase_1.01[.png]/test").resolve(),
  'train': (BASE_DIR / "images/BOSSbase_1.01[.png]/train").resolve(),
  'val': (BASE_DIR / "images/BOSSbase_1.01[.png]/val").resolve()
}
for path in image_paths.values():
  path.mkdir(parents=True, exist_ok=True)
  (path / "cover").resolve().mkdir(parents=True, exist_ok=True)
  (path / "stego").resolve().mkdir(parents=True, exist_ok=True)


def test_model():
  from srnet_model import test_srnet
  test_srnet(
    cover_img_path=image_paths['test'] / "cover",
    stego_img_path=image_paths['test'] / "stego",
  )


def create_stego_images():
  from steganography import LsbSequential
  stego_tool = LsbSequential()

  cover_images = sorted((image_paths['test'] / "cover").glob("*.png"))
  output_dir = image_paths['test'] / "stego"

  for img_path in cover_images:
    output_path = output_dir / img_path.name

    stego_tool.encode_message(
      image_path=img_path,
      threshold=0.5,
      output_path=output_path
    )


if __name__ == "__main__":
  # create_stego_images()
  test_model()
  pass
