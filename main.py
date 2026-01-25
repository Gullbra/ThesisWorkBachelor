from pathlib import Path
from enum import Enum
from rename_imgs import renumber_images


class ImgGroup(str, Enum):
  TEST = "TEST"
  TRAIN = "TRAIN"
  VAL = "VALIDATION"


BASE_DIR = Path(__file__).parent
image_paths = {
  ImgGroup.TEST: (BASE_DIR / "images/BOSSbase_1.01[.png]/test").resolve(),
  ImgGroup.TRAIN: (BASE_DIR / "images/BOSSbase_1.01[.png]/train").resolve(),
  ImgGroup.VAL: (BASE_DIR / "images/BOSSbase_1.01[.png]/val").resolve()
}
for key,path in image_paths.items():
  path.mkdir(parents=True, exist_ok=True)
  (path / "cover").resolve().mkdir(parents=True, exist_ok=True)
  (path / "stego").resolve().mkdir(parents=True, exist_ok=True)
  image_paths[key] = {
    "cover": (path / "cover").resolve(),
    "stego": (path / "stego").resolve()
  }


def test_model():
  from srnet_model import test_srnet, TestMode
  test_srnet(
    cover_img_dir=image_paths[ImgGroup.TEST]["cover"],
    stego_img_dir=image_paths[ImgGroup.TEST]["stego"],
    limit_images=120,
    mode=TestMode.COVER_ONLY,
    verbose=True
  )


def create_stego_images(stenographic_function):
  for key, path in image_paths.values():
    print(f"Processing {key} images...")
    renumber_images(path["cover"], ask_confirmation=False) 
    cover_images = sorted(path["cover"].glob("*.png"))

    for index in range(len(cover_images)):
      image_path = cover_images[index]
      if index % 1000 == 0:
        print(f"{index}/{len(cover_images)} ({index/len(cover_images):.1f}) images embedded")

      stenographic_function(
        image_path,
        path["stego"] / image_path.name
      )


def create_stego_lsb_random(threshold: float):
  from steganography import LsbRandom
  steg_tool = LsbRandom(123456789)
  create_stego_images(lambda cover_image_path, stego_image_path: steg_tool.encode_message(
    image_path=cover_image_path,
    output_path=stego_image_path,
    target_threshold=threshold,
  ))


if __name__ == "__main__":
  # create_stego_lsb_random(0.5)
  test_model()
  pass
