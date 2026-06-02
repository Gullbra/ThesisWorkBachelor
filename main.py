from pathlib import Path
from enum import Enum
import argparse


class ImgGroup(str, Enum):
  TEST = "TEST"
  TRAIN = "TRAIN"
  VAL = "VALIDATION"


DEFAULT_BASE_DIR = Path(__file__).parent / "images/BOSSbase_1.01"


def build_image_paths(base_dir: Path, stego_method: str | None = None) -> dict:
  image_paths = {
    ImgGroup.TEST: (base_dir / "test").resolve(),
    ImgGroup.TRAIN: (base_dir / "train").resolve(),
    ImgGroup.VAL: (base_dir / "val").resolve()
  }

  stego_dir_name = (
    f"stego_{stego_method}"
    if stego_method is not None
    else "stego"
  )

  for key, path in image_paths.items():
    (path / "cover").mkdir(parents=True, exist_ok=True)
    (path / stego_dir_name).mkdir(parents=True, exist_ok=True)

    image_paths[key] = {
      "cover": (path / "cover").resolve(),
      "stego": (path / stego_dir_name).resolve()
    }

  return image_paths


def test_model(image_paths):
  from steganalysis.srnet_model import test_srnet, TestMode

  test_srnet(
    cover_img_dir=image_paths[ImgGroup.TEST]["cover"],
    stego_img_dir=image_paths[ImgGroup.TEST]["stego"],
    limit_images=120,
    mode=TestMode.BALANCED,
    verbose=True
  )


def create_stego_images(image_paths, steganographic_function):
  for key, path in image_paths.items():
    print(f"Processing {key} images...")
    cover_images = sorted(path["cover"].glob("*.png"))

    for index in range(len(cover_images)):
      image_path = cover_images[index]
      if index % 1000 == 0:
        print(f"{index}/{len(cover_images)} ({index/len(cover_images) * 100:.1f}%) images embedded")

      steganographic_function(
        image_path,
        path["stego"] / image_path.name
      )


def create_stego_lsb_sequential(image_paths, threshold: float):
  from steganography import LsbSequential
  steg_tool = LsbSequential()
  create_stego_images(image_paths, lambda cover_image_path, stego_image_path: steg_tool.encode_message(
    image_path=cover_image_path,
    output_path=stego_image_path,
    target_threshold=threshold,
  ))


def create_stego_lsb_random(image_paths, threshold: float):
  from steganography import LsbRandom
  steg_tool = LsbRandom(123456789)
  create_stego_images(image_paths, lambda cover_image_path, stego_image_path: steg_tool.encode_message(
    image_path=cover_image_path,
    output_path=stego_image_path,
    target_threshold=threshold,
  ))


def create_stego_lsb_matching(image_paths, threshold: float):
  from steganography import LsbMatching
  steg_tool = LsbMatching()

  create_stego_images(image_paths, lambda cover_image_path, stego_image_path: steg_tool.encode_message(
    image_path=cover_image_path,
    output_path=stego_image_path,
    target_threshold=threshold,
  ))


def create_stego_lsb_adaptive(image_paths, threshold: float):
  from steganography import LsbSobelEdge
  steg_tool = LsbSobelEdge()

  create_stego_images(image_paths, lambda cover_image_path, stego_image_path: steg_tool.encode_message(
    image_path=cover_image_path,
    output_path=stego_image_path,
    target_threshold=threshold,
  ))


STEGO_METHODS = {
  "sequential": create_stego_lsb_sequential,
  "random":     create_stego_lsb_random,
  "matching":   create_stego_lsb_matching,
  "adaptive":   create_stego_lsb_adaptive,
}


def placeholder(image_paths):
  print("placeholder")


ANALYSIS_METHODS = {
  "cnn": placeholder,
  "rs":  placeholder,
}


def main():
  parser = argparse.ArgumentParser(
    description="Steganography dataset tool",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # Defines the path to dataset flag
  parser.add_argument(
    "--dataset-path", "-d",
    type=Path,
    default=DEFAULT_BASE_DIR,
    metavar="PATH",
    help="Path to the dataset base directory"
  )

  # Defines the main command flag, difining whether steganography of steganalysis should be performed.
  subparsers = parser.add_subparsers(dest="command", required=True)

  # Defines the analysis flag
  analysis_parser = subparsers.add_parser("analysis", help="Run analysis")
  analysis_parser.add_argument(
    "analysis_method",
    choices=ANALYSIS_METHODS.keys(),
    help="Analysis method choice"
  )
  analysis_parser.add_argument(
    "stego_method",
    choices=STEGO_METHODS.keys(),
    help="LSB embedding method to analyse"
  )

  # Defines the flag for creating stego pictures from the dataset
  stego_parser = subparsers.add_parser("stego", help="Create stego images")
  stego_parser.add_argument(
    "stego_method",
    choices=STEGO_METHODS.keys(),
    help="LSB embedding method to use"
  )
  stego_parser.add_argument(
    "--threshold", "-t",
    type=float,
    default=0.25,
    help="Embedding threshold (0.0–1.0)"
  )

  args = parser.parse_args()
  image_paths = build_image_paths(args.dataset_path, args.stego_method)

  print(args)

  if args.command == "stego":
    STEGO_METHODS[args.stego_method](image_paths, args.threshold)
  elif args.command == "analysis":
    ANALYSIS_METHODS[args.analysis_method](image_paths)


if __name__ == "__main__":
  main()
  pass
