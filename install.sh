#!/usr/bin/env bash

set -euo pipefail
# -e: exit on error
# -u: error on undefined variables
# -o pipefail: fail a pipeline if any command in it fails

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

read -p "Would you like to download the BOSSbase dataset? (~1.6 GB) [y/N] " response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
  images_dir="$(dirname "$0")/images"
  zip_path="$images_dir/BOSSbase_1.01.zip"

  mkdir -p "$images_dir"

  echo "Downloading dataset..."
  curl -L -o "$zip_path" "https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"

  echo "Extracting..."
  unzip "$zip_path" -d "$images_dir"

  rm "$zip_path"
  echo "Done! Dataset extracted to $images_dir"

  bash "$images_dir/categorizeImages.sh" "$images_dir/BOSSbase_1.01" 15 15
fi