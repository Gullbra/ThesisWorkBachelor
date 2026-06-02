#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <folder_path> <val_percent> <test_percent>"
  exit 1
fi

folder_path="$1"
val_pct="$2"
test_pct="$3"
train_pct=$((100 - val_pct - test_pct))

if [[ $train_pct -le 0 ]]; then
  echo "Error: val + test percentages must be less than 100"
  exit 1
fi

if [[ -z "$(ls -A "$folder_path" 2>/dev/null)" ]]; then
  echo "Error: dataset folder is empty"
  exit 1
fi

existing=$(find "$folder_path/train" "$folder_path/val" "$folder_path/test" -maxdepth 0 2>/dev/null | wc -l)
if [[ $existing -gt 0 ]]; then
  echo "Error: destination folders already exist, aborting to prevent data loss"
  exit 1
fi

echo "Split: train=$train_pct% val=$val_pct% test=$test_pct%"

read -p "Would you like to convert images to 256x256 .png [y/N]? " response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
  python "$(dirname "$0")/convert_to_png.py" "$folder_path"
fi

mapfile -t images < <(find "$folder_path" -maxdepth 1 \( -name "*.pgm" -o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | sort)
total=${#images[@]}

if [[ $total -eq 0 ]]; then
  echo "Error: no images found in $folder_path"
  exit 1
fi

n_test=$(( total * test_pct / 100 ))
n_val=$(( total * val_pct / 100 ))
n_train=$(( total - n_test - n_val ))

echo "Total: $total images → train=$n_train val=$n_val test=$n_test"

mkdir -p "$folder_path/train" "$folder_path/val" "$folder_path/test"

echo "Moving images..."

for (( i=0; i<total; i++ )); do
  img="${images[$i]}"
  if   [[ $i -lt $n_train ]]; then
    mv "$img" "$folder_path/train/"
  elif [[ $i -lt $(( n_train + n_val )) ]]; then
    mv "$img" "$folder_path/val/"
  else
    mv "$img" "$folder_path/test/"
  fi
done

echo "Done!"
