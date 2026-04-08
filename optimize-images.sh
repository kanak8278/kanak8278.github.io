#!/usr/bin/env zsh

set -euo pipefail

TARGET_DIR="${1:-images/personal}"
MAX_WIDTH="${MAX_WIDTH:-1920}"
JPEG_QUALITY="${JPEG_QUALITY:-75}"
PNG_QUALITY="${PNG_QUALITY:-80}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Target directory not found: $TARGET_DIR" >&2
  exit 1
fi

setopt NULL_GLOB
files=(
  "$TARGET_DIR"/**/*.(jpg|jpeg|png)(N)
  "$TARGET_DIR"/**/*.(JPG|JPEG|PNG)(N)
)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No image files found under $TARGET_DIR"
  exit 0
fi

processed=0
updated=0
saved_bytes=0

for file in "${files[@]}"; do
  [[ -f "$file" ]] || continue

  processed=$((processed + 1))
  before_size=$(stat -f%z "$file")
  extension="${file##*.}"
  extension="${extension:l}"
  tmp_file="${file}.optimized"

  quality="$JPEG_QUALITY"
  if [[ "$extension" == "png" ]]; then
    quality="$PNG_QUALITY"
  fi

  /usr/bin/sips -s formatOptions "$quality" -Z "$MAX_WIDTH" "$file" --out "$tmp_file" >/dev/null 2>&1 || {
    rm -f "$tmp_file"
    echo "Skipped (sips failed): $file"
    continue
  }

  after_size=$(stat -f%z "$tmp_file")

  if [[ "$after_size" -lt "$before_size" ]]; then
    mv "$tmp_file" "$file"
    delta=$((before_size - after_size))
    saved_bytes=$((saved_bytes + delta))
    updated=$((updated + 1))
    echo "Optimized: $file (-$delta bytes)"
  else
    rm -f "$tmp_file"
  fi
done

echo "Processed: $processed files"
echo "Optimized: $updated files"
echo "Saved: $saved_bytes bytes"
