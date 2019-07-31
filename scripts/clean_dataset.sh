#!/usr/bin/bash

RESUME_DIR="${1}"

mkdir -p "$RESUME_DIR"/cleaned

for file in dataset/*.json; do
  echo "Cleaning for $file"
  filename=${file##*/}
  ipython -m src.cleaning $file "$RESUME_DIR"/cleaned/$filename
done
