#!/bin/bash

# Remove the previous ZIP file
rm -f xholan11_xhorni20.zip

# Create a temporary directory
tmp_dir=tmp_dir_zip
mkdir "$tmp_dir"

# Copy files to the temporary directory
mkdir "$tmp_dir/SRC"

if [ ! -f ".pdf" ]; then
    echo "No pdf file for documentation found!"
    exit 1
fi

cp *.pdf "$tmp_dir"
find ./src -name "*.py" -exec cp --parents '{}' "$tmp_dir/SRC" ';'

rm -rf "$tmp_dir/SRC/src/libs"

cp *.ipynb "$tmp_dir/SRC"
cp README.md "$tmp_dir/SRC"
cp -r img "$tmp_dir/SRC"
cp -r data/results/* "$tmp_dir"

# Create the ZIP file
cd "$tmp_dir" || exit 1
zip -r ../xholan11_xhorni20.zip *

# Clean up the temporary directory
cd ..
rm -rf "$tmp_dir"