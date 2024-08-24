#!/bin/bash

DATA_URL="ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz"
QUERIES_URL="ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz"
GROUNDTRUTH_URL="ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz"

# Set the output directory where the dataset will be downloaded
OUTPUT_DIR="./SIFT-dataset"

echo "Downloading the SIFT dataset to $OUTPUT_DIR..."

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download the SIFT1B dataset using wget
#wget -P "$OUTPUT_DIR" "$DATA_URL"

# Download the queries using wget
#wget -P "$OUTPUT_DIR" "$QUERIES_URL"

# Download the groundtruth using wget
#wget -P "$OUTPUT_DIR" "$GROUNDTRUTH_URL"

# Extract the dataset (if it's in a compressed format)

# uncompress the gz file
gunzip -v "$OUTPUT_DIR/bigann_base.bvecs.gz"
gunzip -v "$OUTPUT_DIR/bigann_query.bvecs.gz"
tar -xf "$OUTPUT_DIR/bigann_gnd.tar.gz" -C "$OUTPUT_DIR"

# Extract the dataset (if it's in a compressed format)
# Uncomment the following line if needed
# tar -xf "$OUTPUT_DIR/sift1b.tar.gz" -C "$OUTPUT_DIR"

# Optional: Remove the downloaded archive file
# Uncomment the following line if needed
# rm "$OUTPUT_DIR/sift1b.tar.gz"

echo "Dataset downloaded successfully!"