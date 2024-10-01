#!/bin/bash

DATA_URL="ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz"
QUERIES_URL="ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz"
GROUNDTRUTH_URL="ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz"

# Set the output directory where the dataset will be downloaded
OUTPUT_DIR="./SIFT-dataset"

echo "Downloading the SIFT dataset to $OUTPUT_DIR..."

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download the queries using wget
echo "Downloading the queries..."
wget -P "$OUTPUT_DIR" "$QUERIES_URL"

# Download the groundtruth using wget
echo "Downloading the groundtruth..."
wget -P "$OUTPUT_DIR" "$GROUNDTRUTH_URL"

# Download the SIFT1B dataset using wget
echo "Downloading the SIFT1B dataset...expecting 92GB download"
wget -P "$OUTPUT_DIR" "$DATA_URL"

# Extract the dataset (if it's in a compressed format)

# uncompress the gz file
echo "Extracting the SIFT1B dataset...expecting ~230GB disk space"
gunzip -v "$OUTPUT_DIR/bigann_base.bvecs.gz"

echo "Extracting the queries and groundtruth..."
gunzip -v "$OUTPUT_DIR/bigann_query.bvecs.gz"
tar -xf "$OUTPUT_DIR/bigann_gnd.tar.gz" -C "$OUTPUT_DIR"

echo "Dataset downloaded successfully!"