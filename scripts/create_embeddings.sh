#!/bin/bash

# LaBSE Embeddings
# Comment out this block to disable LaBSE embeddings
echo "Generating embeddings with LaBSE..."

INPUT_DIR="output/granular"              # Input directory for LaBSE
OUTPUT_DIR="output/embeddings/labse"    # Output directory for LaBSE
BATCH_SIZE=5120                          # Batch size for LaBSE
GPU_LIST="0"                             # GPU(s) to use for LaBSE (e.g., "0,1,2")
APPROACH="sentence"                      # Granularity to process ("sentence" or "chunk")
LANGUAGES=("eng" "hin" "tam")            # Languages to process for LaBSE

python labse_embeddings.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --gpu_list $GPU_LIST \
  --approach $APPROACH \
  --langs ${LANGUAGES[@]}

echo "LaBSE embeddings generation completed."

# SONAR Embeddings
# Comment out this block to disable SONAR embeddings
echo "Generating embeddings with SONAR..."

INPUT_DIR="output/granular"              # Input directory for SONAR
OUTPUT_DIR="output/embeddings/sonar"    # Output directory for SONAR
BATCH_SIZE=6144                          # Batch size for SONAR
GPU_LIST="0"                             # GPU(s) to use for SONAR (e.g., "0,1,2")
APPROACH="sentence"                      # Granularity to process ("sentence" or "chunk")
NORM=true                               # Normalize SONAR embeddings
LANGUAGES=("eng" "hin" "tam")            # Languages to process for SONAR

python sonar_embeddings.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --gpu_list $GPU_LIST \
  --approach $APPROACH \
  --langs ${LANGUAGES[@]} \
  $([ "$NORM" == true ] && echo "--norm")

echo "SONAR embeddings generation completed."