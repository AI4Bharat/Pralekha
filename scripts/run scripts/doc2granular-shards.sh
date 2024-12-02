#!/bin/bash

# Variables
INPUT_DIR="data"                     # Base directory with language subdirectories
OUTPUT_SENT_DIR="output/sentences"   # Directory to store sentence JSONL shards
OUTPUT_CHUNK_DIR="output/chunks"     # Directory to store chunk JSONL shards
LANGUAGES=("eng" "hin" "tam")        # Languages to process
SHARD_SIZE=50000                     # Number of entries per shard
BATCH_SIZE=100                       # Number of files processed per batch
SENTSPCHUNK=4                        # Number of sentences per chunk (2,4,8)

# Enable/Disable Sentence Tokenization
# Comment the following block to disable sentence tokenization
echo "Starting sentence tokenization..."
python doc2sentences.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_SENT_DIR \
  --languages ${LANGUAGES[@]} \
  --shard_size $SHARD_SIZE \
  --batch_size $BATCH_SIZE \
  --min_words $MIN_WORDS
echo "Sentence tokenization completed."

# Enable/Disable Chunk Splitting
# Comment the following block to disable chunk splitting
echo "Starting chunk splitting..."
python doc2chunks.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_CHUNK_DIR \
  --languages ${LANGUAGES[@]} \
  --sentspchunk $SENTSPCHUNK \
  --shard_size $SHARD_SIZE \
  --batch_size $BATCH_SIZE \
  --min_words $MIN_WORDS
echo "Chunk splitting completed."