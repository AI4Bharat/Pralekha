#!/bin/bash

# DAC Pipeline: Process embeddings, create FAISS indices, query and mine parallel units, merge results.

# Variables
INPUT_EMBED_DIR="data/document_embeddings"           # Directory containing document embeddings
FAISS_INDEX_DIR="output/faiss_indices"               # Directory for FAISS indices
QUERY_RESULT_DIR="output/query_results"              # Directory for query results
MINED_UNITS_DIR="output/mined_units"                 # Directory for mined parallel units
FINAL_PARALLEL_DOC_DIR="output/parallel_documents"   # Directory for final merged parallel documents
LANGUAGES=("eng" "hin")                              # Languages to process
TOP_N=16                                             # Number of nearest neighbors for FAISS queries
BATCH_SIZE=1600                                      # Batch size for processing mined units

# Step 1: Create FAISS indices
echo "Step 1: Creating FAISS indices..."
mkdir -p "$FAISS_INDEX_DIR"
for LANG in ${LANGUAGES[@]}; do
  echo "Creating FAISS index for language: $LANG"
  python add_vectors_to_faiss-index.py \
    --lang "$LANG" \
    --input_data_dir "$INPUT_EMBED_DIR" \
    --output_index_dir "$FAISS_INDEX_DIR"
done
echo "FAISS index creation completed."

# Step 2: Query FAISS indices
echo "Step 2: Querying FAISS indices..."
mkdir -p "$QUERY_RESULT_DIR"
for LANG1 in ${LANGUAGES[@]}; do
  for LANG2 in ${LANGUAGES[@]}; do
    if [ "$LANG1" != "$LANG2" ]; then
      echo "Querying $LANG1 index with $LANG2 embeddings..."
      QUERY_OUTPUT_DIR="$QUERY_RESULT_DIR/${LANG1}_to_${LANG2}"
      mkdir -p "$QUERY_OUTPUT_DIR"
      python query_faiss-index.py \
        --index_lang "$LANG1" \
        --data_lang "$LANG2" \
        --index_dir "$FAISS_INDEX_DIR" \
        --data_dir "$INPUT_EMBED_DIR" \
        --output_dir "$QUERY_OUTPUT_DIR" \
        --top_n "$TOP_N"
    fi
  done
done
echo "Querying FAISS indices completed."

# Step 3: Mine parallel units
echo "Step 3: Mining parallel units..."
mkdir -p "$MINED_UNITS_DIR"
for LANG1 in ${LANGUAGES[@]}; do
  for LANG2 in ${LANGUAGES[@]}; do
    if [ "$LANG1" != "$LANG2" ]; then
      echo "Mining parallel units between $LANG1 and $LANG2..."
      cd dac
      python max-margin_parallel-units-mining.py \
        --lang1 "$LANG1" \
        --lang2 "$LANG2" \
        --query_results_dir "../$QUERY_RESULT_DIR/${LANG1}_to_${LANG2}" \
        --output_dir "../$MINED_UNITS_DIR" \
        --batch_size "$BATCH_SIZE"
      cd ..
    fi
  done
done
echo "Parallel unit mining completed."

# Step 4: Merge parallel units into documents
echo "Step 4: Merging parallel units into documents..."
mkdir -p "$FINAL_PARALLEL_DOC_DIR"
for LANG1 in ${LANGUAGES[@]}; do
  for LANG2 in ${LANGUAGES[@]}; do
    if [ "$LANG1" != "$LANG2" ]; then
      echo "Merging parallel units between $LANG1 and $LANG2..."
      INPUT_FILE1="$MINED_UNITS_DIR/mined_ids_${LANG1}_${LANG2}_batch_1.pkl"
      INPUT_FILE2="$MINED_UNITS_DIR/mined_ids_${LANG2}_${LANG1}_batch_1.pkl"
      OUTPUT_FILE="$FINAL_PARALLEL_DOC_DIR/merged_parallel_${LANG1}_${LANG2}.tsv"
      cd dac
      python merge_parallel-units_to_docs.py \
        --input_file1 "../$INPUT_FILE1" \
        --input_file2 "../$INPUT_FILE2" \
        --output_file "../$OUTPUT_FILE" \
        --lang1 "$LANG1" \
        --lang2 "$LANG2"
      cd ..
    fi
  done
done
echo "Merging parallel documents completed."

echo "DAC pipeline execution completed successfully."