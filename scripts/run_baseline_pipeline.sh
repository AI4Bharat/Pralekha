#!/bin/bash

# Baseline pipeline: Document embedding, FAISS index creation, and parallel document mining

# Variables
INPUT_SENT_EMBED_DIR="data/sentence_embeddings"       # Directory containing sentence embeddings
OUTPUT_DOC_EMBED_DIR="output/document_embeddings"    # Directory for document-level embeddings
FAISS_INDEX_DIR="output/faiss_indices"               # Directory for FAISS indices
QUERY_RESULT_DIR="output/query_results"              # Directory for query results
PARALLEL_DOC_DIR="output/parallel_docs"              # Directory for mined parallel documents
LANGUAGES=("eng" "hin")                              # Languages to process
EMBED_METHOD="mean_pooling"                          # Document embedding method: mean_pooling, length_weighting, idf_weighting, lidf
TOP_N=16                                             # Number of nearest neighbors for FAISS queries

# Step 1: Compute document embeddings
echo "Step 1: Computing document embeddings..."
mkdir -p "$OUTPUT_DOC_EMBED_DIR"
for LANG in ${LANGUAGES[@]}; do
  echo "Processing language: $LANG"
  python doc_embeddings.py \
    --input_dir "$INPUT_SENT_EMBED_DIR/$LANG" \
    --output_dir "$OUTPUT_DOC_EMBED_DIR" \
    --method "$EMBED_METHOD" \
    --langs "$LANG"
done
echo "Document embedding computation completed."

# Step 2: Create FAISS indices
echo "Step 2: Creating FAISS indices..."
mkdir -p "$FAISS_INDEX_DIR"
for LANG in ${LANGUAGES[@]}; do
  echo "Creating FAISS index for language: $LANG"
  python add_vectors_to_faiss-index.py \
    --lang "$LANG" \
    --input_data_dir "$OUTPUT_DOC_EMBED_DIR" \
    --output_index_dir "$FAISS_INDEX_DIR"
done
echo "FAISS index creation completed."

# Step 3: Query FAISS indices
echo "Step 3: Querying FAISS indices..."
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
        --data_dir "$OUTPUT_DOC_EMBED_DIR" \
        --output_dir "$QUERY_OUTPUT_DIR" \
        --top_n "$TOP_N"
    fi
  done
done
echo "Querying FAISS indices completed."

# Step 4: Mine parallel documents
echo "Step 4: Mining parallel documents..."
mkdir -p "$PARALLEL_DOC_DIR"
for LANG1 in ${LANGUAGES[@]}; do
  for LANG2 in ${LANGUAGES[@]}; do
    if [ "$LANG1" != "$LANG2" ]; then
      echo "Mining parallel documents between $LANG1 and $LANG2..."
      cd baseline
      python max-margin_parallel-docs-mining.py \
        --lang1 "$LANG1" \
        --lang2 "$LANG2" \
        --query_result_dir "../$QUERY_RESULT_DIR" \
        --output_dir "../$PARALLEL_DOC_DIR" \
        --k "$TOP_N"
      cd ..
    fi
  done
done
echo "Parallel document mining completed."

echo "Baseline pipeline execution completed successfully."