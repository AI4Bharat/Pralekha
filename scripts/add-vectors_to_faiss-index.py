#!/usr/bin/env python3

import faiss
import numpy as np
import argparse
import os
import pickle
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create FAISS index from document embeddings for a specific language.')
    parser.add_argument('--lang', type=str, required=True,
                        help='Language code to process.')
    parser.add_argument('--input_data_dir', type=str, required=True,
                        help='Directory where the input embeddings are stored.')
    parser.add_argument('--output_index_dir', type=str, required=True,
                        help='Directory where the index will be saved.')
    return parser.parse_args()

def load_embeddings(embedding_file):
    """
    Load embeddings from a pickled file.

    Returns:
        embeddings: numpy array of shape (num_documents, embedding_dim)
        doc_ids: list of document IDs corresponding to the embeddings
    """
    with open(embedding_file, 'rb') as f:
        document_embeddings = pickle.load(f)

    doc_ids = []
    embeddings = []
    for doc_id, embedding in document_embeddings.items():
        doc_ids.append(doc_id)
        embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype='float32')
    return embeddings, doc_ids

def main():
    args = parse_arguments()

    lang = args.lang
    embedding_file = os.path.join(args.input_data_dir, f"{lang}.pkl")

    if not os.path.isfile(embedding_file):
        print(f"Embedding file {embedding_file} does not exist.")
        return

    embeddings, doc_ids = load_embeddings(embedding_file)
    print(f"Loaded {len(embeddings)} embeddings for language {lang}.")

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    print(f"Created FAISS IndexFlatIP with dimension {embedding_dim}.")

    # Add embeddings to index
    index.add(embeddings)
    print(f"Added {len(embeddings)} vectors to the index.")

    # Map between FAISS IDs and document IDs
    # Since we are using IndexFlatIP, the IDs are assigned in order starting from 0
    id_mapping = {idx: doc_id for idx, doc_id in enumerate(doc_ids)}

    # Create output directory if it doesn't exist
    os.makedirs(args.output_index_dir, exist_ok=True)

    # Save id mapping
    id_mapping_file = os.path.join(args.output_index_dir, f"{lang}_id_mapping.pkl")
    with open(id_mapping_file, 'wb') as f:
        pickle.dump(id_mapping, f)
    print(f"Saved ID mapping to {id_mapping_file}.")

    # Save the index
    index_path = os.path.join(args.output_index_dir, f"{lang}_index.faiss")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}.")

if __name__ == "__main__":
    main()