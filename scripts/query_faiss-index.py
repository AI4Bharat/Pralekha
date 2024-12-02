#!/usr/bin/env python3

import argparse
import os
import faiss
import numpy as np
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description='Query FAISS index with embeddings from another language.')
    parser.add_argument('--index_lang', type=str, required=True,
                        help='Language code of the index (e.g., eng).')
    parser.add_argument('--data_lang', type=str, required=True,
                        help='Language code of the query data (e.g., hin).')
    parser.add_argument('--index_dir', type=str, required=True,
                        help='Directory where the FAISS index is stored.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory where the query embeddings are stored.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the query results will be saved.')
    parser.add_argument('--top_n', type=int, default=16,
                        help='Number of nearest neighbors to retrieve (default: 16).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use for querying (default: 0).')
    args = parser.parse_args()

    # Validate language codes
    valid_langs = {'eng', 'asm', 'ben', 'guj', 'hin', 'kan', 'mal', 'mar',
                   'nep', 'ori', 'pan', 'san', 'snd', 'tam', 'tel', 'urd'}
    if args.index_lang not in valid_langs:
        parser.error(f"Invalid index language code: {args.index_lang}. Choose from {', '.join(valid_langs)}.")
    if args.data_lang not in valid_langs:
        parser.error(f"Invalid data language code: {args.data_lang}. Choose from {', '.join(valid_langs)}.")

    return args

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

def load_index_and_mapping(index_file, mapping_file, gpu_id=0, use_gpu=True):
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found at {index_file}")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"ID mapping file not found at {mapping_file}")

    print(f"Loading index from {index_file}")
    cpu_index = faiss.read_index(index_file)

    print(f"Loading ID mapping from {mapping_file}")
    with open(mapping_file, 'rb') as f:
        id_mapping = pickle.load(f)

    if use_gpu:
        # Initialize GPU resources
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        # Enable useFloat16LookupTables to reduce shared memory usage
        co.useFloat16 = True
        # Optionally, disable precomputed tables to reduce memory usage
        co.usePrecomputed = False

        # Move index to GPU
        print(f"Moving index to GPU {gpu_id} with useFloat16LookupTables=True...")
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, co)
        return gpu_index, id_mapping
    else:
        return cpu_index, id_mapping

def process_queries_with_index(index, id_mapping, embedding_file, args, output_dir):
    # Load embeddings
    embeddings, doc_ids = load_embeddings(embedding_file)
    print(f"Loaded {len(embeddings)} embeddings for language {args.data_lang}.")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Perform search on index
    D, I = index.search(embeddings, args.top_n)

    results = {}
    for idx, (distances, indices) in enumerate(zip(D, I)):
        q_doc_id = doc_ids[idx]
        distances = distances.tolist()
        indices = indices.tolist()
        # Map index IDs back to original doc IDs
        index_doc_ids = [id_mapping.get(idx_id, None) for idx_id in indices]
        # Prepare result
        result_list = []
        for score, idx_id, idx_doc_id in zip(distances, indices, index_doc_ids):
            if idx_doc_id is None:
                continue  # Skip if ID not found in mapping
            result_list.append({
                'score': float(score),
                'index_id': int(idx_id),
                'doc_id': idx_doc_id
            })
        # Store results per query
        results[q_doc_id] = {
            'total_score': sum(item['score'] for item in result_list),
            'results': result_list
        }

    # Save results
    output_file = os.path.join(output_dir, f"query_results_{args.data_lang}_to_{args.index_lang}.pkl")
    with open(output_file, 'wb') as f_out:
        pickle.dump(results, f_out)
    print(f"Saved query results to {output_file}.")

def main():
    args = parse_arguments()

    # Get index file and mapping file
    index_file = os.path.join(args.index_dir, f"{args.index_lang}_index.faiss")
    mapping_file = os.path.join(args.index_dir, f"{args.index_lang}_id_mapping.pkl")

    # Load index and mapping
    index, id_mapping = load_index_and_mapping(index_file, mapping_file, gpu_id=args.gpu_id, use_gpu=True)

    # Load query embeddings
    embedding_file = os.path.join(args.data_dir, f"{args.data_lang}.pkl")
    if not os.path.isfile(embedding_file):
        raise FileNotFoundError(f"Embedding file not found at {embedding_file}")

    # Prepare output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process queries with index
    print(f"Processing queries with index language {args.index_lang} and data language {args.data_lang}")
    process_queries_with_index(index, id_mapping, embedding_file, args, output_dir)

    # Clean up
    del index
    del id_mapping

    print("Querying completed successfully.")

if __name__ == "__main__":
    main()