#!/usr/bin/env python3

import os
import argparse
import h5py
import pickle
from tqdm import tqdm
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate the number of sentences/chunks each document contains.')
    parser.add_argument('--lang', type=str, required=True,
                        help='Language code to process.')
    parser.add_argument('--input_data_dir', type=str, required=True,
                        help='Directory where the input embeddings are stored.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the counts will be saved.')
    parser.add_argument('--chunk', action='store_true',
                        help='If set, use chunk_ids instead of sentence_ids.')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # Prepare input directory
    embeddings_dir = args.input_data_dir
    h5_files = [os.path.join(root, file)
                for root, _, files in os.walk(embeddings_dir)
                for file in files if file.endswith('.h5')]

    if not h5_files:
        print(f"No HDF5 files found in {embeddings_dir}")
        return

    h5_files.sort()

    # Dictionary to store counts per doc_id
    doc_counts = defaultdict(int)

    # Set to store all doc_ids encountered
    all_doc_ids = set()

    for h5_file in tqdm(h5_files, desc="Processing HDF5 files"):
        with h5py.File(h5_file, 'r') as f:
            # Check if required datasets exist
            if 'doc_ids' not in f:
                print(f"'doc_ids' dataset not found in {h5_file}. Skipping this file.")
                continue

            # Determine whether to use sentence_ids or chunk_ids
            id_key = 'chunk_ids' if args.chunk else 'sentence_ids'

            if id_key not in f:
                print(f"'{id_key}' dataset not found in {h5_file}. Skipping this file.")
                continue

            ids = f[id_key][:]
            doc_ids = f['doc_ids'][:]

            # Ensure ids and doc_ids are decoded if necessary
            ids = [id_.decode('utf-8') if isinstance(id_, bytes) else id_ for id_ in ids]
            doc_ids = [doc_id.decode('utf-8') if isinstance(doc_id, bytes) else doc_id for doc_id in doc_ids]

            # Check if lengths of ids and doc_ids match
            if len(ids) != len(doc_ids):
                print(f"Length mismatch between '{id_key}' and 'doc_ids' in {h5_file}. Skipping this file.")
                continue

            # Build counts per doc_id and perform the check
            for id_, doc_id in zip(ids, doc_ids):
                all_doc_ids.add(doc_id)  # Add to set of all doc_ids

                # Initialize doc_counts[doc_id] if not already present
                _ = doc_counts[doc_id]  # Accessing the key ensures it's in the dict

                # Check if id_.split('_')[:-1] == doc_id
                id_prefix = '_'.join(id_.split('_')[:-1])
                if id_prefix != doc_id:
                    print(f"Mismatch in {h5_file}: doc_id '{doc_id}' does not match id '{id_}' (prefix '{id_prefix}')")
                else:
                    doc_counts[doc_id] += 1

    # Calculate and print total number of documents and sentences/chunks
    total_docs = len(all_doc_ids)
    total_sentences = sum(doc_counts.values())

    print(f"Total number of documents: {total_docs}")
    print(f"Total number of sentences/chunks: {total_sentences}")

    # Check for doc_ids with count == 0
    zero_count_docs = [doc_id for doc_id in all_doc_ids if doc_counts[doc_id] == 0]
    if zero_count_docs:
        print(f"Documents with count == 0:")
        for doc_id in zero_count_docs:
            print(f"- {doc_id}")
    else:
        print("No documents with count == 0")

    print (doc_counts['pib-196211'])
    print (doc_counts['pib-33677'])

    # Save the counts
    output_dir = os.path.join(args.output_dir, args.lang)
    os.makedirs(output_dir, exist_ok=True)
    counts_file = os.path.join(output_dir, 'sent_num.pkl')
    with open(counts_file, 'wb') as f_out:
        pickle.dump(doc_counts, f_out)

    print(f"Counts saved to {counts_file}")

if __name__ == '__main__':
    main()
