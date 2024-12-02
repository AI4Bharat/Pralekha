#!/usr/bin/env python3

import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Compute document-level embeddings from sentence-level embeddings.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing sentence-level embeddings.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the document-level embeddings.")
    parser.add_argument("--method", type=str, choices=["mean_pooling", "length_weighting", "idf_weighting", "lidf"], required=True, help="Method to compute document embeddings.")
    parser.add_argument("--langs", type=str, nargs='+', required=True, help="List of language subdirectories to process.")
    parser.add_argument("--corpus_dir", type=str, help="Path to the directory containing the corpus (required for methods involving IDF or sentence length).")
    return parser.parse_args()

def compute_mean_pooling(input_dir, output_file, corpus_dir=None):
    """
    Compute document embeddings using mean pooling.
    """
    embeddings_by_doc = {}
    shard_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    # Determine whether to use 'sentence_ids' or 'chunk_ids'
    id_dataset_name = determine_id_dataset_name(corpus_dir)
    for shard_file in tqdm(shard_files, desc="Loading shards for mean pooling"):
        shard_path = os.path.join(input_dir, shard_file)
        with h5py.File(shard_path, 'r') as f:
            embeddings = f['embeddings'][:]
            doc_ids = f['doc_ids'][:]
            # For mean pooling, we don't need sentence_ids or chunk_ids
            for doc_id, embedding in zip(doc_ids, embeddings):
                doc_id = doc_id.decode('utf-8') if isinstance(doc_id, bytes) else doc_id
                if doc_id not in embeddings_by_doc:
                    embeddings_by_doc[doc_id] = []
                embeddings_by_doc[doc_id].append(embedding)
    # Compute mean pooling
    document_embeddings = {}
    for doc_id, embeddings in embeddings_by_doc.items():
        embeddings = np.array(embeddings)
        document_embedding = np.mean(embeddings, axis=0)
        document_embeddings[doc_id] = document_embedding
    # Save document embeddings
    with open(output_file, 'wb') as f:
        pickle.dump(document_embeddings, f)
    print(f"Saved document embeddings to {output_file}")

def compute_sentence_length(text):
    words = text.strip().split()
    return len(words)

def compute_idf(corpus_by_sentence_id, total_documents, text_key):
    """
    Compute IDF values for sentences or chunks.
    """
    document_frequencies = {}
    for data in corpus_by_sentence_id.values():
        text = data[text_key]
        if text not in document_frequencies:
            document_frequencies[text] = set()
        document_frequencies[text].add(data['doc_id'])
    # Compute IDF
    idf = {}
    for text, doc_ids in document_frequencies.items():
        df = len(doc_ids)
        idf[text] = np.log(total_documents / (df + 1))
    return idf

def compute_length_weighting(input_dir, output_file, corpus_dir, lang):
    """
    Compute document embeddings using sentence length weighting.
    """
    # Load corpus
    corpus_by_sentence_id, text_key, id_key = load_corpus(corpus_dir, lang)
    if not corpus_by_sentence_id:
        print(f"No corpus data for language {lang}. Skipping length weighting computation.")
        return
    # Determine whether to use 'sentence_ids' or 'chunk_ids'
    id_dataset_name = determine_id_dataset_name(corpus_dir)
    # Group embeddings by doc_id and collect sentence lengths
    embeddings_by_doc = {}
    sentence_lengths_by_doc = {}
    # Read embeddings
    shard_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    for shard_file in tqdm(shard_files, desc="Loading shards for length weighting"):
        shard_path = os.path.join(input_dir, shard_file)
        with h5py.File(shard_path, 'r') as f:
            embeddings = f['embeddings'][:]
            doc_ids = f['doc_ids'][:]
            sentence_ids = f[id_dataset_name][:]
            for doc_id, sentence_id, embedding in zip(doc_ids, sentence_ids, embeddings):
                doc_id = doc_id.decode('utf-8') if isinstance(doc_id, bytes) else doc_id
                sentence_id = sentence_id.decode('utf-8') if isinstance(sentence_id, bytes) else sentence_id
                if doc_id not in embeddings_by_doc:
                    embeddings_by_doc[doc_id] = []
                    sentence_lengths_by_doc[doc_id] = []
                embeddings_by_doc[doc_id].append(embedding)
                data = corpus_by_sentence_id.get(sentence_id)
                if data:
                    text = data.get(text_key, '')
                    sentence_length = compute_sentence_length(text)
                else:
                    print(f"Sentence id {sentence_id} not found in corpus.")
                    sentence_length = 1  # Default to 1 if not found
                sentence_lengths_by_doc[doc_id].append(sentence_length)
    # Compute document embeddings using length weights
    document_embeddings = {}
    for doc_id in embeddings_by_doc:
        embeddings = np.array(embeddings_by_doc[doc_id])
        lengths = np.array(sentence_lengths_by_doc[doc_id])
        weighted_embeddings = embeddings * lengths[:, np.newaxis]
        document_embedding = np.sum(weighted_embeddings, axis=0) / np.sum(lengths)
        document_embeddings[doc_id] = document_embedding
    # Save document embeddings
    with open(output_file, 'wb') as f:
        pickle.dump(document_embeddings, f)
    print(f"Saved document embeddings to {output_file}")

def compute_idf_weighting(input_dir, output_file, corpus_dir, lang):
    """
    Compute document embeddings using IDF weighting.
    """
    # Load corpus
    corpus_by_sentence_id, text_key, id_key = load_corpus(corpus_dir, lang)
    if not corpus_by_sentence_id:
        print(f"No corpus data for language {lang}. Skipping IDF weighting computation.")
        return
    # Get total number of documents
    total_documents = get_total_documents(corpus_by_sentence_id)
    # Compute IDF
    idf = compute_idf(corpus_by_sentence_id, total_documents, text_key)
    # Determine whether to use 'sentence_ids' or 'chunk_ids'
    id_dataset_name = determine_id_dataset_name(corpus_dir)
    # Group embeddings by doc_id and collect IDF values
    embeddings_by_doc = {}
    idf_values_by_doc = {}
    # Read embeddings
    shard_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    for shard_file in tqdm(shard_files, desc="Loading shards for IDF weighting"):
        shard_path = os.path.join(input_dir, shard_file)
        with h5py.File(shard_path, 'r') as f:
            embeddings = f['embeddings'][:]
            doc_ids = f['doc_ids'][:]
            sentence_ids = f[id_dataset_name][:]
            for doc_id, sentence_id, embedding in zip(doc_ids, sentence_ids, embeddings):
                doc_id = doc_id.decode('utf-8') if isinstance(doc_id, bytes) else doc_id
                sentence_id = sentence_id.decode('utf-8') if isinstance(sentence_id, bytes) else sentence_id
                if doc_id not in embeddings_by_doc:
                    embeddings_by_doc[doc_id] = []
                    idf_values_by_doc[doc_id] = []
                embeddings_by_doc[doc_id].append(embedding)
                data = corpus_by_sentence_id.get(sentence_id)
                if data:
                    text = data.get(text_key, '')
                    sentence_idf = idf.get(text, 1.0)
                else:
                    print(f"Sentence id {sentence_id} not found in corpus.")
                    sentence_idf = 1.0  # Default IDF if not found
                idf_values_by_doc[doc_id].append(sentence_idf)
    # Compute document embeddings using IDF weights
    document_embeddings = {}
    for doc_id in embeddings_by_doc:
        embeddings = np.array(embeddings_by_doc[doc_id])
        idf_values = np.array(idf_values_by_doc[doc_id])
        weighted_embeddings = embeddings * idf_values[:, np.newaxis]
        document_embedding = np.sum(weighted_embeddings, axis=0) / np.sum(idf_values)
        document_embeddings[doc_id] = document_embedding
    # Save document embeddings
    with open(output_file, 'wb') as f:
        pickle.dump(document_embeddings, f)
    print(f"Saved document embeddings to {output_file}")

def compute_lidf(input_dir, output_file, corpus_dir, lang):
    """
    Compute document embeddings using LIDF method (sentence length and IDF weighting).
    """
    # Load corpus
    corpus_by_sentence_id, text_key, id_key = load_corpus(corpus_dir, lang)
    if not corpus_by_sentence_id:
        print(f"No corpus data for language {lang}. Skipping LIDF computation.")
        return
    # Get total number of documents
    total_documents = get_total_documents(corpus_by_sentence_id)
    # Compute IDF
    idf = compute_idf(corpus_by_sentence_id, total_documents, text_key)
    # Determine whether to use 'sentence_ids' or 'chunk_ids'
    id_dataset_name = determine_id_dataset_name(corpus_dir)
    # Group embeddings by doc_id and collect sentence info
    embeddings_by_doc = {}
    weights_by_doc = {}
    # Read embeddings
    shard_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    for shard_file in tqdm(shard_files, desc="Loading shards for LIDF"):
        shard_path = os.path.join(input_dir, shard_file)
        with h5py.File(shard_path, 'r') as f:
            embeddings = f['embeddings'][:]
            doc_ids = f['doc_ids'][:]
            sentence_ids = f[id_dataset_name][:]
            for doc_id, sentence_id, embedding in zip(doc_ids, sentence_ids, embeddings):
                doc_id = doc_id.decode('utf-8') if isinstance(doc_id, bytes) else doc_id
                sentence_id = sentence_id.decode('utf-8') if isinstance(sentence_id, bytes) else sentence_id
                if doc_id not in embeddings_by_doc:
                    embeddings_by_doc[doc_id] = []
                    weights_by_doc[doc_id] = []
                embeddings_by_doc[doc_id].append(embedding)
                data = corpus_by_sentence_id.get(sentence_id)
                if data:
                    text = data.get(text_key, '')
                    sentence_length = compute_sentence_length(text)
                    sentence_idf = idf.get(text, 1.0)
                else:
                    print(f"Sentence id {sentence_id} not found in corpus.")
                    sentence_length = 1  # Default to 1 if not found
                    sentence_idf = 1.0   # Default IDF if not found
                weight = sentence_length * sentence_idf
                weights_by_doc[doc_id].append(weight)
    # Compute document embeddings using LIDF weights
    document_embeddings = {}
    for doc_id in embeddings_by_doc:
        embeddings = np.array(embeddings_by_doc[doc_id])
        weights = np.array(weights_by_doc[doc_id])
        weighted_embeddings = embeddings * weights[:, np.newaxis]
        document_embedding = np.sum(weighted_embeddings, axis=0) / np.sum(weights)
        document_embeddings[doc_id] = document_embedding
    # Save document embeddings
    with open(output_file, 'wb') as f:
        pickle.dump(document_embeddings, f)
    print(f"Saved document embeddings to {output_file}")

def determine_id_dataset_name(corpus_dir):
    """
    Determine whether to use 'sentence_ids' or 'chunk_ids' when reading from HDF5 files.
    """
    base_name = os.path.basename(corpus_dir)
    if '1' in base_name:
        id_dataset_name = 'sentence_ids'
    else:
        id_dataset_name = 'chunk_ids'
    return id_dataset_name

def load_corpus(corpus_dir, lang):
    """
    Load the corpus data for the specified language.
    Returns:
        corpus_by_sentence_id: dict mapping sentence_id to data
        text_key: 'sentence' or 'chunk', depending on the dataset
        id_key: 'sentence_id' or 'chunk_id', depending on the dataset
    """
    corpus_by_sentence_id = {}
    lang_dir = os.path.join(corpus_dir, lang)
    if not os.path.isdir(lang_dir):
        print(f"Corpus directory {lang_dir} does not exist.")
        return corpus_by_sentence_id, None, None
    # Determine whether to use 'sentence_id' or 'chunk_id' and 'sentence' or 'chunk'
    base_name = os.path.basename(corpus_dir)
    if '1' in base_name:
        id_key = 'sentence_id'
        text_key = 'sentence'
    else:
        id_key = 'chunk_id'
        text_key = 'chunk'
    jsonl_files = [f for f in os.listdir(lang_dir) if f.endswith('.jsonl')]
    for jsonl_file in jsonl_files:
        jsonl_path = os.path.join(lang_dir, jsonl_file)
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                sentence_id = data.get(id_key)
                if sentence_id is None:
                    print(f"{id_key} not found in data: {data}")
                    continue
                corpus_by_sentence_id[sentence_id] = data
    return corpus_by_sentence_id, text_key, id_key

def get_total_documents(corpus_by_sentence_id):
    doc_ids = set()
    for data in corpus_by_sentence_id.values():
        doc_ids.add(data['doc_id'])
    return len(doc_ids)

def process_languages(input_base_dir, output_base_dir, method, langs, corpus_dir=None):
    for lang in tqdm(langs, desc="Processing languages"):
        print(f"Processing language: {lang}")
        input_dir = os.path.join(input_base_dir, lang)
        if not os.path.isdir(input_dir):
            print(f"Input directory {input_dir} does not exist. Skipping.")
            continue
        os.makedirs(output_base_dir, exist_ok=True)
        output_file = os.path.join(output_base_dir, f"{lang}.pkl")
        if method == "mean_pooling":
            compute_mean_pooling(input_dir, output_file, corpus_dir)
        elif method == "length_weighting":
            if corpus_dir is None:
                print("Error: --corpus_dir is required for length_weighting method.")
                return
            compute_length_weighting(input_dir, output_file, corpus_dir, lang)
        elif method == "idf_weighting":
            if corpus_dir is None:
                print("Error: --corpus_dir is required for idf_weighting method.")
                return
            compute_idf_weighting(input_dir, output_file, corpus_dir, lang)
        elif method == "lidf":
            if corpus_dir is None:
                print("Error: --corpus_dir is required for LIDF method.")
                return
            compute_lidf(input_dir, output_file, corpus_dir, lang)
        else:
            print(f"Unknown method: {method}")

def main():
    args = parse_args()
    process_languages(args.input_dir, args.output_dir, args.method, args.langs, args.corpus_dir)

if __name__ == "__main__":
    main()