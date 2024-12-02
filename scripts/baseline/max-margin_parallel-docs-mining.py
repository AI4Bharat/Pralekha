#!/usr/bin/env python3

import os
import argparse
import pickle
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Mine and merge parallel documents using Max Margin approach.')
    parser.add_argument('--lang1', type=str, required=True,
                        help='Language code of the first language (e.g., eng).')
    parser.add_argument('--lang2', type=str, required=True,
                        help='Language code of the second language (e.g., hin).')
    parser.add_argument('--query_result_dir', type=str, required=True,
                        help='Directory containing the query result files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the mined parallel documents will be saved.')
    parser.add_argument('--k', type=int, default=16,
                        help='Number of nearest neighbors (default: 16).')
    args = parser.parse_args()
    return args

def load_query_results(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_mined_pairs(data_lang1_to_lang2, data_lang2_to_lang1, lang1, lang2, k):
    # Prepare dictionaries for fast lookup
    # For lang1_to_lang2
    lang1_to_lang2_scores = {}
    for query_id, result in data_lang1_to_lang2.items():
        results = result.get('results', [])
        # We assume that results are sorted by score
        # Keep top k results
        lang1_to_lang2_scores[query_id] = results[:k]

    # For lang2_to_lang1
    lang2_to_lang1_scores = {}
    for query_id, result in data_lang2_to_lang1.items():
        results = result.get('results', [])
        # We assume that results are sorted by score
        # Keep top k results
        lang2_to_lang1_scores[query_id] = results[:k]

    # Compute margins and mine parallel documents
    mined_pairs = []

    print(f"Computing margins and mining parallel documents for {lang1} to {lang2}...")
    for lang1_id, lang1_results in tqdm(lang1_to_lang2_scores.items()):
        for res in lang1_results:
            lang2_id = res['doc_id']
            score = res['score']

            # Get avg scores from both directions
            # Average the scores of the top k results
            avg_score_lang1 = sum(r['score'] for r in lang1_results) / k

            lang2_results = lang2_to_lang1_scores.get(lang2_id, [])
            if not lang2_results:
                continue
            avg_score_lang2 = sum(r['score'] for r in lang2_results) / k

            # Compute margin
            margin = score / (0.5 * (avg_score_lang1 + avg_score_lang2))

            mined_pairs.append({
                f'id_{lang1}': lang1_id,
                f'id_{lang2}': lang2_id,
                'margin': margin
            })

    return mined_pairs

def merge_and_deduplicate(pairs_list, lang1, lang2):
    """
    Merges two lists of pairs and removes duplicates.
    From high score to low score, if an ID has already appeared, discard the pair.
    """
    # Combine the pairs
    all_pairs = pairs_list[0] + pairs_list[1]

    # Sort the pairs by margin in descending order
    all_pairs.sort(key=lambda x: x['margin'], reverse=True)

    # Deduplicate pairs
    unique_id_lang1 = set()
    unique_id_lang2 = set()
    dedup_pairs = []
    for pair in all_pairs:
        id1 = pair[f'id_{lang1}']
        id2 = pair[f'id_{lang2}']
        margin = pair['margin']
        if id1 not in unique_id_lang1 and id2 not in unique_id_lang2:
            dedup_pairs.append(pair)
            unique_id_lang1.add(id1)
            unique_id_lang2.add(id2)
        else:
            # Skip pairs where either ID has already appeared
            continue
    return dedup_pairs

def main():
    args = parse_arguments()

    lang1 = args.lang1
    lang2 = args.lang2
    query_result_dir = args.query_result_dir
    output_dir = args.output_dir
    k = args.k

    os.makedirs(output_dir, exist_ok=True)

    # Paths to the query result pickle files
    file_lang1_to_lang2 = os.path.join(query_result_dir, f"{lang1}_to_{lang2}", f"query_results_{lang1}_to_{lang2}.pkl")
    file_lang2_to_lang1 = os.path.join(query_result_dir, f"{lang2}_to_{lang1}", f"query_results_{lang2}_to_{lang1}.pkl")

    # Check if files exist
    if not os.path.exists(file_lang1_to_lang2):
        print(f"File not found: {file_lang1_to_lang2}")
        return
    if not os.path.exists(file_lang2_to_lang1):
        print(f"File not found: {file_lang2_to_lang1}")
        return

    # Load the query results
    print(f"Loading query results from {file_lang1_to_lang2}")
    data_lang1_to_lang2 = load_query_results(file_lang1_to_lang2)

    print(f"Loading query results from {file_lang2_to_lang1}")
    data_lang2_to_lang1 = load_query_results(file_lang2_to_lang1)

    # Compute mined pairs in both directions
    mined_pairs_lang1_to_lang2 = compute_mined_pairs(data_lang1_to_lang2, data_lang2_to_lang1, lang1, lang2, k)
    mined_pairs_lang2_to_lang1 = compute_mined_pairs(data_lang2_to_lang1, data_lang1_to_lang2, lang2, lang1, k)

    # For the second direction, swap the IDs to match the first direction
    for pair in mined_pairs_lang2_to_lang1:
        # Swap the IDs
        pair[f'id_{lang1}'], pair[f'id_{lang2}'] = pair[f'id_{lang2}'], pair[f'id_{lang1}']

    # Merge and deduplicate
    print("Merging and deduplicating pairs")
    merged_pairs = merge_and_deduplicate([mined_pairs_lang1_to_lang2, mined_pairs_lang2_to_lang1], lang1, lang2)

    # Save merged and deduplicated pairs to output file
    output_file = os.path.join(output_dir, f'mined_parallel_{lang1}_{lang2}.tsv')
    with open(output_file, 'w') as f_out:
        for pair in merged_pairs:
            f_out.write(f"{pair[f'id_{lang1}']}\t{pair[f'id_{lang2}']}\t{pair['margin']}\n")

    print(f"Mined parallel documents saved to {output_file}")
    print("Mining and merging completed.")

if __name__ == '__main__':
    main()
