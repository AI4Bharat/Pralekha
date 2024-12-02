import os
import sys
import argparse
import pickle
import glob
from tqdm import tqdm
import time
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Mine parallel sentence IDs based on cosine similarity.')
    parser.add_argument('--lang1', type=str, required=True,
                        help='Language code of the first language (e.g., eng).')
    parser.add_argument('--lang2', type=str, required=True,
                        help='Language code of the second language (e.g., hin).')
    parser.add_argument('--query_results_dir', type=str, required=True,
                        help='Directory containing the query result .pkl files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the mined parallel IDs will be saved.')
    parser.add_argument('--batch_size', type=int, default=1600,
                        help='Approximate number of .pkl files to process in each batch (default: 1600).')
    args = parser.parse_args()
    return args

def load_and_merge_results(batch_files):
    # Process the batch of pkl files
    merged_results = {}
    group_results = {}
    for pkl_file in tqdm(batch_files, desc="Processing batch of pkl files"):
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
            for qid, res_list in results.items():
                if qid not in group_results:
                    group_results[qid] = []
                group_results[qid].extend(res_list)
    # For each query ID, keep the top result with the highest score
    for qid, res_list in group_results.items():
        top_result = max(res_list, key=lambda x: x['score'])
        merged_results[qid] = top_result
    return merged_results

def extract_pairs(merged_results):
    # Extract sentence IDs and scores
    pairs = []
    for qid, result in merged_results.items():
        pair = {
            'query_id': qid,
            'sentence_id': result['sentence_id'],
            'score': result['score']
        }
        pairs.append(pair)
    return pairs

def main():
    args = parse_arguments()
    query_results_dir = args.query_results_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    lang1 = args.lang1
    lang2 = args.lang2

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Cosine similarity: Processing language pair: {lang1} to {lang2}")
    print("Finding all .pkl files...")
    # Find all .pkl files in the directory
    all_pkl_files = glob.glob(os.path.join(query_results_dir, '*.pkl'))
    if not all_pkl_files:
        print(f"No .pkl files found in {query_results_dir}")
        sys.exit(1)
    all_pkl_files.sort()

    # Group files by base name without batch number
    file_groups = defaultdict(list)
    for pkl_file in all_pkl_files:
        # Extract base name without batch number
        base_name = pkl_file
        if '_batch_' in pkl_file:
            base_name = pkl_file.split('_batch_')[0]
        file_groups[base_name].append(pkl_file)

    # Create batches of base names, ensuring files with the same base name are in the same batch
    base_names = list(file_groups.keys())
    batches = []
    current_batch = []
    current_batch_size = 0

    for base_name in base_names:
        num_files = len(file_groups[base_name])
        if current_batch_size + num_files > batch_size and current_batch:
            # Start a new batch
            batches.append(current_batch)
            current_batch = [base_name]
            current_batch_size = num_files
        else:
            current_batch.append(base_name)
            current_batch_size += num_files

    if current_batch:
        batches.append(current_batch)

    num_batches = len(batches)

    for batch_idx, batch_base_names in enumerate(batches):
        start_time = time.time()
        print(f"Processing batch {batch_idx + 1}/{num_batches}")
        # Collect all pkl files in this batch
        batch_files = []
        for base_name in batch_base_names:
            batch_files.extend(file_groups[base_name])

        merged_results = load_and_merge_results(batch_files)
        pairs = extract_pairs(merged_results)

        # Save results for this batch
        output_file = os.path.join(output_dir, f'mined_ids_{lang1}_{lang2}_batch_{batch_idx + 1}.pkl')
        save_results(pairs, output_file)
        end_time = time.time()
        print(f"Time taken for batch {batch_idx + 1}: {end_time - start_time:.2f} seconds")

        # Free memory
        del merged_results
        del pairs

    print("All batches processed and results saved.")

def save_results(pairs, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(pairs, f)
    print(f"Saved mined parallel IDs to {output_file}")

if __name__ == '__main__':
    main()
