#!/usr/bin/env python3

import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Merge bidirectional mined parallel sentences.')
    parser.add_argument('--input_file1', type=str, required=True,
                        help='Path to the first input file (e.g., mined_parallel_ben_eng.tsv).')
    parser.add_argument('--input_file2', type=str, required=True,
                        help='Path to the second input file (e.g., mined_parallel_eng_ben.tsv).')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output merged file.')
    parser.add_argument('--lang1', type=str, required=True,
                        help='Language code of the first language (e.g., ben).')
    parser.add_argument('--lang2', type=str, required=True,
                        help='Language code of the second language (e.g., eng).')
    args = parser.parse_args()
    return args

def read_file(filepath, reverse_columns=False):
    """
    Reads a TSV file and returns a list of tuples (id_lang1, id_lang2, score).
    If reverse_columns is True, swaps the first two columns.
    """
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                id1, id2, score = parts[0], parts[1], float(parts[2])
                if reverse_columns:
                    id1, id2 = id2, id1  # Swap the IDs
                pairs.append((id1, id2, score))
            else:
                print(f"Invalid line in {filepath}: {line.strip()}")
    return pairs

def merge_and_deduplicate(pairs_list):
    """
    Merges two lists of pairs and removes duplicates.
    From high score to low score, if an ID has already appeared, discard the pair.
    """
    # Combine the pairs
    all_pairs = pairs_list[0] + pairs_list[1]

    # Sort the pairs by score in descending order
    all_pairs.sort(key=lambda x: x[2], reverse=True)

    # Deduplicate pairs
    unique_id1 = set()
    unique_id2 = set()
    dedup_pairs = []
    for id1, id2, score in all_pairs:
        if id1 not in unique_id1 and id2 not in unique_id2:
            dedup_pairs.append((id1, id2, score))
            unique_id1.add(id1)
            unique_id2.add(id2)
        else:
            # Skip pairs where either ID has already appeared
            continue
    return dedup_pairs

def write_output(pairs, output_file):
    """
    Writes the list of pairs to the output file.
    """
    with open(output_file, 'w') as f:
        for id1, id2, score in pairs:
            f.write(f"{id1}\t{id2}\t{score}\n")

def main():
    args = parse_arguments()

    # Read the first input file
    print(f"Reading {args.input_file1}")
    pairs1 = read_file(args.input_file1, reverse_columns=False)

    # Read the second input file and reverse columns
    print(f"Reading and reversing {args.input_file2}")
    pairs2 = read_file(args.input_file2, reverse_columns=True)

    # Merge and deduplicate
    print("Merging and deduplicating pairs")
    merged_pairs = merge_and_deduplicate([pairs1, pairs2])

    # Write the output
    print(f"Writing merged pairs to {args.output_file}")
    write_output(merged_pairs, args.output_file)

    print("Processing completed.")

if __name__ == '__main__':
    main()
