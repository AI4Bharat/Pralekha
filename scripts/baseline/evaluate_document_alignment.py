#!/usr/bin/env python3

import argparse
import os
from glob import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate document alignment given mined document pairs.')
    parser.add_argument('--alignment_dir', type=str, required=True,
                        help='Directory containing the mined document alignment files.')
    parser.add_argument('--aligned_counts_file', type=str, required=True,
                        help='Path to the file containing gold standard aligned counts for each language.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the evaluation results will be saved.')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.0, 1.0, 1.03, 1.035, 1.04, 1.045, 1.05, 1.055, 1.06, 1.065, 1.07],
                        help='List of threshold values to use for margin score filtering.')
    parser.add_argument('--output_mode', type=str, choices=['all', 'average'], default='all',
                        help='Output mode: "all" to output results for all languages and the average, "average" to output only the averaged results.')
    args = parser.parse_args()
    return args

def load_alignment_pairs(alignment_file):
    """
    Load document alignment pairs from the alignment file.
    Returns a list of tuples: (doc_id_lang1, doc_id_lang2, margin_score)
    """
    alignment_pairs = []
    with open(alignment_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                doc_id_lang1 = parts[0]
                doc_id_lang2 = parts[1]
                margin_score = float(parts[2])
                alignment_pairs.append((doc_id_lang1, doc_id_lang2, margin_score))
            else:
                print(f"Invalid line in alignment file: {line.strip()}")
    return alignment_pairs

def load_aligned_counts(file_path):
    """
    Load aligned counts from a file.
    The file format should be:
    lang_code aligned_count
    """
    aligned_counts = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                print(f"Invalid line in aligned counts file: {line.strip()}")
                continue
            lang_code = parts[0]
            try:
                aligned_count = int(parts[1])
                aligned_counts[lang_code] = aligned_count
            except ValueError:
                print(f"Invalid aligned count for language {lang_code}: {parts[1]}")
    return aligned_counts

def main():
    args = parse_arguments()
    alignment_dir = args.alignment_dir
    output_dir = args.output_dir
    thresholds = args.thresholds
    output_mode = args.output_mode

    os.makedirs(output_dir, exist_ok=True)

    # Load aligned counts from the provided file
    gold_aligned_numbers = load_aligned_counts(args.aligned_counts_file)

    # Collect all alignment files in the directory
    alignment_files = glob(os.path.join(alignment_dir, 'eng_*', 'mined_parallel_eng_*.tsv'))

    if not alignment_files:
        print(f"No alignment files found in {alignment_dir}")
        return

    # Dictionary to store evaluation results for each language
    language_results = {}

    # For each alignment file
    for alignment_file in alignment_files:
        # Extract language code from filename
        base_name = os.path.basename(alignment_file)
        dir_name = os.path.basename(os.path.dirname(alignment_file))
        if dir_name.startswith('eng_'):
            lang1 = 'eng'
            lang2 = dir_name[4:]  # Get language code after 'eng_'
        else:
            print(f"Unexpected directory name: {dir_name}")
            continue

        if lang2 not in gold_aligned_numbers:
            print(f"No gold standard alignment number found for language {lang2}. Skipping.")
            continue

        gold_aligned_number = gold_aligned_numbers[lang2]

        # Load alignment pairs
        alignment_pairs = load_alignment_pairs(alignment_file)

        # Initialize a list to store evaluation results
        evaluation_results = []

        # For each threshold
        for threshold in thresholds:
            # Filter pairs based on threshold
            filtered_pairs = [(doc_id_lang1, doc_id_lang2, margin_score)
                              for doc_id_lang1, doc_id_lang2, margin_score in alignment_pairs
                              if margin_score >= threshold]

            total_predicted_pairs = len(filtered_pairs)
            if total_predicted_pairs == 0:
                precision, recall, f1_score = 0.0, 0.0, 0.0
            else:
                # Count correct predictions (assuming doc IDs are the same when they are correctly aligned)
                correct_predictions = sum(1 for doc_id_lang1, doc_id_lang2, _ in filtered_pairs
                                          if doc_id_lang1 == doc_id_lang2)

                # Calculate precision, recall, F1 score
                precision = correct_predictions / total_predicted_pairs
                recall = correct_predictions / gold_aligned_number
                f1_score = (2 * precision * recall / (precision + recall)
                            if precision + recall > 0 else 0.0)

            # Store results
            evaluation_results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1_score': f1_score,
            })

        # Store results for the language
        language_results[lang2] = evaluation_results

        # If output_mode is 'all', save individual language results
        if output_mode == 'all':
            output_file = os.path.join(output_dir, f"evaluation_eng_{lang2}.txt")
            with open(output_file, 'w') as f_out:
                f_out.write("Thd\tR\tP\tF1\n")
                for result in evaluation_results:
                    f_out.write(f"{result['threshold']:.3f}\t{result['recall']:.4f}\t{result['precision']:.4f}\t{result['f1_score']:.4f}\n")

    num_languages = len(language_results)
    if num_languages == 0:
        print("No evaluation results to average.")
        return

    # Compute average metrics across all languages
    if output_mode in ['all', 'average']:
        print("Averaged Results Across All Languages:")
        print("Thd\tAvg_R\tAvg_P\tAvg_F1")

        averaged_results = []
        for i, threshold in enumerate(thresholds):
            total_recall = sum(language_results[lang][i]['recall'] for lang in language_results)
            total_precision = sum(language_results[lang][i]['precision'] for lang in language_results)
            total_f1 = sum(language_results[lang][i]['f1_score'] for lang in language_results)

            avg_recall = total_recall / num_languages
            avg_precision = total_precision / num_languages
            avg_f1 = total_f1 / num_languages

            averaged_results.append({
                'threshold': threshold,
                'avg_recall': avg_recall,
                'avg_precision': avg_precision,
                'avg_f1_score': avg_f1,
            })

            print(f"{threshold:.3f}\t{avg_recall:.4f}\t{avg_precision:.4f}\t{avg_f1:.4f}")

        # Save averaged results to a file
        avg_output_file = os.path.join(output_dir, "evaluation_averaged.txt")
        with open(avg_output_file, 'w') as f_out:
            f_out.write("Thd\tAvg_R\tAvg_P\tAvg_F1\n")
            for result in averaged_results:
                f_out.write(f"{result['threshold']:.3f}\t{result['avg_recall']:.4f}\t{result['avg_precision']:.4f}\t{result['avg_f1_score']:.4f}\n")

if __name__ == '__main__':
    main()
