#!/usr/bin/env python3

import argparse
import os
import pickle
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate document alignment evaluation metrics given sentence alignment results.')
    parser.add_argument('--alignment_files', type=str, nargs='+', required=True,
                        help='Paths to the sentence alignment files (e.g., merged_mined_ben_eng.tsv).')
    parser.add_argument('--lang1', type=str, nargs='+', required=True,
                        help='Language codes of the first languages (e.g., ben guj hin).')
    parser.add_argument('--lang2', type=str, required=True,
                        help='Language code of the second language (e.g., eng).')
    parser.add_argument('--sent_num_lang1_files', type=str, nargs='+', required=True,
                        help='Paths to sent_num.pkl for lang1 languages.')
    parser.add_argument('--sent_num_lang2', type=str, required=True,
                        help='Path to sent_num.pkl for lang2.')
    parser.add_argument('--aligned_counts_file', type=str, required=True,
                        help='Path to the file containing gold standard aligned counts for each language.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the output files will be saved.')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help='List of threshold values to use.')
    parser.add_argument('--output_mode', type=str, choices=['all', 'average'], default='all',
                        help='Output mode: "all" to output results for all languages and the average, "average" to output only the averaged results.')
    args = parser.parse_args()
    return args

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

def load_sentence_alignment(alignment_file):
    alignment_pairs = []
    with open(alignment_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                id_lang1 = parts[0]
                id_lang2 = parts[1]
                margin_score = float(parts[2])
                alignment_pairs.append((id_lang1, id_lang2, margin_score))
            else:
                print(f"Invalid line in alignment file: {line.strip()}")
    return alignment_pairs

def get_doc_id(id):
    if id.startswith("b'"):
        id = id[2:-1]
    doc_id = '_'.join(id.split('_')[:-1])
    return doc_id

def calculate_alignment_docs(alignment_pairs):
    doc_pair_counts = {}
    for id_lang1, id_lang2, _ in tqdm(alignment_pairs, desc='Counting aligned sentences for doc pairs'):
        doc_id_lang1 = get_doc_id(id_lang1)
        doc_id_lang2 = get_doc_id(id_lang2)
        doc_pair = (doc_id_lang1, doc_id_lang2)
        doc_pair_counts[doc_pair] = doc_pair_counts.get(doc_pair, 0) + 1
    return doc_pair_counts

def get_doc_pair_alignment_rates(doc_pair_counts, doc_sent_num_lang1, doc_sent_num_lang2):
    doc_pair_alignment_rates = []
    for doc_pair, aligned_count in doc_pair_counts.items():
        doc_id_lang1, doc_id_lang2 = doc_pair
        num_sentences_lang1 = doc_sent_num_lang1.get(doc_id_lang1, 0)
        num_sentences_lang2 = doc_sent_num_lang2.get(doc_id_lang2, 0)
        if num_sentences_lang1 == 0 or num_sentences_lang2 == 0:
            continue
        alignment_rate = 2 * aligned_count / (num_sentences_lang1 + num_sentences_lang2)
        stats = (aligned_count, num_sentences_lang1, num_sentences_lang2, alignment_rate)
        doc_pair_alignment_rates.append((doc_pair, stats))
    return doc_pair_alignment_rates

def deduplicate_doc_pairs(doc_pair_alignment_rates):
    deduped_doc_pairs = []
    seen_docs_lang1 = set()
    seen_docs_lang2 = set()
    for doc_pair, stats in doc_pair_alignment_rates:
        doc_id_lang1, doc_id_lang2 = doc_pair
        if doc_id_lang1 not in seen_docs_lang1 and doc_id_lang2 not in seen_docs_lang2:
            deduped_doc_pairs.append((doc_pair, stats))
            seen_docs_lang1.add(doc_id_lang1)
            seen_docs_lang2.add(doc_id_lang2)
    return deduped_doc_pairs

def save_results(doc_pairs, output_file):
    with open(output_file, 'w') as f_out:
        f_out.write('doc_id_lang1\tdoc_id_lang2\taligned_sentences\tnum_sentences_lang1\tnum_sentences_lang2\talignment_rate\n')
        for doc_pair, stats in doc_pairs:
            doc_id_lang1, doc_id_lang2 = doc_pair
            aligned_count, num_sentences_lang1, num_sentences_lang2, alignment_rate = stats
            f_out.write(f'{doc_id_lang1}\t{doc_id_lang2}\t{aligned_count}\t{num_sentences_lang1}\t{num_sentences_lang2}\t{alignment_rate:.6f}\n')

def my_decode(id_lang):
    if isinstance(id_lang, bytes):
        return id_lang.decode('utf-8')
    return id_lang

def main():
    args = parse_arguments()
    langs1 = args.lang1
    lang2 = args.lang2
    alignment_files = args.alignment_files
    sent_num_lang1_files = args.sent_num_lang1_files
    sent_num_lang2_file = args.sent_num_lang2
    aligned_counts_file = args.aligned_counts_file
    output_dir = args.output_dir
    thresholds = args.thresholds
    output_mode = args.output_mode

    os.makedirs(output_dir, exist_ok=True)

    aligned_numbers = load_aligned_counts(aligned_counts_file)

    with open(sent_num_lang2_file, 'rb') as f:
        doc_sent_num_lang2 = pickle.load(f)
    doc_sent_num_lang2 = {my_decode(key): int(value) for key, value in doc_sent_num_lang2.items()}

    language_results = {}

    for lang1, alignment_file, sent_num_lang1_file in zip(langs1, alignment_files, sent_num_lang1_files):
        gold_aligned_number = aligned_numbers.get(lang1, None)
        if gold_aligned_number is None:
            print(f"No gold standard alignment number found for language {lang1}. Skipping.")
            continue

        with open(sent_num_lang1_file, 'rb') as f:
            doc_sent_num_lang1 = pickle.load(f)
        doc_sent_num_lang1 = {my_decode(key): int(value) for key, value in doc_sent_num_lang1.items()}

        alignment_pairs = load_sentence_alignment(alignment_file)
        alignment_pairs = [(my_decode(id_lang1), my_decode(id_lang2), margin_score) for id_lang1, id_lang2, margin_score in alignment_pairs]

        doc_pair_counts = calculate_alignment_docs(alignment_pairs)
        doc_pair_alignment_rates = get_doc_pair_alignment_rates(doc_pair_counts, doc_sent_num_lang1, doc_sent_num_lang2)
        doc_pair_alignment_rates.sort(key=lambda x: x[1][3], reverse=True)
        deduped_doc_pairs = deduplicate_doc_pairs(doc_pair_alignment_rates)

        evaluation_results = []

        if output_mode == 'all':
            print(f"Results for {lang1}:")
            print("Thd\tR\tP\tF1")

        for threshold in thresholds:
            filtered_doc_pairs = [(doc_pair, stats) for doc_pair, stats in deduped_doc_pairs if stats[3] >= threshold]
            num_same_doc_pairs = sum(1 for doc_pair, _ in filtered_doc_pairs if doc_pair[0] == doc_pair[1])
            total_predicted_pairs = len(filtered_doc_pairs)
            correct_predictions = num_same_doc_pairs

            precision = correct_predictions / total_predicted_pairs if total_predicted_pairs > 0 else 0
            recall = correct_predictions / gold_aligned_number if gold_aligned_number > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            evaluation_results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1_score': f1_score,
            })

            if output_mode == 'all':
                print(f"{threshold}\t{recall:.4f}\t{precision:.4f}\t{f1_score:.4f}")

        language_results[lang1] = evaluation_results

        if output_mode == 'all':
            output_file = os.path.join(output_dir, f"evaluation_eng_{lang1}.txt")
            with open(output_file, 'w') as f_out:
                f_out.write("Thd\tR\tP\tF1\n")
                for result in evaluation_results:
                    f_out.write(f"{result['threshold']}\t{result['recall']:.4f}\t{result['precision']:.4f}\t{result['f1_score']:.4f}\n")

    if len(language_results) > 0:
        num_languages = len(language_results)
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

            print(f"{threshold}\t{avg_recall:.4f}\t{avg_precision:.4f}\t{avg_f1:.4f}")

        avg_output_file = os.path.join(output_dir, "evaluation_averaged.txt")
        with open(avg_output_file, 'w') as f_out:
            f_out.write("Thd\tAvg_R\tAvg_P\tAvg_F1\n")
            for result in averaged_results:
                f_out.write(f"{result['threshold']}\t{result['avg_recall']:.4f}\t{result['avg_precision']:.4f}\t{result['avg_f1_score']:.4f}\n")

if __name__ == '__main__':
    main()