import os
import re
import csv
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from indicnlp.tokenize.sentence_tokenize import sentence_split as indic_sentence_split
from urduhack.tokenization import sentence_tokenizer as urdu_sentence_tokenize
from tqdm import tqdm

ISO_MAPPING = {
    'eng': 'en',
    'hin': 'hi',
    'mar': 'mr',
    'ben': 'bn',
    'guj': 'gu',
    'pan': 'pa',
    'ori': 'or',
    'tam': 'ta',
    'tel': 'te',
    'kan': 'kn',
    'mal': 'ml',
    'urd': 'ur'
}

def clean_text(text):
    cleaned_text = re.sub(r'[\n\r\t]+', ' ', text)  # Replace newlines, carriage returns, and tabs with space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Collapse multiple spaces into one and trim
    return cleaned_text

def tokenize_text(text, lang):
    two_letter_lang = ISO_MAPPING.get(lang, lang)
    if two_letter_lang in ['en', 'hi', 'mr', 'bn', 'gu', 'pa', 'or', 'ta', 'te', 'kn', 'ml']:
        sentences = indic_sentence_split(text, two_letter_lang)
    elif two_letter_lang == 'ur':
        sentences = urdu_sentence_tokenize(text)
    return sentences

def word_count(text):
    return len(text.split())

def filter_sentences(sentences, min_words):
    valid_sentences = []
    for sentence in sentences:
        wc = word_count(sentence)
        if wc >= min_words:
            valid_sentences.append(sentence)
    return valid_sentences

def process_file(input_file_path, doc_id, lang, min_words):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = clean_text(text)
    sentences = tokenize_text(text, lang)

    valid_sentences = filter_sentences(sentences, min_words)

    valid_records = []

    for i, sentence in enumerate(valid_sentences, 1):
        sentence_id = f"{doc_id}_S{i}"
        json_record = {
            "doc_id": doc_id,
            "sentence_id": sentence_id,
            "sentence": sentence
        }
        valid_records.append(json_record)

    return valid_records, len(valid_sentences)

def extract_prefix_and_number(doc_id):
    match = re.match(r"([a-zA-Z]+)-(\d+)", doc_id)
    if match:
        prefix, number = match.groups()
        return prefix, int(number)
    return doc_id, 0

def save_shards(records, output_lang_dir, language_code, shard_size):
    os.makedirs(output_lang_dir, exist_ok=True)

    records.sort(key=lambda record: extract_prefix_and_number(record["doc_id"]))

    for i in range(0, len(records), shard_size):
        shard_records = records[i:i + shard_size]
        shard_index = i // shard_size + 1

        output_shard_file = os.path.join(output_lang_dir, f"{language_code}_shard-{shard_index}.jsonl")

        with open(output_shard_file, 'w', encoding='utf-8') as shard_file:
            for record in shard_records:
                shard_file.write(json.dumps(record, ensure_ascii=False) + '\n')

def write_csv_summary(language, sentence_counts, output_lang_dir):
    csv_file_path = os.path.join(output_lang_dir, f"{language}_sentence_summary.csv")
    
    sorted_sentence_counts = sorted(sentence_counts.items(), key=lambda x: extract_prefix_and_number(x[0]))

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["doc_id", "sentence_count"])
        for doc_id, count in sorted_sentence_counts:
            writer.writerow([doc_id, count])

def process_batch(batch, lang, input_lang_dir, min_words):
    batch_valid_records = []
    sentence_counts = {}
    for file_name in batch:
        doc_id = os.path.splitext(file_name)[0]
        input_file_path = os.path.join(input_lang_dir, file_name)
        valid_records, sentence_count = process_file(input_file_path, doc_id, lang, min_words)
        batch_valid_records.extend(valid_records)
        sentence_counts[doc_id] = sentence_count
    return batch_valid_records, sentence_counts

def process_language(language_code, input_base_directory, output_base_directory, shard_size, batch_size, min_words):
    input_lang_dir = os.path.join(input_base_directory, language_code)
    output_lang_dir = os.path.join(output_base_directory, language_code)

    os.makedirs(output_lang_dir, exist_ok=True)

    if os.path.isdir(input_lang_dir):
        file_list = [f for f in os.listdir(input_lang_dir) if f.endswith('.txt')]
        batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]

        all_valid_records = []
        all_sentence_counts = {}

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_batch, batch, language_code, input_lang_dir, min_words): batch for batch in batches}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {language_code}', unit='batch'):
                valid_records, sentence_counts = future.result()
                all_valid_records.extend(valid_records)
                all_sentence_counts.update(sentence_counts)

        save_shards(all_valid_records, output_lang_dir, language_code, shard_size)
        write_csv_summary(language_code, all_sentence_counts, output_lang_dir)
    else:
        print(f"Skipping {language_code}: Not a directory")

def process_directory(input_base_directory, output_base_directory, languages, shard_size, batch_size, min_words):
    for language_code in languages:
        process_language(language_code, input_base_directory, output_base_directory, shard_size, batch_size, min_words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files and tokenize sentences into JSONL files by language.")
    parser.add_argument('--input_dir', type=str, required=True, help='Base directory with language subdirectories containing text files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store the valid output JSONL files.')
    parser.add_argument('--languages', type=str, nargs='+', required=True, help='List of language codes to process, e.g., eng hin mar.')
    parser.add_argument('--shard_size', type=int, default=50000, help='Number of sentences per shard.')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of files to process in a batch.')
    parser.add_argument('--min_words', type=int, default=3, help='Minimum number of words per sentence to include.')
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.languages, args.shard_size, args.batch_size, args.min_words)