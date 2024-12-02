import os
import json
import re
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[\n\r\t]+', ' ', text)).strip()

def split_sentences_with_punctuation(text):
    pattern = r'([.!?।]|[\u0964\u0965\u06D4\u0BE7\u0B83\u0C77\u0CE6\u0D79\u0D3A\u0A4D\u0AF0\u0C63\u0B56\u0970\u1CDA\uA875])'
    sentences = re.split(pattern, text)
    sentences = [''.join(sentences[i:i+2]).strip() for i in range(0, len(sentences), 2)]
    return sentences

def split_text_into_chunks(sentences, sentence_count_per_chunk=2):
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= sentence_count_per_chunk:
            chunks.append(' '.join(current_chunk).strip())
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())

    return chunks

def word_count(chunk):
    return len(chunk.split())

def filter_chunks(chunks, min_words=3):
    return [chunk for chunk in chunks if word_count(chunk) >= min_words]

def process_file(file_path, sentence_count_per_chunk=2, min_words=3):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text = clean_text(text)
    sentences = split_sentences_with_punctuation(text)
    chunks = split_text_into_chunks(sentences, sentence_count_per_chunk)
    chunks = filter_chunks(chunks, min_words)

    base_name = os.path.basename(file_path).rsplit('.', 1)[0]
    chunk_data_list = []

    for i, chunk in enumerate(chunks, start=1):
        chunk_data = {
            "doc_id": base_name,
            "chunk_id": f"{base_name}_C{i}",
            "chunk": chunk
        }
        chunk_data_list.append(chunk_data)

    return chunk_data_list

def extract_prefix_and_number(doc_id):
    match = re.match(r"([a-zA-Z]+)-(\d+)", doc_id)
    return (match.groups() if match else (doc_id, 0))

def save_shards(records, output_lang_dir, shard_size, lang_code):
    os.makedirs(output_lang_dir, exist_ok=True)
    records.sort(key=lambda record: extract_prefix_and_number(record["doc_id"]))

    for i in range(0, len(records), shard_size):
        shard_index = i // shard_size + 1
        shard_records = records[i:i + shard_size]
        shard_file = os.path.join(output_lang_dir, f"{lang_code}_shard-{shard_index}.jsonl")

        with open(shard_file, 'w', encoding='utf-8') as f:
            for record in shard_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

def process_batch(batch, sentence_count_per_chunk, min_words, input_lang_dir):
    batch_records = []
    for file_name in batch:
        file_path = os.path.join(input_lang_dir, file_name)
        chunks = process_file(file_path, sentence_count_per_chunk, min_words)
        batch_records.extend(chunks)
    return batch_records

def process_language(lang_code, input_base_dir, output_base_dir, sentence_count_per_chunk, shard_size, batch_size, min_words):
    input_lang_dir = os.path.join(input_base_dir, lang_code)
    output_lang_dir = os.path.join(output_base_dir, lang_code)

    if not os.path.isdir(input_lang_dir):
        return

    file_list = [f for f in os.listdir(input_lang_dir) if f.endswith('.txt')]
    batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]

    all_records = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_batch, batch, sentence_count_per_chunk, min_words, input_lang_dir): batch for batch in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {lang_code}', unit='batch'):
            all_records.extend(future.result())

    save_shards(all_records, output_lang_dir, shard_size, lang_code)

def process_directory(input_base_dir, output_base_dir, langs, sentence_count_per_chunk, shard_size, batch_size, min_words):
    for lang_code in langs:
        process_language(lang_code, input_base_dir, output_base_dir, sentence_count_per_chunk, shard_size, batch_size, min_words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk text files and save them into JSONL shards.")
    parser.add_argument('--input_dir', type=str, required=True, help="Base directory with language subdirectories containing text files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store the output JSONL files.")
    parser.add_argument('--languages', type=str, nargs='+', required=True, help="List of language codes to process")
    parser.add_argument('--sentspchunk', type=int, default=2, choices=[2, 4, 8], help="Number of sentences per chunk (e.g., 2, 4, or 8).")
    parser.add_argument('--shard_size', type=int, default=50000, help="Number of chunks per shard.")
    parser.add_argument('--batch_size', type=int, default=100, help="Number of files to process in a batch.")
    parser.add_argument('--min_words', type=int, default=3, help="Minimum number of words per chunk (default is 3).")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.languages, args.sentspchunk, args.shard_size, args.batch_size, args.min_words)