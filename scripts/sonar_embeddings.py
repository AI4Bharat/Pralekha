import os
import h5py
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from tqdm import tqdm
import jsonlines

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "cointegrated/SONAR_200_text_encoder"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANGUAGE_MAPPING = {
    "eng": "eng_Latn",
    "ben": "ben_Beng",
    "guj": "guj_Gujr",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
    "mal": "mal_Mlym",
    "mar": "mar_Deva",
    "ori": "ori_Orya",
    "pan": "pan_Guru",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
    "urd": "urd_Arab"
}

class JsonlDataset(Dataset):
    def __init__(self, file_path, approach):
        self.data = []
        self.approach = approach
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Sonar:
    def __init__(self, model_name="cointegrated/SONAR_200_text_encoder", gpu_list=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = M2M100Encoder.from_pretrained(model_name)

        if torch.cuda.is_available():
            if gpu_list:
                self.device_ids = [int(i) for i in gpu_list.split(',')]
                torch.cuda.set_device(self.device_ids[0])
                self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            else:
                self.device_ids = list(range(torch.cuda.device_count()))
                self.model = torch.nn.DataParallel(self.model)
            self.model.to('cuda')
        else:
            print("Using CPU for embedding.")

    def encode_mean_pool(self, texts, lang, norm=False):
        self.tokenizer.src_lang = lang
        with torch.inference_mode():
            batch = self.tokenizer(
                texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=256
            ).to(device)
            
            seq_embs = self.model(**batch).last_hidden_state
            mask = batch.attention_mask
            mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
            if norm:
                mean_emb = torch.nn.functional.normalize(mean_emb)
        return mean_emb

    def get_embeddings(self, texts, lang, norm=False):
        embeddings = self.encode_mean_pool(texts, lang=lang, norm=norm)
        return embeddings.cpu().numpy()

def process_file(input_file, output_file, model, batch_size, num_workers, lang, norm, error_log_file, approach):
    dataset = JsonlDataset(input_file, approach)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=lambda x: x)

    all_embeddings = []
    all_doc_ids = []
    all_id_fields = []

    for batch in dataloader:
        try:
            if approach == 'chunk':
                batch_texts = [item['chunk'] for item in batch]
                batch_ids = [item['chunk_id'] for item in batch]
            else:
                batch_texts = [item['sentence'] for item in batch]
                batch_ids = [item['sentence_id'] for item in batch]

            batch_embeddings = model.get_embeddings(batch_texts, lang=lang, norm=norm)
            all_embeddings.append(batch_embeddings)
            all_doc_ids.extend([item['doc_id'] for item in batch])
            all_id_fields.extend(batch_ids)
        except Exception as e:
            with open(error_log_file, 'a') as f:
                f.write(f"Failed to process batch in file {input_file} with error: {str(e)}\n")

    all_embeddings = np.vstack(all_embeddings)
    all_doc_ids = np.array(all_doc_ids, dtype='S')
    all_id_fields = np.array(all_id_fields, dtype='S')

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('embeddings', data=all_embeddings)
        f.create_dataset('doc_ids', data=all_doc_ids)
        if approach == 'chunk':
            f.create_dataset('chunk_ids', data=all_id_fields)
        else:
            f.create_dataset('sentence_ids', data=all_id_fields)

def process_directory(input_dir, output_dir, model, batch_size, norm, approach):
    num_workers = 4

    os.makedirs(output_dir, exist_ok=True)
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    error_log_file = os.path.join(output_dir, f'{approach}_failed_batches.txt')

    for filename in tqdm(jsonl_files, desc=f"Processing {approach}", unit="file"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.h5")
        process_file(input_file, output_file, model, batch_size, num_workers, approach, norm, error_log_file, approach)

def process_language_subdirectories(base_input_dir, base_output_dir, model, batch_size, norm, approach, langs):
    for lang_dir in os.listdir(base_input_dir):
        if langs and lang_dir not in langs:
            continue

        input_lang_dir = os.path.join(base_input_dir, lang_dir)
        if not os.path.isdir(input_lang_dir):
            continue

        if lang_dir not in LANGUAGE_MAPPING:
            print(f"Skipping unrecognized language directory: {lang_dir}")
            continue

        sonar_lang_code = LANGUAGE_MAPPING[lang_dir]
        output_lang_dir = os.path.join(base_output_dir, lang_dir)
        os.makedirs(output_lang_dir, exist_ok=True)

        process_directory(input_lang_dir, output_lang_dir, model, batch_size, norm, approach)

def parse_args():
    parser = argparse.ArgumentParser(description="Embed texts using the SONAR model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input JSONL files directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output HDF5 files with embeddings.")
    parser.add_argument("--batch_size", type=int, default=6144, help="Batch size for embedding.")
    parser.add_argument("--norm", action='store_true', help="Whether to normalize embeddings.")
    parser.add_argument("--gpu_list", type=str, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2').")
    parser.add_argument("--approach", type=str, choices=["chunk", "sentence"], required=True, help="Specify whether to process chunks or sentences.")
    parser.add_argument("--langs", type=str, nargs='+', help="List of language directories to process, e.g., 'eng hin mar'.")
    return parser.parse_args()

def main():
    args = parse_args()

    model = Sonar(model_name=model_name, gpu_list=args.gpu_list)
    process_language_subdirectories(args.input_dir, args.output_dir, model, args.batch_size, args.norm, args.approach, args.langs)

if __name__ == "__main__":
    main()