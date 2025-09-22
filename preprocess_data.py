# ==========================
# Script 1: Preprocess & Push to Hub (Batch-wise, single repo)
# ==================================
import os
import gc
import logging
import tempfile
import time
from datasets import load_dataset, Audio, Dataset, disable_caching
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from huggingface_hub import HfApi, HfFolder, get_full_repo_name, CommitOperationAdd

# Disable global caching for datasets
disable_caching()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------
# Hard-coded defaults
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
PROCESSED_DATASET_REPO = "AhunInteligence/w2v-bert-2.0-amharic-preprocessed"
CHUNK_SIZE = 1024

# ----------------------------
# Load tokenizer and processor once
# ----------------------------
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_REPO)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
api = HfApi()
HF_TOKEN = os.environ.get("HF_TOKEN", HfFolder.get_token())

# ----------------------------
# Preprocessing function
# ----------------------------
def prepare_dataset_batch(examples):
    audio_arrays = [audio["array"] for audio in examples["audio"]]
    batch = {}
    batch["input_values"] = processor(audio_arrays, sampling_rate=16000, padding=True).input_values
    batch["labels"] = processor.tokenizer(examples["transcription"]).input_ids
    return batch

# ----------------------------
# Main processing logic
# ----------------------------
splits_to_process = [TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT]
repo_url = get_full_repo_name(PROCESSED_DATASET_REPO)

# Create the destination repository once before processing
try:
    api.create_repo(repo_url, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)
    logging.info(f"Successfully created or found destination repository '{repo_url}'.")
except Exception as e:
    logging.critical(f"Failed to create/find repository '{repo_url}': {e}. Terminating.")
    exit()

for split_name in splits_to_process:
    logging.info(f"\nProcessing '{split_name}' split...")
    
    # Load the original dataset split in streaming mode
    raw_dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split_name,
        streaming=True,
    ).cast_column("audio", Audio(sampling_rate=16_000))

    chunked_list = []
    chunk_count = 0

    for example in raw_dataset:
        chunked_list.append(example)
        if len(chunked_list) == CHUNK_SIZE:
            chunk_count += 1
            logging.info(f"  -> Processing chunk {chunk_count}...")
            
            chunk_dataset = Dataset.from_list(chunked_list)
            
            processed_chunk = chunk_dataset.map(
                prepare_dataset_batch,
                batched=True,
                batch_size=CHUNK_SIZE,
                remove_columns=["audio", "transcription"],
                num_proc=1, # Setting to 1 to avoid multiprocessing memory issues
            )
            
            # Use HfApi to commit the file to the repo with a unique name
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                processed_chunk.to_parquet(tmp_file.name)
                file_name_in_repo = f"{split_name}/chunk-{chunk_count:05d}.parquet"
                
                logging.info(f"Creating commit for '{file_name_in_repo}'.")
                api.create_commit(
                    repo_id=repo_url,
                    repo_type="dataset",
                    operations=[
                        CommitOperationAdd(path_in_repo=file_name_in_repo, path_or_fileobj=tmp_file.name)
                    ],
                    commit_message=f"Add processed {split_name} chunk {chunk_count}",
                    token=HF_TOKEN,
                )
                os.remove(tmp_file.name)

            del chunked_list
            del chunk_dataset
            del processed_chunk
            gc.collect()
            
            chunked_list = []

    # Process any remaining examples in the last, incomplete batch
    if chunked_list:
        chunk_count += 1
        logging.info(f"  -> Processing final chunk {chunk_count}...")
        chunk_dataset = Dataset.from_list(chunked_list)
        processed_chunk = chunk_dataset.map(
            prepare_dataset_batch,
            batched=True,
            batch_size=len(chunked_list),
            remove_columns=["audio", "transcription"],
            num_proc=1, # Setting to 1
        )
        
        # Use HfApi to commit the final file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            processed_chunk.to_parquet(tmp_file.name)
            file_name_in_repo = f"{split_name}/final-chunk-{chunk_count:05d}.parquet"
            
            logging.info(f"Creating commit for '{file_name_in_repo}'.")
            api.create_commit(
                repo_id=repo_url,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=file_name_in_repo, path_or_fileobj=tmp_file.name)
                ],
                commit_message=f"Add processed {split_name} final chunk {chunk_count}",
                token=HF_TOKEN,
            )
            os.remove(tmp_file.name)

        del chunked_list
        del chunk_dataset
        del processed_chunk
        gc.collect()

logging.info("\nAll data processing and uploading complete.")
