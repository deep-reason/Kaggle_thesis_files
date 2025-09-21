# Install necessary libraries
# !pip install -q transformers datasets soundfile accelerate huggingface_hub resampy torchaudio

import os
import io
import time
import json
import logging
import numpy as np
import torch
from functools import partial
from datasets import Dataset, Audio, load_dataset
from huggingface_hub import HfApi, HfFolder, get_full_repo_name, create_repo, CommitOperationAdd
import soundfile as sf
from torchaudio.transforms import Resample
import shutil
import traceback
import tempfile

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_FILE_PATH = "/kaggle/working/processing_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'),
        logging.StreamHandler()
    ]
)

# -----------------------------------------------------------------------------
# Configuration & Credentials
# -----------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", HfFolder.get_token())
if not HF_TOKEN:
    logging.critical("HF_TOKEN environment variable is not set. Please set your Hugging Face token as a Kaggle Secret.")
    raise ValueError("HF_TOKEN environment variable is not set. Please set your Hugging Face token as a Kaggle Secret.")

TARGET_SR = 16000
STATE_FILE = "/kaggle/working/processed_samples.json"

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
def load_processed_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Failed to load state file: {e}. Starting fresh.")
    return {}

def save_processed_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

# -----------------------------------------------------------------------------
# Processing Functions
# -----------------------------------------------------------------------------

def is_normalized(sample, threshold=1e-6):
    audio_data = sample['audio']['array']
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    if abs(np.mean(audio_data)) > threshold or not np.isclose(np.max(np.abs(audio_data)), 1.0, atol=threshold):
        return False
    return True

def is_correct_sampling_rate(sample):
    return sample['audio']['sampling_rate'] == TARGET_SR

def normalize_audio_sample(sample):
    audio_data = sample['audio']['array']
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data - np.mean(audio_data)
    peak_amp = np.max(np.abs(audio_data))
    if peak_amp > 1e-6:
        audio_data = audio_data / peak_amp
    sample['audio']['array'] = audio_data
    return sample

def resample_audio_sample(sample):
    audio = sample['audio']
    audio_data = audio['array']
    current_sr = audio['sampling_rate']
    if current_sr != TARGET_SR:
        resampler = Resample(orig_freq=current_sr, new_freq=TARGET_SR)
        audio_tensor = torch.from_numpy(audio_data).float()
        audio_data = resampler(audio_tensor).numpy()
    sample['audio']['array'] = audio_data
    sample['audio']['sampling_rate'] = TARGET_SR
    return sample

# -----------------------------------------------------------------------------
# Main Orchestration Logic
# -----------------------------------------------------------------------------
def process_and_push_dataset(src_dataset_name, full_dest_repo_name, num_workers=4, batch_size=500):
    api = HfApi()
    splits = ['train', 'validation', 'test']
    state = load_processed_state()
    
    for split_name in splits:
        processed_count = state.get(split_name, 0)
        logging.info(f"\n--- Starting processing for split: '{split_name}' ---")
        logging.info(f"Resuming from sample {processed_count}.")
        
        try:
            ds = load_dataset(src_dataset_name, split=split_name, streaming=True)
            ds_non_stream = load_dataset(src_dataset_name, split=split_name)
            total_samples = len(ds_non_stream)
        except Exception as e:
            logging.error(f"Failed to load dataset split '{split_name}': {e}")
            continue
        
        logging.info(f"Loaded '{split_name}' split with {total_samples} samples.")
        logging.info(f"Starting processing and pushing to new repository for '{split_name}'...")
        ds_iterator = iter(ds)
        for _ in range(processed_count):
            try:
                next(ds_iterator)
            except StopIteration:
                logging.info(f"All samples for '{split_name}' already processed in previous runs.")
                processed_count = total_samples
                break
        start_time = time.time()
        while processed_count < total_samples:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(ds_iterator))
            except StopIteration:
                pass
            if not batch:
                break
            processed_batch_list = []
            for sample in batch:
                if not is_normalized(sample):
                    sample = normalize_audio_sample(sample)
                if not is_correct_sampling_rate(sample):
                    sample = resample_audio_sample(sample)
                # We now keep the numpy array directly
                processed_batch_list.append({
                    'audio': sample['audio'],
                    'transcription': sample['transcription']
                })
            # Create the dataset with the audio column containing the arrays
            processed_batch_ds = Dataset.from_list(processed_batch_list)
            
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                processed_batch_ds.to_parquet(tmp_file.name)
                
                batch_number = processed_count // batch_size
                file_name = f"{split_name}/batch-{batch_number:05d}.parquet"
                
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        logging.info(f"Creating commit for '{file_name}'. Attempt {attempt + 1}/{max_retries}.")
                        api.create_commit(
                            repo_id=full_dest_repo_name,
                            repo_type="dataset",
                            operations=[
                                CommitOperationAdd(path_in_repo=file_name, path_or_fileobj=tmp_file.name)
                            ],
                            commit_message=f"Add processed batch to {split_name}",
                            token=HF_TOKEN,
                        )
                        # Success: break from the retry loop
                        break
                    except Exception as e:
                        logging.error(f"Commit failed on attempt {attempt + 1}/{max_retries}: {e}")
                        if attempt < max_retries - 1:
                            wait_time = 10 * (2 ** attempt)
                            logging.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logging.critical(f"Failed to commit after {max_retries} attempts. Terminating.")
                            exit()
            
            os.remove(tmp_file.name)
            
            processed_count += len(processed_batch_list)
            state[split_name] = processed_count
            save_processed_state(state)
            
            logging.info(f"✅ Batch pushed for '{split_name}'. Progress: {processed_count}/{total_samples} samples.")

        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"📊 Processing for '{split_name}' completed. Total time: {total_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    SRC_DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
    DEST_REPO_NAME = "w2v-bert-2.0-finetuning-amharic-cleaned"
    
    api = HfApi()
    repo_url = get_full_repo_name(DEST_REPO_NAME)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            api.create_repo(repo_url, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)
            logging.info(f"Successfully created or found destination repository '{repo_url}' on attempt {attempt + 1}.")
            break
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries} to create repository failed: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logging.critical(f"Failed to create/find repository '{DEST_REPO_NAME}' after {max_retries} attempts. Terminating.")
                exit()
    
    process_and_push_dataset(
        src_dataset_name=SRC_DATASET_NAME,
        full_dest_repo_name=repo_url,
        num_workers=os.cpu_count(),
        batch_size=8192
    )
