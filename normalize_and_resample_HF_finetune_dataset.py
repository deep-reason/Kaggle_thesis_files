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
    """Loads the processing state from a JSON file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Failed to load state file: {e}. Starting fresh.")
    return {}

def save_processed_state(state):
    """Saves the current processing state to a JSON file."""
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
        
        # New: Use a list to store the files to be uploaded in a single commit
        upload_files = []
        
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

                processed_batch_list.append({
                    'audio_path': str(sample['audio']['path']) if sample['audio']['path'] is not None else "",
                    'transcription': sample['transcription']
                })
            
            processed_batch_ds = Dataset.from_list(processed_batch_list)
            
            # --- New Logic: Save to local file and add to upload list ---
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                processed_batch_ds.to_parquet(tmp_file.name)
                
                batch_number = processed_count // batch_size
                file_name = f"{split_name}/batch-{batch_number:05d}.parquet"
                
                # Add the temporary file to the list of files to be uploaded
                upload_files.append((tmp_file.name, file_name))
            
            # Check if we should push a commit
            # This is a key change: we'll commit every N batches (e.g., every 5 batches)
            commit_frequency = 5
            
            if (batch_number + 1) % commit_frequency == 0 or processed_count + len(processed_batch_list) >= total_samples:
                # Group all files into a single commit
                logging.info(f"Creating commit with {len(upload_files)} files for {split_name} split.")
                api.create_commit(
                    repo_id=full_dest_repo_name,
                    repo_type="dataset",
                    operations=[
                        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
                        for local_path, path_in_repo in upload_files
                    ],
                    commit_message=f"Add batches to {split_name}",
                    token=HF_TOKEN,
                )
                
                # Cleanup and reset for the next commit
                for local_path, _ in upload_files:
                    os.remove(local_path)
                upload_files = []
                
                logging.info(f"✅ Commit pushed for '{split_name}'. Progress: {processed_count + len(processed_batch_list)}/{total_samples} samples.")

            processed_count += len(processed_batch_list)
            state[split_name] = processed_count
            save_processed_state(state) 
            
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"📊 Processing for '{split_name}' completed. Total time: {total_time:.2f} seconds")

    logging.info("\n\n✅ All dataset splits processed successfully.")
    logging.info("The new dataset on the Hugging Face Hub is now normalized and at 16kHz.")

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
        batch_size=15000
    )
