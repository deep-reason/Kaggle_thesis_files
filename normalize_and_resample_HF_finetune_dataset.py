# Install necessary libraries
# !pip install -q transformers datasets soundfile accelerate huggingface_hub resampy torchaudio

import os
import io
import time
import logging
import numpy as np
import torch
from functools import partial
from datasets import Dataset, Audio, load_dataset, concatenate_datasets, DatasetDict
from huggingface_hub import HfApi, HfFolder, get_full_repo_name, create_repo
import soundfile as sf
from torchaudio.transforms import Resample
import shutil
import traceback

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

# ... (rest of your imports and functions) ...

# -----------------------------------------------------------------------------
# Main Orchestration Logic
# -----------------------------------------------------------------------------

def process_and_push_dataset(src_dataset_name, dest_repo_name, num_workers=4, batch_size=500):
    api = HfApi()
    repo_url = get_full_repo_name(dest_repo_name)
    create_repo(repo_url, private=False, exist_ok=True, token=HF_TOKEN)
    logging.info(f"Successfully created or found destination repository '{repo_url}'.")

    splits = ['train', 'validation', 'test']
    
    for split_name in splits:
        logging.info(f"\n--- Starting processing for split: '{split_name}' ---")
        try:
            ds = load_dataset(src_dataset_name, split=split_name, streaming=True)
            ds_non_stream = load_dataset(src_dataset_name, split=split_name)
            total_samples = len(ds_non_stream)
        except Exception as e:
            logging.error(f"Failed to load dataset split '{split_name}': {e}")
            continue

        logging.info(f"Loaded '{split_name}' split with {total_samples} samples.")
        logging.info(f"Starting processing and pushing to new repository for '{split_name}'...")
        start_time = time.time()
        
        # Use an iterator over the streaming dataset
        ds_iterator = iter(ds)
        total_processed = 0

        while True:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(ds_iterator))
            except StopIteration:
                # End of dataset
                pass
            
            if not batch:
                break

            processed_batch_list = []
            for sample in batch:
                if not is_normalized(sample):
                    sample = normalize_audio_sample(sample)
                
                if not is_correct_sampling_rate(sample):
                    sample = resample_audio_sample(sample)
                
                processed_batch_list.append(sample)
            
            # Convert the list of samples to a Dataset object
            processed_batch_ds = Dataset.from_list(processed_batch_list)
            
            # Push the batch to the Hub with a specific split
            processed_batch_ds.push_to_hub(
                dest_repo_name,
                split=split_name,
                commit_message=f"Add processed batch to {split_name}",
                private=False,
                token=HF_TOKEN,
                # The `append=True` argument is no longer needed here.
            )
            total_processed += len(processed_batch_list)
            logging.info(f"✅ Batch pushed for '{split_name}'. Progress: {total_processed}/{total_samples} samples.")

        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"📊 Processing for '{split_name}' completed. Total time: {total_time:.2f} seconds")
    
    logging.info("\n\n✅ All dataset splits processed successfully.")
    logging.info("The new dataset on the Hugging Face Hub is now normalized and at 16kHz.")

if __name__ == "__main__":
    SRC_DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
    DEST_REPO_NAME = "w2v-bert-2.0-finetuning-amharic-processed"

    # Removed the src_subset_name argument from the function call
    process_and_push_dataset(
        src_dataset_name=SRC_DATASET_NAME,
        dest_repo_name=DEST_REPO_NAME,
        num_workers=os.cpu_count(),
        batch_size=500
    )
