# Install necessary libraries
# !pip install -q transformers datasets soundfile accelerate huggingface_hub resampy torchaudio

import os
import io
import time
import logging
import numpy as np
import torch
from functools import partial
from datasets import Dataset, Audio, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, HfFolder, get_full_repo_name, create_repo
import soundfile as sf
from torchaudio.transforms import Resample
import shutil
import traceback

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

LOG_FILE_PATH = "/kaggle/working/processing_log.txt"

# Set up logging to a file and the console
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

def process_and_push_dataset(src_dataset_name, src_subset_name, dest_repo_name, num_workers=4, batch_size=500):
    api = HfApi()
    repo_url = get_full_repo_name(dest_repo_name)

    create_repo(repo_url, private=False, exist_ok=True, token=HF_TOKEN)
    logging.info(f"Successfully created or found destination repository '{repo_url}'.")

    ds = load_dataset(src_dataset_name, src_subset_name, split='train', streaming=True)
    ds_non_stream = load_dataset(src_dataset_name, src_subset_name, split='train')
    total_samples = len(ds_non_stream)
    
    logging.info(f"Loaded dataset with {total_samples} samples.")
    logging.info("Starting processing and pushing to new repository...")
    start_time = time.time()
    batch_counter = 0
    total_processed = 0
    current_batch_list = []
    
    for sample in ds:
        if not is_normalized(sample):
            sample = normalize_audio_sample(sample)
        
        if not is_correct_sampling_rate(sample):
            sample = resample_audio_sample(sample)

        current_batch_list.append(sample)
        total_processed += 1

        if len(current_batch_list) >= batch_size:
            processed_batch = Dataset.from_list(current_batch_list)
            processed_batch.push_to_hub(
                dest_repo_name,
                split='train',
                commit_message=f"Add processed batch {batch_counter}",
                private=False,
                token=HF_TOKEN,
                append=True
            )
            logging.info(f"✅ Batch {batch_counter} pushed. Progress: {total_processed}/{total_samples} samples.")
            current_batch_list = []
            batch_counter += 1

    if current_batch_list:
        processed_batch = Dataset.from_list(current_batch_list)
        processed_batch.push_to_hub(
            dest_repo_name,
            split='train',
            commit_message=f"Add final processed batch {batch_counter}",
            private=False,
            token=HF_TOKEN,
            append=True
        )
        logging.info(f"✅ Final batch pushed. Progress: {total_processed}/{total_samples} samples.")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info("\n\n📊 **Final Statistics Summary**")
    logging.info("-----------------------------------")
    logging.info(f"Total time elapsed: {total_time:.2f} seconds")
    logging.info("Processing complete. The new dataset on the Hugging Face Hub is now normalized and at 16kHz.")

# Example usage
if __name__ == "__main__":
    SRC_DATASET_NAME = "Helsinki-NLP/opus-100"
    SRC_SUBSET_NAME = "en-am"
    DEST_REPO_NAME = "your-username/my-fully-processed-amharic-dataset"

    process_and_push_dataset(
        src_dataset_name=SRC_DATASET_NAME,
        src_subset_name=SRC_SUBSET_NAME,
        dest_repo_name=DEST_REPO_NAME,
        num_workers=os.cpu_count(),
        batch_size=500
    )
