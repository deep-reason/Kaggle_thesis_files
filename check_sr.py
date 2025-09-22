#!/usr/bin/env python3
"""
check_sampling_rate_hardcoded.py

Checks all audio samples in a Hugging Face dataset for a target sampling rate
and prints statistics at the end.
"""

from collections import Counter
from datasets import load_dataset
import torchaudio

# ====== HARD-CODED SETTINGS ======
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"   # Replace with your dataset
SPLIT = "valid"                        # Replace with the split you want to check
AUDIO_COLUMN = "audio"                 # Replace if your audio column has a different name
TARGET_SR = 16000
MAX_SAMPLES = None                     # Set to an integer for quick tests

# ====== FUNCTION ======
def check_sampling_rate():
    print(f"Loading dataset '{DATASET_NAME}' split '{SPLIT}'...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    total = 0
    sr_counter = Counter()
    mismatched_samples = []

    print("Checking audio sampling rates...")
    for i, row in enumerate(ds):
        if MAX_SAMPLES and i >= MAX_SAMPLES:
            break

        # Handle HF Audio type or file path
        audio = row[AUDIO_COLUMN]
        if isinstance(audio, dict):
            # HF Audio object: {'array': array([...]), 'sampling_rate': 16000}
            sr = audio["sampling_rate"]
        else:
            # Fallback: assume it's a path
            waveform, sr = torchaudio.load(audio)

        sr_counter[sr] += 1
        if sr != TARGET_SR:
            mismatched_samples.append(i)
        
        total += 1
        if (i + 1) % 1000 == 0:
            print(f"Checked {i+1} samples...")

    print("\n=== Sampling Rate Statistics ===")
    print(f"Total samples checked: {total}")
    print("Sampling rates found:", dict(sr_counter))
    print(f"Samples not matching target ({TARGET_SR} Hz): {len(mismatched_samples)}")
    if mismatched_samples:
        print("Indices of mismatched samples (first 20):", mismatched_samples[:20])

    return sr_counter, mismatched_samples

# ====== MAIN ======
if __name__ == "__main__":
    check_sampling_rate()
