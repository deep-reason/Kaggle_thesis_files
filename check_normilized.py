#!/usr/bin/env python3
"""
Check if Hugging Face supervised audio dataset samples are normalized (centered at 0)
"""

import numpy as np
from datasets import load_dataset

# --- Hard-coded dataset and split ---
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"   # Replace with your dataset
SPLIT_NAME = "valid"                        # Replace with the split you want to check
AUDIO_COLUMN = "audio"

def main():
    # Load dataset
    ds = load_dataset(DATASET_NAME, split=SPLIT_NAME)

    total_samples = len(ds)
    normalized_count = 0
    min_values = []
    max_values = []
    mean_values = []

    for i, item in enumerate(ds):
        # Load raw waveform
        waveform = item[AUDIO_COLUMN]["array"]
        waveform = np.array(waveform, dtype=np.float32)

        min_values.append(np.min(waveform))
        max_values.append(np.max(waveform))
        mean_values.append(np.mean(waveform))

        # Check if waveform is roughly in [-1, 1]
        if np.max(np.abs(waveform)) <= 1.0:
            normalized_count += 1

        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{total_samples} samples...")

    print("\n--- Normalization Statistics ---")
    print(f"Total samples checked       : {total_samples}")
    print(f"Samples normalized in [-1,1]: {normalized_count}")
    print(f"Percentage normalized       : {100 * normalized_count / total_samples:.2f}%")
    print(f"Waveform min value range    : {min(min_values):.6f} to {max(min_values):.6f}")
    print(f"Waveform max value range    : {min(max_values):.6f} to {max(max_values):.6f}")
    print(f"Waveform mean value range   : {min(mean_values):.6f} to {max(mean_values):.6f}")

if __name__ == "__main__":
    main()
