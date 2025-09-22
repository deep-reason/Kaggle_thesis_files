# ==========================
# Kaggle-friendly prepare_dataset.py
# ==========================
import os
import shutil
import gc
from datasets import load_dataset, Dataset
from datasets import Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import numpy as np

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
LOCAL_OUTPUT_DIR = "./prepared_dataset"

BATCH_SIZE = 1024  # safe for Kaggle RAM

os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load tokenizer/processor
# ----------------------------
print("Loading tokenizer/feature extractor...")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_REPO)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# ----------------------------
# Prepare batch
# ----------------------------
def prepare_batch(batch):
    # batch: {"audio": [...], "transcription": [...]}
    audio_arrays = []
    for a in batch["audio"]:
        if isinstance(a, dict) and "array" in a:
            audio_arrays.append(a["array"])
        else:
            # Already numpy array
            audio_arrays.append(a)
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="np", padding=True)
    labels = tokenizer(batch["transcription"], padding=True, return_tensors="np").input_ids
    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }

# ----------------------------
# Process a single parquet file
# ----------------------------
def process_parquet_file(file_path, split_name, shard_start_idx=0):
    print(f"Processing file: {file_path}")
    ds = Dataset.from_parquet(file_path)

    batch_buffer = {"audio": [], "transcription": []}
    shard_idx = shard_start_idx

    split_dir = os.path.join(LOCAL_OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for ex in ds:
        batch_buffer["audio"].append(ex["audio"])
        batch_buffer["transcription"].append(ex["transcription"])

        if len(batch_buffer["audio"]) >= BATCH_SIZE:
            processed = prepare_batch(batch_buffer)
            shard_ds = Dataset.from_dict(processed)

            shard_file = os.path.join(split_dir, f"{split_name}-{shard_idx:05d}.parquet")
            shard_ds.to_parquet(shard_file)
            print(f"Saved shard: {shard_file}")

            # Push immediately to HF
            shard_ds.push_to_hub(repo_id=OUTPUT_DATASET_REPO, private=False, token=True)
            print(f"Pushed shard {shard_idx} to HF.")

            # Free RAM/disk
            del shard_ds, processed, batch_buffer
            gc.collect()
            batch_buffer = {"audio": [], "transcription": []}
            shard_idx += 1

    # Process remaining examples
    if len(batch_buffer["audio"]) > 0:
        processed = prepare_batch(batch_buffer)
        shard_ds = Dataset.from_dict(processed)
        shard_file = os.path.join(split_dir, f"{split_name}-{shard_idx:05d}.parquet")
        shard_ds.to_parquet(shard_file)
        print(f"Saved final shard: {shard_file}")

        # Push to HF
        shard_ds.push_to_hub(repo_id=OUTPUT_DATASET_REPO, private=False, token=True)
        print(f"Pushed final shard {shard_idx} to HF.")

        del shard_ds, processed, batch_buffer
        gc.collect()
        shard_idx += 1

    return shard_idx  # return next shard index

# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading source HF dataset (streaming to get parquet files)...")
    streaming_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)

    # Only need filenames, 1-by-1
    data_files = streaming_dataset.files  # list of parquet paths

    for split_name in ["train", "valid", "test"]:
        shard_idx = 0
        split_dir = os.path.join(LOCAL_OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"Processing split: {split_name} ...")
        # Get all parquet files for this split
        parquet_files = [f for f in data_files if split_name in f and f.endswith(".parquet")]

        for pf in parquet_files:
            shard_idx = process_parquet_file(pf, split_name, shard_start_idx=shard_idx)

        # Optional: clean local split dir after push
        shutil.rmtree(split_dir, ignore_errors=True)
        gc.collect()
        print(f"✅ Finished split: {split_name}")

    print("✅ All splits processed and pushed!")

if __name__ == "__main__":
    main()
