#!/usr/bin/env python3
# prepare_dataset_streaming_upload_shards.py
"""
Stream a HF dataset, preprocess into shards, upload each shard parquet directly
to a Hugging Face dataset repo (one file per shard) and delete local shards
to avoid filling Kaggle disk.

Requirements:
  - datasets
  - transformers
  - huggingface_hub
  - numpy

Make sure HF_TOKEN is set in the environment (Kaggle "Secrets" or export HF_TOKEN=...).
"""

import os
import sys
import time
import gc
import json
from typing import List
from datasets import Dataset, Audio, load_dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
import numpy as np
from huggingface_hub import HfApi, Repository, hf_hub_download
from requests.exceptions import HTTPError

# ----------------------------
# Config (edit these as needed)
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"   # source dataset (HF)
DATASET_CONFIG = "default"                                         # usually "default"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"      # destination HF dataset repo id
LOCAL_SHARD_DIR = "./prepared_shards"
BATCH_SIZE = 2048           # number of examples per shard (tune to fit memory/disk)
MAX_RETRIES = 3             # upload retries
SLEEP_BETWEEN_RETRIES = 5   # seconds

# ----------------------------
# Environment / token check
# ----------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    sys.exit(
        "ERROR: HF_TOKEN not found in environment. "
        "Set HF_TOKEN (Kaggle 'Secrets' or export HF_TOKEN=...) and re-run."
    )

# ----------------------------
# Prepare local directories
# ----------------------------
os.makedirs(LOCAL_SHARD_DIR, exist_ok=True)

# ----------------------------
# Initialize HF API
# ----------------------------
api = HfApi(token=HF_TOKEN)

def list_existing_repo_files(repo_id: str) -> List[str]:
    """
    Returns list of files in the dataset repo (top-level and nested paths).
    Uses repo_type='dataset'.
    """
    try:
        return api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except HTTPError as e:
        # Helpful error if repo doesn't exist or is private
        raise RuntimeError(
            f"Failed to list files in repo '{repo_id}' (repo_type='dataset').\n"
            f"HTTPError: {e}\n"
            f"Make sure the repo exists and the token has permission to access it."
        )

# ----------------------------
# Load tokenizer & processor
# ----------------------------
print("Loading tokenizer & feature extractor...")
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
# Preprocessing (batch)
# ----------------------------
def prepare_batch(batch: dict) -> dict:
    """
    batch: {'audio': [audio_dict,...], 'transcription': [str,...]}
    returns dict with numpy arrays suitable for Dataset.from_dict
    """
    # audio arrays: some examples may come as {"array": np.ndarray, "sampling_rate": int}
    audio_arrays = []
    for a in batch["audio"]:
        if isinstance(a, dict) and "array" in a:
            audio_arrays.append(a["array"])
        else:
            # If dataset provides direct path or bytes, let processor handle it;
            # but most HF streaming audio gives dict with `array`.
            audio_arrays.append(a)

    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="np", padding=True)
    # For labels use tokenizer directly (works with list[str])
    labels = tokenizer(batch["transcription"], padding=True, return_tensors="np").input_ids

    # convert returned np arrays to python/numpy-friendly types
    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs.get("attention_mask"),
        "labels": labels,
    }

# ----------------------------
# Upload helper
# ----------------------------
def upload_file_with_retries(local_path: str, repo_id: str, path_in_repo: str, retries=MAX_RETRIES):
    """
    Upload a single local file to HF dataset repo at path_in_repo.
    Retries on failure.
    """
    attempt = 0
    while attempt < retries:
        try:
            print(f"Uploading {local_path} -> {repo_id}:{path_in_repo} (attempt {attempt+1}) ...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            print("Upload succeeded.")
            return True
        except Exception as e:
            attempt += 1
            print(f"Upload failed (attempt {attempt}): {e}")
            if attempt < retries:
                print(f"Sleeping {SLEEP_BETWEEN_RETRIES}s before retry...")
                time.sleep(SLEEP_BETWEEN_RETRIES)
            else:
                print("Max retries reached; upload failed permanently.")
                raise
    return False

# ----------------------------
# Shard naming / resume helpers
# ----------------------------
def next_shard_index_for_split(existing_files: List[str], split_name: str) -> int:
    """
    Inspect existing repo files and return the next shard index so we don't overlap names.
    We expect files under: 'train/', 'valid/', 'test/' or top-level names like 'train-00000.parquet'.
    """
    prefix = split_name.rstrip("/") + "/"
    indices = []
    for p in existing_files:
        if p.startswith(prefix) and p.endswith(".parquet"):
            fname = os.path.basename(p)
            # try to extract index from name like train-00012.parquet or train-00012-of-00050.parquet
            parts = fname.split("-")
            if len(parts) >= 2:
                idxpart = parts[1]
                # remove suffix like of/00050 and extension
                idx = idxpart.split(".")[0].split("of")[0]
                try:
                    indices.append(int(idx))
                except Exception:
                    pass
    return max(indices) + 1 if indices else 0

# ----------------------------
# Main split processing
# ----------------------------
def process_split_splitstream(split_name: str, batch_size: int = BATCH_SIZE):
    """
    Stream the split, build shards, upload each shard and delete local file after upload.
    """
    print(f"Streaming split: {split_name}")
    ds_stream = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split_name, streaming=True)
    # cast audio lazily so examples have audio dicts with array
    ds_stream = ds_stream.cast_column("audio", Audio(sampling_rate=16000))

    # compute starting shard index from existing remote files (resume support)
    try:
        existing = list_existing_repo_files(OUTPUT_DATASET_REPO)
    except RuntimeError as e:
        print(e)
        print("Assuming repo is empty / not accessible - starting shard index at 0")
        existing = []

    shard_idx = next_shard_index_for_split(existing, split_name)
    print(f"Starting shard index for {split_name}: {shard_idx}")

    batch_buffer = {"audio": [], "transcription": []}
    local_split_dir = os.path.join(LOCAL_SHARD_DIR, split_name)
    os.makedirs(local_split_dir, exist_ok=True)
    processed_count = 0

    for example in ds_stream:
        # streaming examples may sometimes be nested; ensure transcription exists
        if "transcription" not in example:
            # skip malformed examples
            continue

        batch_buffer["audio"].append(example["audio"])
        batch_buffer["transcription"].append(example["transcription"])
        processed_count += 1

        if len(batch_buffer["audio"]) >= batch_size:
            # prepare, save parquet, upload, delete local
            processed = prepare_batch(batch_buffer)
            shard_ds = Dataset.from_dict(processed)

            shard_fname = f"{split_name}-{shard_idx:05d}.parquet"
            shard_local_path = os.path.join(local_split_dir, shard_fname)

            print(f"Writing local shard: {shard_local_path} (examples: {len(batch_buffer['audio'])})")
            shard_ds.to_parquet(shard_local_path)

            # upload as e.g. "train/train-00000.parquet"
            path_in_repo = f"{split_name}/{shard_fname}"
            upload_file_with_retries(shard_local_path, OUTPUT_DATASET_REPO, path_in_repo)

            # remove local file to free disk
            try:
                os.remove(shard_local_path)
            except Exception as e:
                print("Warning: failed to remove local shard:", e)

            # free memory
            del shard_ds
            del processed
            batch_buffer = {"audio": [], "transcription": []}
            gc.collect()

            shard_idx += 1

    # leftover
    if len(batch_buffer["audio"]) > 0:
        processed = prepare_batch(batch_buffer)
        shard_ds = Dataset.from_dict(processed)

        shard_fname = f"{split_name}-{shard_idx:05d}.parquet"
        shard_local_path = os.path.join(local_split_dir, shard_fname)

        print(f"Writing final local shard: {shard_local_path} (examples: {len(batch_buffer['audio'])})")
        shard_ds.to_parquet(shard_local_path)

        path_in_repo = f"{split_name}/{shard_fname}"
        upload_file_with_retries(shard_local_path, OUTPUT_DATASET_REPO, path_in_repo)

        try:
            os.remove(shard_local_path)
        except Exception as e:
            print("Warning: failed to remove local shard:", e)

        del shard_ds
        del processed
        gc.collect()
        shard_idx += 1

    print(f"Finished streaming split {split_name}. Processed {processed_count} examples.")

# ----------------------------
# Entrypoint
# ----------------------------
def main():
    splits = ["train", "valid", "test"]
    for s in splits:
        print("\n" + "=" * 60)
        process_split_splitstream(s, batch_size=BATCH_SIZE)

    # final cleanup local dir if empty
    try:
        if os.path.isdir(LOCAL_SHARD_DIR) and not any(os.scandir(LOCAL_SHARD_DIR)):
            os.rmdir(LOCAL_SHARD_DIR)
    except Exception:
        pass

    print("\nAll splits processed and uploaded. Done.")

if __name__ == "__main__":
    main()
